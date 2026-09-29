import json
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    BASE_DIR
    / "config"
    / "portfolio_config.json"
)

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_predictions.csv"
)

FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "historical_features.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ensemble_signals.csv"
)


class SignalEnsemble:

    def __init__(self):

        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            self.config = json.load(file)

        self.universe_type = (
            self.config["universe_type"]
            .upper()
        )

        self.custom_universe = [
            symbol.upper()
            for symbol in self.config.get(
                "custom_universe",
                []
            )
        ]

    def load_data(self):

        predictions = pd.read_csv(
            PREDICTIONS_PATH
        )

        features = pd.read_csv(
            FEATURES_PATH
        )

        predictions["Date"] = pd.to_datetime(
            predictions["Date"]
        )

        features["Date"] = pd.to_datetime(
            features["Date"]
        )

        predictions["Symbol"] = (
            predictions["Symbol"]
            .astype(str)
            .str.upper()
        )

        features["Symbol"] = (
            features["Symbol"]
            .astype(str)
            .str.upper()
        )

        if self.universe_type == "CUSTOM":

            predictions = predictions[
                predictions["Symbol"].isin(
                    self.custom_universe
                )
            ]

            features = features[
                features["Symbol"].isin(
                    self.custom_universe
                )
            ]

        data = predictions.merge(
            features,
            on=["Date", "Symbol"],
            how="left",
            suffixes=("", "_feature")
        )

        return data

    def percentile_rank(self, series):

        return series.rank(
            pct=True,
            method="average"
        )

    def build(self):

        print("=" * 70)
        print(
            "QUANTFORGE DYNAMIC SIGNAL ENSEMBLE"
        )
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        data = self.load_data()

        if data.empty:

            raise RuntimeError(
                "No data available for ensemble"
            )

        data["ML_Score"] = (
            data["Predicted_Alpha"]
            .clip(0, 1)
        )

        data["Momentum_20_Rank"] = (
            data
            .groupby("Date")["Momentum_20"]
            .transform(
                self.percentile_rank
            )
        )

        data["Momentum_60_Rank"] = (
            data
            .groupby("Date")["Momentum_60"]
            .transform(
                self.percentile_rank
            )
        )

        data["Momentum_Score"] = (
            0.60
            * data["Momentum_20_Rank"]
            + 0.40
            * data["Momentum_60_Rank"]
        )

        data["SMA20_Rank"] = (
            data
            .groupby("Date")["Price_vs_SMA20"]
            .transform(
                self.percentile_rank
            )
        )

        data["SMA50_Rank"] = (
            data
            .groupby("Date")["Price_vs_SMA50"]
            .transform(
                self.percentile_rank
            )
        )

        data["Trend_Rank"] = (
            data
            .groupby("Date")["Trend_Strength"]
            .transform(
                self.percentile_rank
            )
        )

        data["Volume_Rank"] = (
            data
            .groupby("Date")["Volume_Ratio"]
            .transform(
                self.percentile_rank
            )
        )

        data["Trend_Score"] = (
            0.40
            * data["SMA20_Rank"]
            + 0.40
            * data["SMA50_Rank"]
            + 0.20
            * data["Trend_Rank"]
        )

        data["Technical_Score"] = (
            0.50
            * data["Momentum_Score"]
            + 0.30
            * data["Trend_Score"]
            + 0.20
            * data["Volume_Rank"]
        )

        volatility = (
            data["Volatility_20"]
            .clip(lower=0)
        )

        data["Volatility_Penalty"] = (
            1
            / (
                1
                + volatility
            )
        )

        data["Ensemble_Score"] = (
            0.55
            * data["ML_Score"]
            + 0.30
            * data["Technical_Score"]
            + 0.15
            * data["Volatility_Penalty"]
        )

        data["Ensemble_Rank"] = (
            data
            .groupby("Date")[
                "Ensemble_Score"
            ]
            .rank(
                pct=True,
                method="average"
            )
        )

        data["Ensemble_Signal"] = np.select(
            [
                data["Ensemble_Rank"] >= 0.80,
                data["Ensemble_Rank"] >= 0.60,
                data["Ensemble_Rank"] >= 0.40,
                data["Ensemble_Rank"] >= 0.20
            ],
            [
                "STRONG_BUY",
                "BUY",
                "WATCH",
                "SELL"
            ],
            default="STRONG_SELL"
        )

        output_columns = [
            "Date",
            "Symbol",
            "Return_1D",
            "Predicted_Alpha",
            "Prediction_Rank",
            "ML_Score",
            "Momentum_Score",
            "Trend_Score",
            "Technical_Score",
            "Volatility_20",
            "Volatility_Penalty",
            "Ensemble_Score",
            "Ensemble_Rank",
            "Ensemble_Signal"
        ]

        output = data[
            output_columns
        ].copy()

        output = output.replace(
            [np.inf, -np.inf],
            np.nan
        )

        output = output.dropna(
            subset=[
                "Ensemble_Score",
                "Ensemble_Rank"
            ]
        )

        output = output.sort_values(
            ["Date", "Ensemble_Rank"],
            ascending=[
                True,
                False
            ]
        )

        output.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print()
        print("=" * 70)
        print(
            "SIGNAL ENSEMBLE COMPLETED"
        )
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        print(
            f"Symbols: "
            f"{output['Symbol'].nunique()}"
        )

        print(
            f"Rows: {len(output)}"
        )

        print(
            f"Dates: "
            f"{output['Date'].nunique()}"
        )

        print(
            "Signal Distribution:"
        )

        print(
            output["Ensemble_Signal"]
            .value_counts()
            .to_string()
        )

        print(
            f"Output: {OUTPUT_PATH}"
        )

        print("=" * 70)


if __name__ == "__main__":

    engine = SignalEnsemble()

    engine.build()