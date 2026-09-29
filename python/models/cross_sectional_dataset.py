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

FEATURE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "historical_features.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cross_sectional_dataset.csv"
)


class CrossSectionalDataset:

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

    def load_features(self):

        if not FEATURE_PATH.exists():

            raise FileNotFoundError(
                f"Feature file not found: {FEATURE_PATH}"
            )

        data = pd.read_csv(
            FEATURE_PATH
        )

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        data = data.dropna(
            subset=["Date", "Symbol"]
        )

        data["Symbol"] = (
            data["Symbol"]
            .astype(str)
            .str.upper()
        )

        return data

    def filter_universe(self, data):

        if self.universe_type == "CUSTOM":

            if not self.custom_universe:

                raise ValueError(
                    "Custom universe is empty"
                )

            data = data[
                data["Symbol"].isin(
                    self.custom_universe
                )
            ]

        return data.copy()

    def create_target(self, data):

        data = data.sort_values(
            ["Symbol", "Date"]
        ).copy()

        data["Future_Return_20D"] = (
            data
            .groupby("Symbol")["Return_1D"]
            .shift(-20)
        )

        data["CrossSectional_Return"] = (
            data
            .groupby("Date")["Future_Return_20D"]
            .transform("mean")
        )

        data["Excess_Return"] = (
            data["Future_Return_20D"]
            - data["CrossSectional_Return"]
        )

        data["Alpha_Rank"] = (
            data
            .groupby("Date")["Future_Return_20D"]
            .rank(
                pct=True,
                method="average"
            )
        )

        data["Alpha_Target"] = (
            data["Alpha_Rank"] >= 0.60
        ).astype(int)

        return data

    def clean_dataset(self, data):

        feature_columns = [
            "Return_1D",
            "Return_5D",
            "Return_10D",
            "Return_20D",
            "Return_60D",
            "RSI",
            "MACD",
            "MACD_Histogram",
            "Volatility_10",
            "Volatility_20",
            "Volatility_60",
            "Volume_Ratio",
            "Price_vs_SMA20",
            "Price_vs_SMA50",
            "Price_vs_SMA200",
            "SMA50_vs_SMA200",
            "Momentum_20",
            "Momentum_60",
            "Distance_From_High",
            "Distance_From_Low",
            "Price_ZScore_20",
            "Drawdown_60",
            "Trend_Strength",
            "High_Low_Range"
        ]

        required_columns = (
            feature_columns
            + [
                "Future_Return_20D",
                "CrossSectional_Return",
                "Excess_Return",
                "Alpha_Rank",
                "Alpha_Target"
            ]
        )

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna(
            subset=required_columns
        )

        return data

    def build(self):

        print("=" * 70)
        print(
            "QUANTFORGE DYNAMIC "
            "CROSS-SECTIONAL DATASET"
        )
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        data = self.load_features()

        data = self.filter_universe(
            data
        )

        print(
            f"Available Symbols: "
            f"{data['Symbol'].nunique()}"
        )

        data = self.create_target(
            data
        )

        data = self.clean_dataset(
            data
        )

        data = data.sort_values(
            ["Date", "Symbol"]
        )

        data.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print()
        print("=" * 70)
        print(
            "CROSS-SECTIONAL DATASET "
            "COMPLETED"
        )
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        print(
            f"Symbols: "
            f"{data['Symbol'].nunique()}"
        )

        print(
            f"Rows: {len(data)}"
        )

        print(
            f"Features: 24"
        )

        print(
            f"Alpha Target = 1: "
            f"{int(data['Alpha_Target'].sum())}"
        )

        print(
            f"Alpha Target = 0: "
            f"{int((data['Alpha_Target'] == 0).sum())}"
        )

        print(
            f"Date Range: "
            f"{data['Date'].min().date()} → "
            f"{data['Date'].max().date()}"
        )

        print(
            f"Output: {OUTPUT_PATH}"
        )

        print("=" * 70)


if __name__ == "__main__":

    engine = CrossSectionalDataset()

    engine.build()