import json
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "config" / "portfolio_config.json"

ENSEMBLE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ensemble_signals.csv"
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
    / "risk_allocated_predictions.csv"
)


class DynamicRiskAllocator:

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

        self.max_positions = int(
            self.config["max_positions"]
        )

        self.min_weight = float(
            self.config["min_position_weight"]
        )

        self.max_weight = float(
            self.config["max_position_weight"]
        )

    def load_data(self):

        ensemble = pd.read_csv(
            ENSEMBLE_PATH
        )

        features = pd.read_csv(
            FEATURES_PATH,
            usecols=[
                "Date",
                "Symbol",
                "Volatility_20"
            ]
        )

        ensemble["Date"] = pd.to_datetime(
            ensemble["Date"]
        )

        features["Date"] = pd.to_datetime(
            features["Date"]
        )

        ensemble["Symbol"] = (
            ensemble["Symbol"]
            .astype(str)
            .str.upper()
        )

        features["Symbol"] = (
            features["Symbol"]
            .astype(str)
            .str.upper()
        )

        data = ensemble.merge(
            features,
            on=["Date", "Symbol"],
            how="left",
            suffixes=("", "_feature")
        )

        return data

    def allocate(self, data):

        data = data.copy()

        data["Volatility_20"] = (
            data["Volatility_20"]
            .fillna(
                data["Volatility_20"]
                .median()
            )
        )

        data["Risk_Adjusted_Score"] = (
            data["Ensemble_Score"]
            /
            data["Volatility_20"]
            .clip(lower=0.005)
        )

        data["Rank"] = (
            data
            .groupby("Date")[
                "Risk_Adjusted_Score"
            ]
            .rank(
                ascending=False,
                method="first"
            )
        )

        data = data[
            data["Rank"] <= self.max_positions
        ].copy()

        def calculate_weights(group):

            scores = (
                group["Risk_Adjusted_Score"]
                .clip(lower=0)
            )

            if scores.sum() <= 0:

                weights = np.ones(
                    len(group)
                ) / len(group)

            else:

                weights = (
                    scores / scores.sum()
                ).to_numpy()

            weights = np.clip(
                weights,
                self.min_weight,
                self.max_weight
            )

            weights = (
                weights / weights.sum()
            )

            return pd.Series(
                weights,
                index=group.index
            )

        data["Position_Weight"] = (
            data
            .groupby("Date", group_keys=False)
            .apply(
                calculate_weights,
                include_groups=False
            )
            .sort_index()
        )

        data["Position_Weight"] = (
            data["Position_Weight"]
            .fillna(0)
        )

        data["Portfolio_Exposure"] = 1.0

        data["Allocation_Rank"] = (
            data
            .groupby("Date")[
                "Position_Weight"
            ]
            .rank(
                ascending=False,
                method="first"
            )
        )

        return data

    def run(self):

        print("=" * 70)
        print(
            "QUANTFORGE DYNAMIC RISK ALLOCATOR"
        )
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        print(
            f"Maximum Positions: "
            f"{self.max_positions}"
        )

        print(
            f"Weight Range: "
            f"{self.min_weight:.2%} - "
            f"{self.max_weight:.2%}"
        )

        data = self.load_data()

        if data.empty:

            raise RuntimeError(
                "No ensemble data available"
            )

        data = self.allocate(
            data
        )

        data = data.sort_values(
            ["Date", "Rank"]
        )

        data.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print()
        print("=" * 70)
        print(
            "RISK ALLOCATION COMPLETED"
        )
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        print(
            f"Symbols Used: "
            f"{data['Symbol'].nunique()}"
        )

        print(
            f"Trading Dates: "
            f"{data['Date'].nunique()}"
        )

        print(
            f"Position Rows: "
            f"{len(data)}"
        )

        print(
            f"Average Positions/Day: "
            f"{data.groupby('Date').size().mean():.2f}"
        )

        print(
            f"Output: {OUTPUT_PATH}"
        )

        print("=" * 70)


if __name__ == "__main__":

    engine = DynamicRiskAllocator()

    engine.run()