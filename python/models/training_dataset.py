import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class TrainingDatasetBuilder:

    def __init__(self, config_path):

        self.config_path = Path(config_path)

        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.symbols = self.config["universe"]

        self.processed_path = (
            self.config_path.parent.parent
            / "data"
            / "processed"
        )

    def load_features(self, symbol):

        filename = (
            symbol.replace(".", "_")
            + "_features.csv"
        )

        filepath = self.processed_path / filename

        if not filepath.exists():
            return None

        return pd.read_csv(filepath)

    def build_dataset(self, data, symbol):

        data = data.copy()

        data["Future_Return_5"] = (
            data["Close"].shift(-5)
            / data["Close"]
            - 1
        )

        data["Target"] = (
            data["Future_Return_5"] > 0.01
        ).astype(int)

        data["Symbol"] = symbol

        features = [
            "Returns_1",
            "Returns_5",
            "Returns_10",
            "Returns_20",
            "RSI",
            "MACD",
            "MACD_Histogram",
            "Volatility_10",
            "Volatility_20",
            "Volume_Ratio",
            "Price_vs_SMA20",
            "Price_vs_SMA50",
            "Momentum_20",
            "Distance_From_High",
            "Distance_From_Low",
            "Trend_Strength",
            "High_Low_Range"
        ]

        columns = (
            ["Datetime"]
            + features
            + [
                "Future_Return_5",
                "Target",
                "Symbol"
            ]
        )

        dataset = data[columns]

        dataset = dataset.replace(
            [np.inf, -np.inf],
            np.nan
        )

        dataset = dataset.dropna()

        return dataset

    def build_all(self):

        datasets = []

        print("=" * 70)
        print("QUANTFORGE ML DATASET BUILDER")
        print("=" * 70)

        for symbol in self.symbols:

            try:

                data = self.load_features(symbol)

                if data is None:
                    continue

                dataset = self.build_dataset(
                    data,
                    symbol
                )

                datasets.append(dataset)

                print(
                    f"[ML DATA] Processed {symbol}"
                )

            except Exception as error:

                print(
                    f"[ERROR] {symbol}: {error}"
                )

        if not datasets:

            print("[ML DATA] No datasets created")
            return

        final_dataset = pd.concat(
            datasets,
            ignore_index=True
        )

        output_path = (
            self.processed_path
            / "ml_training_dataset.csv"
        )

        final_dataset.to_csv(
            output_path,
            index=False
        )

        print()
        print(
            f"[ML DATA] Rows: {len(final_dataset)}"
        )

        print(
            f"[ML DATA] Features: "
            f"{len(final_dataset.columns)}"
        )

        print(
            f"[ML DATA] Target distribution:"
        )

        print(
            final_dataset["Target"]
            .value_counts()
        )

        print(
            f"[ML DATA] Saved: {output_path}"
        )

        print("=" * 70)


if __name__ == "__main__":

    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "config.yaml"
    )

    builder = TrainingDatasetBuilder(
        config_path
    )

    builder.build_all()