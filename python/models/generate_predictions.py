import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib


BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    BASE_DIR
    / "config"
    / "portfolio_config.json"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cross_sectional_dataset.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_predictions.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "cross_sectional_alpha_model.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "cross_sectional_alpha_metadata.json"
)


class DynamicAlphaModel:

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

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_dataset(self):

        if not DATA_PATH.exists():

            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        data = pd.read_csv(
            DATA_PATH
        )

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        data["Symbol"] = (
            data["Symbol"]
            .astype(str)
            .str.upper()
        )

        data = data.dropna(
            subset=["Date", "Symbol"]
        )

        if self.universe_type == "CUSTOM":

            data = data[
                data["Symbol"].isin(
                    self.custom_universe
                )
            ]

        return data

    def prepare_features(self, data):

        excluded = [
            "Date",
            "Symbol",
            "Alpha_Target",
            "Future_Return_20D",
            "CrossSectional_Return",
            "Excess_Return",
            "Alpha_Rank"
        ]

        features = [
            column
            for column in data.columns
            if column not in excluded
        ]

        return features

    def split_data(self, data):

        dates = sorted(
            data["Date"].unique()
        )

        total_dates = len(dates)

        train_end = int(
            total_dates * 0.60
        )

        validation_end = int(
            total_dates * 0.80
        )

        train_dates = dates[
            :train_end
        ]

        validation_dates = dates[
            train_end:validation_end
        ]

        test_dates = dates[
            validation_end:
        ]

        train = data[
            data["Date"].isin(
                train_dates
            )
        ]

        validation = data[
            data["Date"].isin(
                validation_dates
            )
        ]

        test = data[
            data["Date"].isin(
                test_dates
            )
        ]

        return train, validation, test

    def train(self, X_train, y_train):

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        return model

    def generate_predictions(
        self,
        model,
        data,
        features
    ):

        X = data[features]

        probabilities = (
            model.predict_proba(X)
        )

        classes = list(
            model.classes_
        )

        if 1 in classes:

            alpha_index = classes.index(1)

            alpha_probability = (
                probabilities[:, alpha_index]
            )

        else:

            alpha_probability = np.zeros(
                len(data)
            )

        result = data[
            [
                "Date",
                "Symbol",
                "Return_1D",
                "Alpha_Target"
            ]
        ].copy()

        result["Predicted_Alpha"] = (
            alpha_probability
        )

        result["Prediction_Rank"] = (
            result
            .groupby("Date")[
                "Predicted_Alpha"
            ]
            .rank(
                pct=True,
                method="average"
            )
        )

        result["Predicted_Signal"] = np.select(
            [
                result["Prediction_Rank"] >= 0.80,
                result["Prediction_Rank"] >= 0.60,
                result["Prediction_Rank"] >= 0.40,
                result["Prediction_Rank"] >= 0.20
            ],
            [
                "STRONG_BUY",
                "BUY",
                "WATCH",
                "SELL"
            ],
            default="STRONG_SELL"
        )

        return result

    def run(self):

        print("=" * 70)
        print(
            "QUANTFORGE DYNAMIC ALPHA MODEL"
        )
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        data = self.load_dataset()

        if data.empty:

            raise RuntimeError(
                "No data available for current universe"
            )

        features = self.prepare_features(
            data
        )

        data = data.dropna(
            subset=features + ["Alpha_Target"]
        )

        train, validation, test = (
            self.split_data(data)
        )

        print(
            f"Symbols: "
            f"{data['Symbol'].nunique()}"
        )

        print(
            f"Total Rows: {len(data)}"
        )

        print(
            f"Train Rows: {len(train)}"
        )

        print(
            f"Validation Rows: {len(validation)}"
        )

        print(
            f"Test Rows: {len(test)}"
        )

        X_train = train[features]

        y_train = train[
            "Alpha_Target"
        ]

        model = self.train(
            X_train,
            y_train
        )

        validation_predictions = (
            self.generate_predictions(
                model,
                validation,
                features
            )
        )

        test_predictions = (
            self.generate_predictions(
                model,
                test,
                features
            )
        )

        predictions = pd.concat(
            [
                validation_predictions,
                test_predictions
            ],
            ignore_index=True
        )

        predictions = predictions.sort_values(
            ["Date", "Prediction_Rank"],
            ascending=[
                True,
                False
            ]
        )

        predictions.to_csv(
            OUTPUT_PATH,
            index=False
        )

        joblib.dump(
            model,
            MODEL_PATH
        )

        metadata = {
            "universe_type": self.universe_type,
            "symbols": sorted(
                data["Symbol"]
                .unique()
                .tolist()
            ),
            "symbol_count": int(
                data["Symbol"].nunique()
            ),
            "feature_count": len(features),
            "features": features,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "model": "RandomForestClassifier",
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_leaf": 8,
            "random_state": 42
        }

        with open(
            METADATA_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2
            )

        print()
        print("=" * 70)
        print(
            "DYNAMIC ALPHA MODEL COMPLETED"
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
            f"Features: {len(features)}"
        )

        print(
            f"Predictions: {len(predictions)}"
        )

        print(
            f"Model: {MODEL_PATH}"
        )

        print(
            f"Predictions: {OUTPUT_PATH}"
        )

        print("=" * 70)


if __name__ == "__main__":

    engine = DynamicAlphaModel()

    engine.run()