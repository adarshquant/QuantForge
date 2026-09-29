import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class AlphaModel:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        self.features = [
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

    def load_data(self):

        data = pd.read_csv(
            self.dataset_path
        )

        data["Datetime"] = pd.to_datetime(
            data["Datetime"]
        )

        data = data.sort_values(
            "Datetime"
        )

        data = data.dropna(
            subset=self.features + ["Target"]
        )

        return data

    def split_data(self, data):

        total = len(data)

        train_end = int(total * 0.70)
        validation_end = int(total * 0.85)

        train = data.iloc[:train_end]
        validation = data.iloc[
            train_end:validation_end
        ]
        test = data.iloc[validation_end:]

        return train, validation, test

    def train(self, train):

        X_train = train[self.features]
        y_train = train["Target"]

        self.model.fit(
            X_train,
            y_train
        )

    def evaluate(self, data, name):

        X = data[self.features]
        y = data["Target"]

        predictions = self.model.predict(X)

        probabilities = (
            self.model
            .predict_proba(X)[:, 1]
        )

        print()
        print(
            f"===== {name} ====="
        )

        print(
            f"Rows: {len(data)}"
        )

        print(
            f"Accuracy: "
            f"{accuracy_score(y, predictions):.4f}"
        )

        print(
            f"Precision: "
            f"{precision_score(y, predictions, zero_division=0):.4f}"
        )

        print(
            f"Recall: "
            f"{recall_score(y, predictions, zero_division=0):.4f}"
        )

        print(
            f"F1: "
            f"{f1_score(y, predictions, zero_division=0):.4f}"
        )

        if y.nunique() > 1:

            print(
                f"ROC-AUC: "
                f"{roc_auc_score(y, probabilities):.4f}"
            )

    def run(self):

        print("=" * 70)
        print("QUANTFORGE ML ALPHA MODEL")
        print("=" * 70)

        data = self.load_data()

        print(
            f"[ML] Total rows: {len(data)}"
        )

        train, validation, test = (
            self.split_data(data)
        )

        print(
            f"[ML] Training rows: {len(train)}"
        )

        print(
            f"[ML] Validation rows: {len(validation)}"
        )

        print(
            f"[ML] Test rows: {len(test)}"
        )

        self.train(train)

        self.evaluate(
            train,
            "TRAINING"
        )

        self.evaluate(
            validation,
            "VALIDATION"
        )

        self.evaluate(
            test,
            "TEST"
        )

        print()
        print("=" * 70)


if __name__ == "__main__":

    base_dir = (
        Path(__file__).resolve().parents[2]
    )

    dataset_path = (
        base_dir
        / "data"
        / "processed"
        / "ml_training_dataset.csv"
    )

    model = AlphaModel(
        dataset_path
    )

    model.run()