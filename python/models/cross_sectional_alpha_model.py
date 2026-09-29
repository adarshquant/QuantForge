import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class CrossSectionalAlphaModel:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.base_dir = self.dataset_path.parents[2]

        self.model_dir = (
            self.base_dir
            / "data"
            / "processed"
            / "models"
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_path = (
            self.model_dir
            / "cross_sectional_alpha_model.joblib"
        )

        self.metadata_path = (
            self.model_dir
            / "cross_sectional_alpha_metadata.json"
        )

        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )

    def load_data(self):

        data = pd.read_csv(
            self.dataset_path
        )

        data["Date"] = pd.to_datetime(
            data["Date"],
            utc=True
        ).dt.date

        data = data.sort_values(
            ["Date", "Symbol"]
        ).reset_index(drop=True)

        return data

    def prepare_data(self, data):

        excluded = {
            "Date",
            "Symbol",
            "Alpha_Target"
        }

        features = [
            column
            for column in data.columns
            if column not in excluded
            and pd.api.types.is_numeric_dtype(
                data[column]
            )
        ]

        data = data.dropna(
            subset=features + ["Alpha_Target"]
        ).copy()

        return data, features

    def split_data(self, data):

        dates = np.array(
            sorted(data["Date"].unique())
        )

        total = len(dates)

        train_end = int(total * 0.60)
        validation_end = int(total * 0.80)

        train_dates = dates[:train_end]
        validation_dates = dates[
            train_end:validation_end
        ]
        test_dates = dates[
            validation_end:
        ]

        train = data[
            data["Date"].isin(train_dates)
        ]

        validation = data[
            data["Date"].isin(validation_dates)
        ]

        test = data[
            data["Date"].isin(test_dates)
        ]

        return train, validation, test

    def evaluate(self, name, y_true, probabilities):

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        try:
            roc_auc = roc_auc_score(
                y_true,
                probabilities
            )
        except ValueError:
            roc_auc = 0.0

        print()
        print(
            f"===== {name} ====="
        )

        print(
            f"Rows: {len(y_true)}"
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall: {recall:.4f}"
        )

        print(
            f"F1: {f1:.4f}"
        )

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

    def rank_analysis(
        self,
        data,
        probabilities
    ):

        result = data[
            ["Date", "Symbol", "Alpha_Target"]
        ].copy()

        result["Alpha_Probability"] = probabilities

        result["Rank"] = (
            result
            .groupby("Date")[
                "Alpha_Probability"
            ]
            .rank(
                ascending=False,
                method="first"
            )
        )

        result["Universe_Size"] = (
            result
            .groupby("Date")[
                "Symbol"
            ]
            .transform("count")
        )

        result["Top_Quintile"] = (
            result["Rank"]
            <= result["Universe_Size"] * 0.20
        )

        top_quintile_rate = (
            result.loc[
                result["Top_Quintile"],
                "Alpha_Target"
            ].mean()
        )

        overall_rate = (
            result["Alpha_Target"].mean()
        )

        print()
        print("===== CROSS-SECTIONAL RANKING =====")

        print(
            f"Overall positive rate: "
            f"{overall_rate:.4f}"
        )

        print(
            f"Top quintile positive rate: "
            f"{top_quintile_rate:.4f}"
        )

        return result

    def save_model(
        self,
        features,
        metrics
    ):

        joblib.dump(
            self.model,
            self.model_path
        )

        metadata = {
            "features": features,
            "metrics": metrics,
            "model": "RandomForestClassifier",
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_leaf": 8
        }

        with open(
            self.metadata_path,
            "w"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        print()
        print(
            f"[ALPHA] Model saved: "
            f"{self.model_path}"
        )

        print(
            f"[ALPHA] Metadata saved: "
            f"{self.metadata_path}"
        )

    def run(self):

        print("=" * 70)
        print(
            "QUANTFORGE CROSS-SECTIONAL ALPHA MODEL"
        )
        print("=" * 70)

        data = self.load_data()

        data, features = self.prepare_data(
            data
        )

        train, validation, test = (
            self.split_data(data)
        )

        print()
        print(
            f"Total rows: {len(data)}"
        )

        print(
            f"Features: {len(features)}"
        )

        print(
            f"Training rows: {len(train)}"
        )

        print(
            f"Validation rows: {len(validation)}"
        )

        print(
            f"Test rows: {len(test)}"
        )

        print(
            f"Training dates: "
            f"{train['Date'].min()} → "
            f"{train['Date'].max()}"
        )

        print(
            f"Validation dates: "
            f"{validation['Date'].min()} → "
            f"{validation['Date'].max()}"
        )

        print(
            f"Test dates: "
            f"{test['Date'].min()} → "
            f"{test['Date'].max()}"
        )

        X_train = train[features]
        y_train = train["Alpha_Target"]

        X_validation = validation[features]
        y_validation = validation["Alpha_Target"]

        X_test = test[features]
        y_test = test["Alpha_Target"]

        print()
        print("===== TRAINING =====")

        self.model.fit(
            X_train,
            y_train
        )

        train_probabilities = (
            self.model
            .predict_proba(X_train)[:, 1]
        )

        validation_probabilities = (
            self.model
            .predict_proba(X_validation)[:, 1]
        )

        test_probabilities = (
            self.model
            .predict_proba(X_test)[:, 1]
        )

        metrics = {}

        metrics["train"] = self.evaluate(
            "TRAINING",
            y_train,
            train_probabilities
        )

        metrics["validation"] = self.evaluate(
            "VALIDATION",
            y_validation,
            validation_probabilities
        )

        metrics["test"] = self.evaluate(
            "TEST",
            y_test,
            test_probabilities
        )

        self.rank_analysis(
            test,
            test_probabilities
        )

        importance = pd.DataFrame(
            {
                "Feature": features,
                "Importance":
                    self.model.feature_importances_
            }
        ).sort_values(
            "Importance",
            ascending=False
        )

        print()
        print("===== FEATURE IMPORTANCE =====")

        print(
            importance.head(15).to_string(
                index=False
            )
        )

        self.save_model(
            features,
            metrics
        )

        print()
        print("=" * 70)
        print(
            "CROSS-SECTIONAL ALPHA MODEL COMPLETE"
        )
        print("=" * 70)


if __name__ == "__main__":

    base_dir = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    dataset_path = (
        base_dir
        / "data"
        / "processed"
        / "cross_sectional_dataset.csv"
    )

    model = CrossSectionalAlphaModel(
        dataset_path
    )

    model.run()