from pathlib import Path

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


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cross_sectional_dataset.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "walk_forward_results.csv"
)

N_ESTIMATORS = 400
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 8
TRAIN_DAYS = 400
TEST_DAYS = 60
STEP_DAYS = 60
RANDOM_STATE = 42


def load_dataset():

    data = pd.read_csv(INPUT_PATH)

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.sort_values(
        ["Date", "Symbol"]
    ).reset_index(drop=True)

    data = data.dropna(
        subset=["Date", "Alpha_Target"]
    )

    return data


def prepare_features(data):

    exclude_columns = {
        "Date",
        "Symbol",
        "Alpha_Target",
        "Future_Return_20D",
        "CrossSectional_Return",
        "Excess_Return",
        "Alpha_Rank"
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in exclude_columns
        and pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]

    return feature_columns


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_fold(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    if len(np.unique(y_test)) > 1:

        metrics["ROC_AUC"] = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        metrics["ROC_AUC"] = np.nan

    return metrics


def run_walk_forward(
    data,
    feature_columns
):

    unique_dates = np.array(
        sorted(
            data["Date"].unique()
        )
    )

    results = []

    start = TRAIN_DAYS

    fold = 1

    while (
        start + TEST_DAYS
        <= len(unique_dates)
    ):

        train_dates = unique_dates[
            start - TRAIN_DAYS:start
        ]

        test_dates = unique_dates[
            start:start + TEST_DAYS
        ]

        train_data = data[
            data["Date"].isin(
                train_dates
            )
        ].copy()

        test_data = data[
            data["Date"].isin(
                test_dates
            )
        ].copy()

        train_data = train_data.dropna(
            subset=feature_columns
        )

        test_data = test_data.dropna(
            subset=feature_columns
        )

        if (
            train_data.empty
            or test_data.empty
        ):

            start += STEP_DAYS
            fold += 1
            continue

        X_train = train_data[
            feature_columns
        ]

        y_train = train_data[
            "Alpha_Target"
        ].astype(int)

        X_test = test_data[
            feature_columns
        ]

        y_test = test_data[
            "Alpha_Target"
        ].astype(int)

        model = train_model(
            X_train,
            y_train
        )

        metrics = evaluate_fold(
            model,
            X_test,
            y_test
        )

        results.append(
            {
                "Fold": fold,
                "Train_Start": train_dates[0],
                "Train_End": train_dates[-1],
                "Test_Start": test_dates[0],
                "Test_End": test_dates[-1],
                "Train_Rows": len(train_data),
                "Test_Rows": len(test_data),
                **metrics
            }
        )

        print()
        print(
            f"Fold {fold}"
        )

        print(
            f"Train: "
            f"{train_dates[0]} → "
            f"{train_dates[-1]}"
        )

        print(
            f"Test: "
            f"{test_dates[0]} → "
            f"{test_dates[-1]}"
        )

        print(
            f"Accuracy: {metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['Precision']:.4f}"
        )

        print(
            f"Recall: {metrics['Recall']:.4f}"
        )

        print(
            f"F1: {metrics['F1']:.4f}"
        )

        print(
            f"ROC-AUC: {metrics['ROC_AUC']:.4f}"
        )

        start += STEP_DAYS

        fold += 1

    return pd.DataFrame(results)


def main():

    print()
    print("=" * 70)
    print("QUANTFORGE WALK-FORWARD VALIDATION")
    print("=" * 70)

    data = load_dataset()

    feature_columns = prepare_features(
        data
    )

    print()
    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Trading Dates: "
        f"{data['Date'].nunique()}"
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    results = run_walk_forward(
        data,
        feature_columns
    )

    if results.empty:

        raise ValueError(
            "No walk-forward folds were generated."
        )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    print(
        f"Folds: {len(results)}"
    )

    print(
        f"Mean Accuracy: "
        f"{results['Accuracy'].mean():.4f}"
    )

    print(
        f"Mean Precision: "
        f"{results['Precision'].mean():.4f}"
    )

    print(
        f"Mean Recall: "
        f"{results['Recall'].mean():.4f}"
    )

    print(
        f"Mean F1: "
        f"{results['F1'].mean():.4f}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{results['ROC_AUC'].mean():.4f}"
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print(
        "Walk-forward validation completed successfully."
    )


if __name__ == "__main__":

    main()