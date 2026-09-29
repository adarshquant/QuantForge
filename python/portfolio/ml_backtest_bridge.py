from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_predictions.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_backtest_input.csv"
)


def main():

    data = pd.read_csv(INPUT_PATH)

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Date",
            "Symbol",
            "Return_1D",
            "Predicted_Alpha",
            "Prediction_Rank"
        ]
    )

    data = data.sort_values(
        [
            "Date",
            "Prediction_Rank"
        ],
        ascending=[
            True,
            False
        ]
    )
    data = data.sort_values(
        [
            "Symbol",
            "Date"
        ]
    )

    data["Next_Return_1D"] = (
        data.groupby("Symbol")["Return_1D"]
        .shift(-1)
    )

    data = data.dropna(
        subset=["Next_Return_1D"]
    )

    data = data.sort_values(
        [
            "Date",
            "Prediction_Rank"
        ],
        ascending=[
            True,
            False
        ]
    )

    output = data[
        [
            "Date",
            "Symbol",
            "Next_Return_1D",
            "Predicted_Alpha",
            "Prediction_Rank",
            "Predicted_Signal"
        ]
    ].copy()

    output = output.rename(
        columns={
            "Next_Return_1D": "Return_1D"
        }
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("=" * 70)
    print("QUANTFORGE ML → C++ BACKTEST BRIDGE")
    print("=" * 70)

    print()
    print(
        f"Rows: {len(output)}"
    )

    print(
        f"Trading sessions: "
        f"{output['Date'].nunique()}"
    )

    print(
        f"Symbols: "
        f"{output['Symbol'].nunique()}"
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print(
        output.tail(15).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()