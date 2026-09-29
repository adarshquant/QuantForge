import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

REGIME_PATH = BASE_DIR / "data" / "processed" / "regime_adjusted_backtest_input.csv"
ML_RETURN_PATH = BASE_DIR / "data" / "processed" / "ml_backtest_input.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "final_backtest_input.csv"

regime = pd.read_csv(REGIME_PATH)
returns = pd.read_csv(ML_RETURN_PATH)

regime["Date"] = pd.to_datetime(regime["Date"])
returns["Date"] = pd.to_datetime(returns["Date"])

returns = returns[
    [
        "Date",
        "Symbol",
        "Return_1D"
    ]
].rename(
    columns={
        "Return_1D": "Next_Day_Return"
    }
)

df = regime.merge(
    returns,
    on=["Date", "Symbol"],
    how="inner"
)

required_columns = [
    "Date",
    "Symbol",
    "Predicted_Alpha",
    "Prediction_Rank",
    "Position_Weight",
    "Next_Day_Return"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["Return_1D"] = pd.to_numeric(
    df["Next_Day_Return"],
    errors="coerce"
)

df["Predicted_Alpha"] = pd.to_numeric(
    df["Predicted_Alpha"],
    errors="coerce"
)

df["Prediction_Rank"] = pd.to_numeric(
    df["Prediction_Rank"],
    errors="coerce"
)

df["Position_Weight"] = pd.to_numeric(
    df["Position_Weight"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Date",
        "Symbol",
        "Return_1D",
        "Predicted_Alpha",
        "Prediction_Rank",
        "Position_Weight"
    ]
)

output_columns = [
    "Date",
    "Symbol",
    "Return_1D",
    "Predicted_Alpha",
    "Prediction_Rank",
    "Position_Weight"
]

df = df[output_columns]

df = df.sort_values(
    ["Date", "Prediction_Rank", "Symbol"]
).reset_index(drop=True)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("=" * 70)
print("QUANTFORGE FINAL BACKTEST INPUT")
print("=" * 70)
print(f"Rows: {len(df)}")
print(f"Trading sessions: {df['Date'].nunique()}")
print(f"Symbols: {df['Symbol'].nunique()}")
print()
print(df.tail(15).to_string(index=False))
print()
print(f"Output: {OUTPUT_PATH}")
print("FINAL BACKTEST INPUT CREATED SUCCESSFULLY")