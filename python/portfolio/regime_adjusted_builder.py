import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RISK_PATH = BASE_DIR / "data" / "processed" / "risk_allocated_predictions.csv"
REGIME_PATH = BASE_DIR / "data" / "processed" / "historical_market_regimes.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "regime_adjusted_backtest_input.csv"

risk = pd.read_csv(RISK_PATH)
regime = pd.read_csv(REGIME_PATH)

risk["Date"] = pd.to_datetime(risk["Date"])
regime["Date"] = pd.to_datetime(regime["Date"])

df = risk.merge(
    regime[
        [
            "Date",
            "Market_Regime",
            "Regime_Risk_Multiplier"
        ]
    ],
    on="Date",
    how="inner"
)

numeric_columns = [
    "Return_1D",
    "Predicted_Alpha",
    "Prediction_Rank",
    "Position_Weight",
    "Regime_Risk_Multiplier"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "Date",
        "Symbol",
        "Position_Weight",
        "Regime_Risk_Multiplier"
    ]
)

df["Portfolio_Exposure"] = df["Regime_Risk_Multiplier"]

df["Position_Weight"] = (
    df["Position_Weight"] *
    df["Portfolio_Exposure"]
)

df = df.sort_values(
    ["Date", "Position_Weight"],
    ascending=[True, False]
).reset_index(drop=True)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("=" * 70)
print("QUANTFORGE REGIME-ADJUSTED PORTFOLIO")
print("=" * 70)
print(f"Rows: {len(df)}")
print(f"Trading sessions: {df['Date'].nunique()}")
print(f"Symbols: {df['Symbol'].nunique()}")
print()
print("REGIME DISTRIBUTION")
print(df["Market_Regime"].value_counts())
print()
print("AVERAGE PORTFOLIO EXPOSURE")
print(
    df.groupby("Market_Regime")["Portfolio_Exposure"]
    .first()
    .to_string()
)
print()
print(
    df[
        [
            "Date",
            "Symbol",
            "Market_Regime",
            "Predicted_Alpha",
            "Position_Weight",
            "Portfolio_Exposure"
        ]
    ].tail(15).to_string(index=False)
)
print()
print(f"Output: {OUTPUT_PATH}")
print("REGIME ADJUSTMENT COMPLETED SUCCESSFULLY")