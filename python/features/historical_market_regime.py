from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

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
    / "historical_market_regimes.csv"
)


def calculate_regime(group):

    trend_score = (
        group["Price_vs_SMA20"] > 0
    ).mean()

    momentum = group[
        "Momentum_20"
    ].mean()

    volatility = group[
        "Volatility_20"
    ].mean()

    if volatility > 0.035:
        regime = "HIGH_VOLATILITY"
        risk_multiplier = 0.50

    elif (
        trend_score >= 0.70
        and momentum > 0.015
    ):
        regime = "BULL"
        risk_multiplier = 1.00

    elif (
        trend_score <= 0.30
        and momentum < -0.015
    ):
        regime = "BEAR"
        risk_multiplier = 0.50

    else:
        regime = "SIDEWAYS"
        risk_multiplier = 0.75

    return pd.Series(
        {
            "Market_Regime": regime,
            "Trend_Score": trend_score,
            "Market_Momentum": momentum,
            "Market_Volatility": volatility,
            "Regime_Risk_Multiplier": risk_multiplier
        }
    )


def main():

    data = pd.read_csv(
        DATA_PATH
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Date",
            "Price_vs_SMA20",
            "Momentum_20",
            "Volatility_20"
        ]
    )

    regimes = (
        data.groupby("Date")
        .apply(
            calculate_regime
        )
        .reset_index()
    )

    regimes = regimes.sort_values(
        "Date"
    )

    regimes.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("=" * 70)
    print(
        "QUANTFORGE HISTORICAL MARKET REGIME ENGINE"
    )
    print("=" * 70)

    print()
    print(
        f"Trading dates: "
        f"{len(regimes)}"
    )

    print()
    print(
        regimes[
            [
                "Date",
                "Market_Regime",
                "Trend_Score",
                "Market_Momentum",
                "Market_Volatility",
                "Regime_Risk_Multiplier"
            ]
        ].tail(15).to_string(
            index=False
        )
    )

    print()
    print(
        "Regime distribution:"
    )

    print(
        regimes[
            "Market_Regime"
        ].value_counts().to_string()
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()