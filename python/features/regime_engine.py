import pandas as pd
import numpy as np


class RegimeEngine:

    def calculate(self, data):

        data = data.copy()

        data["Returns"] = data["Close"].pct_change()

        data["SMA_20"] = (
            data["Close"]
            .rolling(20)
            .mean()
        )

        data["SMA_50"] = (
            data["Close"]
            .rolling(50)
            .mean()
        )

        data["Volatility"] = (
            data["Returns"]
            .rolling(20)
            .std()
        )

        data["Momentum"] = (
            data["Close"]
            / data["Close"].shift(20)
            - 1
        )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna(
            subset=[
                "SMA_20",
                "Volatility",
                "Momentum"
            ]
        )

        if data.empty:
            return "UNKNOWN"

        latest = data.iloc[-1]

        if pd.isna(latest["SMA_50"]):

            trend = (
                latest["Close"]
                > latest["SMA_20"]
            )

        else:

            trend = (
                latest["SMA_20"]
                > latest["SMA_50"]
            )

        momentum = latest["Momentum"]
        volatility = latest["Volatility"]

        if volatility > 0.035:

            regime = "HIGH_VOLATILITY"

        elif trend and momentum > 0.03:

            regime = "BULL"

        elif not trend and momentum < -0.03:

            regime = "BEAR"

        else:

            regime = "SIDEWAYS"

        return regime


if __name__ == "__main__":

    data = pd.read_csv(
        "data/raw/RELIANCE_NS.csv"
    )

    engine = RegimeEngine()

    regime = engine.calculate(data)

    print(
        f"Market Regime: {regime}"
    )