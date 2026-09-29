import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class MarketRegimeEngine:

    def __init__(self, config_path):

        self.config_path = Path(config_path)

        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.symbols = self.config["universe"]

        self.data_path = (
            self.config_path.parent.parent
            / "data"
            / "raw"
        )

    def load_data(self, symbol):

        filename = symbol.replace(".", "_") + ".csv"

        filepath = self.data_path / filename

        if not filepath.exists():
            return None

        data = pd.read_csv(filepath)

        data["Returns"] = data["Close"].pct_change()

        data["SMA_20"] = (
            data["Close"]
            .rolling(20)
            .mean()
        )

        data["Momentum"] = (
            data["Close"]
            / data["Close"].shift(20)
            - 1
        )

        data["Volatility"] = (
            data["Returns"]
            .rolling(20)
            .std()
        )

        return data

    def calculate(self):

        stocks = []

        for symbol in self.symbols:

            data = self.load_data(symbol)

            if data is None:
                continue

            data = data.dropna(
                subset=[
                    "SMA_20",
                    "Momentum",
                    "Volatility"
                ]
            )

            if data.empty:
                continue

            latest = data.iloc[-1]

            trend = (
                latest["Close"]
                > latest["SMA_20"]
            )

            stocks.append(
                {
                    "trend": int(trend),
                    "momentum": latest["Momentum"],
                    "volatility": latest["Volatility"]
                }
            )

        if not stocks:
            return "UNKNOWN"

        frame = pd.DataFrame(stocks)

        trend_score = frame["trend"].mean()

        momentum = frame["momentum"].mean()

        volatility = frame["volatility"].mean()

        if volatility > 0.035:

            return "HIGH_VOLATILITY"

        if trend_score >= 0.70 and momentum > 0.015:

            return "BULL"

        if trend_score <= 0.30 and momentum < -0.015:

            return "BEAR"

        return "SIDEWAYS"


if __name__ == "__main__":

    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "config.yaml"
    )

    engine = MarketRegimeEngine(config_path)

    regime = engine.calculate()

    print(
        f"Overall Market Regime: {regime}"
    )