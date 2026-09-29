import pandas as pd
import yaml
from pathlib import Path


class StockScanner:

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

        return pd.read_csv(filepath)

    def calculate_indicators(self, data):

        data["Returns"] = data["Close"].pct_change()

        data["SMA_10"] = (
            data["Close"]
            .rolling(10)
            .mean()
        )

        data["SMA_20"] = (
            data["Close"]
            .rolling(20)
            .mean()
        )

        delta = data["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        data["RSI"] = 100 - (
            100 / (1 + rs)
        )

        data["Volatility"] = (
            data["Returns"]
            .rolling(20)
            .std()
        )

        data["Volume_MA"] = (
            data["Volume"]
            .rolling(20)
            .mean()
        )

        data["Volume_Ratio"] = (
            data["Volume"]
            / data["Volume_MA"]
        )

        return data

    def score_stock(self, data):

        latest = data.iloc[-1]

        score = 0

        if latest["Close"] > latest["SMA_10"]:
            score += 15

        if latest["SMA_10"] > latest["SMA_20"]:
            score += 20

        if latest["RSI"] > 50:
            score += 15

        if latest["RSI"] < 70:
            score += 10

        if latest["Volume_Ratio"] > 1.2:
            score += 15

        if latest["Returns"] > 0:
            score += 15

        if latest["Volatility"] < 0.03:
            score += 10

        if score >= 75:
            signal = "STRONG BUY"
        elif score >= 60:
            signal = "BUY"
        elif score >= 45:
            signal = "WATCH"
        elif score >= 30:
            signal = "SELL"
        else:
            signal = "STRONG SELL"

        return score, signal

    def scan(self):

        results = []

        print("=" * 70)
        print("QUANTFORGE STOCK SCANNER")
        print("=" * 70)

        for symbol in self.symbols:

            try:

                data = self.load_data(symbol)

                if data is None:
                    continue

                data = self.calculate_indicators(data)

                data = data.dropna()

                if data.empty:
                    continue

                score, signal = self.score_stock(data)

                results.append(
                    {
                        "Symbol": symbol,
                        "Score": score,
                        "Signal": signal
                    }
                )

            except Exception as error:

                print(
                    f"[ERROR] {symbol}: {error}"
                )

        results = sorted(
            results,
            key=lambda x: x["Score"],
            reverse=True
        )

        print()

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['Symbol']:<18}"
                f"{result['Signal']:<15}"
                f"{result['Score']}/100"
            )

        print()
        print("=" * 70)

        return results


if __name__ == "__main__":

    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "config.yaml"
    )

    scanner = StockScanner(config_path)

    scanner.scan()