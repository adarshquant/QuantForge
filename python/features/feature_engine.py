import pandas as pd
import numpy as np
from pathlib import Path


class FeatureEngine:

    def __init__(self, data_path, processed_path):

        self.data_path = Path(data_path)
        self.processed_path = Path(processed_path)

        self.processed_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_data(self, symbol):

        filename = symbol.replace(".", "_") + ".csv"
        filepath = self.data_path / filename

        if not filepath.exists():
            return None

        return pd.read_csv(filepath)

    def build_features(self, data):

        data["Returns_1"] = data["Close"].pct_change(1)
        data["Returns_5"] = data["Close"].pct_change(5)
        data["Returns_10"] = data["Close"].pct_change(10)
        data["Returns_20"] = data["Close"].pct_change(20)

        data["SMA_10"] = data["Close"].rolling(10).mean()
        data["SMA_20"] = data["Close"].rolling(20).mean()
        data["SMA_50"] = data["Close"].rolling(50).mean()

        data["EMA_12"] = data["Close"].ewm(span=12).mean()
        data["EMA_26"] = data["Close"].ewm(span=26).mean()

        data["MACD"] = (
            data["EMA_12"] -
            data["EMA_26"]
        )

        data["MACD_Signal"] = (
            data["MACD"]
            .ewm(span=9)
            .mean()
        )

        delta = data["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        data["RSI"] = (
            100 -
            (100 / (1 + rs))
        )

        data["Volatility_10"] = (
            data["Returns_1"]
            .rolling(10)
            .std()
        )

        data["Volatility_20"] = (
            data["Returns_1"]
            .rolling(20)
            .std()
        )

        data["Volume_MA_20"] = (
            data["Volume"]
            .rolling(20)
            .mean()
        )

        data["Volume_Ratio"] = (
            data["Volume"] /
            data["Volume_MA_20"]
        )

        data["Price_vs_SMA20"] = (
            data["Close"] /
            data["SMA_20"] - 1
        )

        data["Price_vs_SMA50"] = (
            data["Close"] /
            data["SMA_50"] - 1
        )

        data["High_Low_Range"] = (
            data["High"] -
            data["Low"]
        ) / data["Close"]

        data["Momentum_20"] = (
            data["Close"] /
            data["Close"].shift(20) - 1
        )

        data["Rolling_High_20"] = (
            data["High"]
            .rolling(20)
            .max()
        )

        data["Rolling_Low_20"] = (
            data["Low"]
            .rolling(20)
            .min()
        )

        data["Distance_From_High"] = (
            data["Close"] /
            data["Rolling_High_20"] - 1
        )

        data["Distance_From_Low"] = (
            data["Close"] /
            data["Rolling_Low_20"] - 1
        )

        data["Trend_Strength"] = (
            data["SMA_10"] /
            data["SMA_50"] - 1
        )

        data["MACD_Histogram"] = (
            data["MACD"] -
            data["MACD_Signal"]
        )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna()

        return data

    def process(self, symbol):

        data = self.load_data(symbol)

        if data is None:
            return None

        data = self.build_features(data)

        filename = symbol.replace(".", "_") + "_features.csv"

        filepath = self.processed_path / filename

        data.to_csv(
            filepath,
            index=False
        )

        print(
            f"[FEATURE] Processed {symbol}"
        )

        return data

    def process_all(self, symbols):

        print("=" * 70)
        print("QUANTFORGE FEATURE ENGINE")
        print("=" * 70)

        for symbol in symbols:

            try:

                self.process(symbol)

            except Exception as error:

                print(
                    f"[ERROR] {symbol}: {error}"
                )

        print("=" * 70)


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parents[2]

    data_path = base_dir / "data" / "raw"
    processed_path = base_dir / "data" / "processed"

    symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS"
    ]

    engine = FeatureEngine(
        data_path,
        processed_path
    )

    engine.process_all(symbols)