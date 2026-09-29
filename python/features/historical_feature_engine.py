import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from python.data.universe_engine import UniverseEngine


class HistoricalFeatureEngine:

    def __init__(self):

        self.base_dir = BASE_DIR

        self.portfolio_config_path = (
            self.base_dir
            / "config"
            / "portfolio_config.json"
        )

        self.historical_path = (
            self.base_dir
            / "data"
            / "historical"
        )

        self.processed_path = (
            self.base_dir
            / "data"
            / "processed"
        )

        self.processed_path.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.portfolio_config_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.config = json.load(file)

        self.universe_engine = UniverseEngine()

        self.universe_type = (
            self.config["universe_type"]
            .upper()
        )

        self.symbols = self.resolve_universe()

    def resolve_universe(self):

        if self.universe_type == "CUSTOM":

            symbols = self.config.get(
                "custom_universe",
                []
            )

            if not symbols:
                raise ValueError(
                    "Custom universe is empty"
                )

            return list(
                dict.fromkeys(
                    symbol.upper()
                    for symbol in symbols
                )
            )

        return self.universe_engine.get_symbols(
            self.universe_type
        )

    def load_data(self, symbol):

        filename = (
            symbol.replace(".", "_")
            + "_historical.csv"
        )

        filepath = (
            self.historical_path
            / filename
        )

        if not filepath.exists():

            print(
                f"[WARNING] Missing historical data: "
                f"{symbol}"
            )

            return None

        data = pd.read_csv(filepath)

        if data.empty:

            print(
                f"[WARNING] Empty historical data: "
                f"{symbol}"
            )

            return None

        date_column = None

        for column in ["Date", "Datetime"]:

            if column in data.columns:
                date_column = column
                break

        if date_column is None:

            raise ValueError(
                f"No date column found for {symbol}"
            )

        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce"
        )

        data = data.dropna(
            subset=[date_column]
        )

        data = data.sort_values(
            date_column
        )

        data = data.drop_duplicates(
            subset=[date_column]
        )

        data = data.rename(
            columns={
                date_column: "Date"
            }
        )

        data["Symbol"] = symbol

        return data

    def calculate_features(self, data):

        data = data.copy()

        close = data["Close"]
        volume = data["Volume"]
        high = data["High"]
        low = data["Low"]

        data["Return_1D"] = (
            close.pct_change(1)
        )

        data["Return_5D"] = (
            close.pct_change(5)
        )

        data["Return_10D"] = (
            close.pct_change(10)
        )

        data["Return_20D"] = (
            close.pct_change(20)
        )

        data["Return_60D"] = (
            close.pct_change(60)
        )

        data["SMA10"] = (
            close.rolling(10).mean()
        )

        data["SMA20"] = (
            close.rolling(20).mean()
        )

        data["SMA50"] = (
            close.rolling(50).mean()
        )

        data["SMA100"] = (
            close.rolling(100).mean()
        )

        data["SMA200"] = (
            close.rolling(200).mean()
        )

        data["EMA12"] = (
            close.ewm(
                span=12,
                adjust=False
            ).mean()
        )

        data["EMA26"] = (
            close.ewm(
                span=26,
                adjust=False
            ).mean()
        )

        data["MACD"] = (
            data["EMA12"]
            - data["EMA26"]
        )

        data["MACD_Signal"] = (
            data["MACD"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        data["MACD_Histogram"] = (
            data["MACD"]
            - data["MACD_Signal"]
        )

        delta = close.diff()

        gain = (
            delta.where(delta > 0, 0)
            .rolling(14)
            .mean()
        )

        loss = (
            -delta.where(delta < 0, 0)
            .rolling(14)
            .mean()
        )

        rs = gain / loss.replace(0, np.nan)

        data["RSI"] = (
            100
            - (
                100
                / (1 + rs)
            )
        )

        data["Volatility_10"] = (
            data["Return_1D"]
            .rolling(10)
            .std()
        )

        data["Volatility_20"] = (
            data["Return_1D"]
            .rolling(20)
            .std()
        )

        data["Volatility_60"] = (
            data["Return_1D"]
            .rolling(60)
            .std()
        )

        volume_average = (
            volume.rolling(20).mean()
        )

        data["Volume_Ratio"] = (
            volume
            / volume_average.replace(0, np.nan)
        )

        data["Price_vs_SMA20"] = (
            close / data["SMA20"] - 1
        )

        data["Price_vs_SMA50"] = (
            close / data["SMA50"] - 1
        )

        data["Price_vs_SMA200"] = (
            close / data["SMA200"] - 1
        )

        data["SMA50_vs_SMA200"] = (
            data["SMA50"]
            / data["SMA200"]
            - 1
        )

        data["Momentum_20"] = (
            close
            / close.shift(20)
            - 1
        )

        data["Momentum_60"] = (
            close
            / close.shift(60)
            - 1
        )

        rolling_high = (
            high.rolling(20).max()
        )

        rolling_low = (
            low.rolling(20).min()
        )

        data["Rolling_High_20"] = (
            rolling_high
        )

        data["Rolling_Low_20"] = (
            rolling_low
        )

        data["Distance_From_High"] = (
            close / rolling_high - 1
        )

        data["Distance_From_Low"] = (
            close / rolling_low - 1
        )

        rolling_mean = (
            close.rolling(20).mean()
        )

        rolling_std = (
            close.rolling(20).std()
        )

        data["Price_ZScore_20"] = (
            (close - rolling_mean)
            / rolling_std.replace(0, np.nan)
        )

        rolling_peak = (
            close.rolling(60).max()
        )

        data["Drawdown_60"] = (
            close / rolling_peak - 1
        )

        data["Trend_Strength"] = (
            (
                data["Price_vs_SMA20"]
                + data["Price_vs_SMA50"]
                + data["Price_vs_SMA200"]
            )
            / 3
        )

        data["High_Low_Range"] = (
            (high - low)
            / close.replace(0, np.nan)
        )

        return data

    def process_symbol(self, symbol):

        data = self.load_data(symbol)

        if data is None:
            return None

        data = self.calculate_features(data)

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        feature_columns = [
            "Return_1D",
            "Return_5D",
            "Return_10D",
            "Return_20D",
            "Return_60D",
            "SMA10",
            "SMA20",
            "SMA50",
            "SMA100",
            "SMA200",
            "EMA12",
            "EMA26",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "RSI",
            "Volatility_10",
            "Volatility_20",
            "Volatility_60",
            "Volume_Ratio",
            "Price_vs_SMA20",
            "Price_vs_SMA50",
            "Price_vs_SMA200",
            "SMA50_vs_SMA200",
            "Momentum_20",
            "Momentum_60",
            "Distance_From_High",
            "Distance_From_Low",
            "Price_ZScore_20",
            "Drawdown_60",
            "Trend_Strength",
            "High_Low_Range"
        ]

        data = data.dropna(
            subset=feature_columns
        )

        return data

    def update(self):

        print("=" * 70)
        print(
            "QUANTFORGE DYNAMIC HISTORICAL "
            "FEATURE ENGINE"
        )
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        print(
            f"Symbols: {len(self.symbols)}"
        )

        print("=" * 70)

        results = []

        for index, symbol in enumerate(
            self.symbols,
            start=1
        ):

            print(
                f"[{index}/{len(self.symbols)}] "
                f"Processing {symbol}..."
            )

            try:

                data = self.process_symbol(
                    symbol
                )

                if data is None:
                    continue

                results.append(data)

                print(
                    f"[FEATURES] {symbol} | "
                    f"Rows: {len(data)}"
                )

            except Exception as error:

                print(
                    f"[ERROR] {symbol}: {error}"
                )

        if not results:

            raise RuntimeError(
                "No feature datasets were generated"
            )

        combined = pd.concat(
            results,
            ignore_index=True
        )

        combined = combined.sort_values(
            ["Date", "Symbol"]
        )

        output_path = (
            self.processed_path
            / "historical_features.csv"
        )

        combined.to_csv(
            output_path,
            index=False
        )

        print()
        print("=" * 70)
        print("FEATURE ENGINE COMPLETED")
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        print(
            f"Symbols Processed: "
            f"{combined['Symbol'].nunique()}"
        )

        print(
            f"Total Rows: {len(combined)}"
        )

        print(
            f"Feature Count: "
            f"{len(combined.columns)}"
        )

        print(
            f"Output: {output_path}"
        )

        print("=" * 70)


if __name__ == "__main__":

    engine = HistoricalFeatureEngine()

    engine.update()