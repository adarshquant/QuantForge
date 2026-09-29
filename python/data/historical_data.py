import json
import time
import yfinance as yf
from pathlib import Path
from datetime import datetime
import sys

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from python.data.universe_engine import UniverseEngine


class HistoricalDataEngine:

    def __init__(self, config_path=None):

        self.base_dir = BASE_DIR

        self.portfolio_config_path = (
            self.base_dir
            / "config"
            / "portfolio_config.json"
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

        self.period = "5y"
        self.interval = "1d"

        self.historical_path = (
            self.base_dir
            / "data"
            / "historical"
        )

        self.historical_path.mkdir(
            parents=True,
            exist_ok=True
        )

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

    def fetch(self, symbol):

        print(
            f"[HISTORICAL] Fetching {symbol}..."
        )

        data = yf.Ticker(symbol).history(
            period=self.period,
            interval=self.interval,
            auto_adjust=True
        )

        if data.empty:

            print(
                f"[WARNING] No historical data for {symbol}"
            )

            return None

        data = data.reset_index()

        data["Symbol"] = symbol

        return data

    def save(self, symbol, data):

        filename = (
            symbol.replace(".", "_")
            + "_historical.csv"
        )

        filepath = (
            self.historical_path
            / filename
        )

        data.to_csv(
            filepath,
            index=False
        )

        print(
            f"[HISTORICAL] Saved {symbol} "
            f"| Rows: {len(data)}"
        )

    def update(self):

        print("=" * 70)
        print("QUANTFORGE DYNAMIC HISTORICAL DATA ENGINE")
        print("=" * 70)

        print(
            f"Portfolio: "
            f"{self.config['portfolio_name']}"
        )

        print(
            f"Universe: "
            f"{self.universe_type}"
        )

        print(
            f"Universe Size: "
            f"{len(self.symbols)}"
        )

        print(
            f"Historical Period: "
            f"{self.period}"
        )

        print(
            f"Interval: "
            f"{self.interval}"
        )

        print("=" * 70)

        start_time = datetime.now()

        successful = 0
        failed = 0

        for index, symbol in enumerate(
            self.symbols,
            start=1
        ):

            print(
                f"\n[{index}/{len(self.symbols)}] "
                f"{symbol}"
            )

            try:

                data = self.fetch(symbol)

                if data is not None:

                    self.save(
                        symbol,
                        data
                    )

                    successful += 1

                else:

                    failed += 1

            except Exception as error:

                failed += 1

                print(
                    f"[ERROR] {symbol}: {error}"
                )

            time.sleep(0.15)

        duration = (
            datetime.now() - start_time
        )

        print()
        print("=" * 70)
        print("HISTORICAL DATA UPDATE COMPLETED")
        print("=" * 70)

        print(
            f"Universe: {self.universe_type}"
        )

        print(
            f"Requested Symbols: "
            f"{len(self.symbols)}"
        )

        print(
            f"Successful: {successful}"
        )

        print(
            f"Failed: {failed}"
        )

        print(
            f"Duration: {duration}"
        )

        print("=" * 70)


if __name__ == "__main__":

    engine = HistoricalDataEngine()

    engine.update()