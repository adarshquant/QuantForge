import yfinance as yf
import yaml
from pathlib import Path
from datetime import datetime


class MarketDataEngine:

    def __init__(self, config_path):

        self.config_path = Path(config_path)

        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.symbols = self.config["universe"]
        self.period = self.config["live_data"]["period"]
        self.interval = self.config["live_data"]["interval"]

        self.raw_data_path = (
            self.config_path.parent.parent
            / "data"
            / "raw"
        )

        self.raw_data_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def fetch(self, symbol):

        print(f"[DATA] Fetching {symbol}...")

        data = yf.Ticker(symbol).history(
            period=self.period,
            interval=self.interval,
            auto_adjust=True
        )

        if data.empty:

            print(
                f"[WARNING] No data received for {symbol}"
            )

            return None

        return data.reset_index()

    def save(self, symbol, data):

        filename = symbol.replace(".", "_") + ".csv"

        filepath = self.raw_data_path / filename

        data.to_csv(
            filepath,
            index=False
        )

        print(f"[DATA] Saved {symbol}")

    def update(self):

        print("=" * 60)
        print("QUANTFORGE MARKET DATA UPDATE")
        print("=" * 60)

        start_time = datetime.now()

        for symbol in self.symbols:

            try:

                data = self.fetch(symbol)

                if data is not None:
                    self.save(symbol, data)

            except Exception as error:

                print(f"[ERROR] {symbol}: {error}")

        duration = datetime.now() - start_time

        print(f"[DATA] Update completed in {duration}")
        print("=" * 60)


if __name__ == "__main__":

    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "config.yaml"
    )

    engine = MarketDataEngine(config_path)

    engine.update()