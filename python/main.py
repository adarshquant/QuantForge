import time
from pathlib import Path

from data.market_data import MarketDataEngine
from features.feature_engine import FeatureEngine
from signals.scanner import StockScanner


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"


class QuantForge:

    def __init__(self):

        self.data_engine = MarketDataEngine(
            CONFIG_PATH
        )

        self.feature_engine = FeatureEngine(
            RAW_DATA_PATH,
            PROCESSED_DATA_PATH
        )

        self.scanner = StockScanner(
            CONFIG_PATH
        )

        self.symbols = self.data_engine.symbols

        self.interval = (
            self.data_engine.config["engine"]
            ["update_interval_minutes"] * 60
        )

        self.running = True

    def run_cycle(self):

        print()
        print("=" * 70)
        print("QUANTFORGE CYCLE")
        print("=" * 70)

        self.data_engine.update()

        self.feature_engine.process_all(
            self.symbols
        )

        self.scanner.scan()

        print()
        print("[CORE] Cycle completed")
        print(
            f"[CORE] Next cycle in "
            f"{self.interval // 60} minutes"
        )

    def run(self):

        print("=" * 70)
        print("QUANTFORGE AUTONOMOUS ENGINE")
        print("=" * 70)

        print("[CORE] Engine online")
        print("[DATA] Market data engine online")
        print("[FEATURE] Feature engine online")
        print("[SIGNAL] Stock scanner online")
        print("[CORE] Autonomous mode enabled")

        print("=" * 70)

        while self.running:

            try:

                self.run_cycle()

                time.sleep(self.interval)

            except KeyboardInterrupt:

                self.running = False

                print()
                print("[CORE] QuantForge stopped")

            except Exception as error:

                print(
                    f"[ERROR] {error}"
                )

                time.sleep(30)


if __name__ == "__main__":

    engine = QuantForge()

    engine.run()