import pandas as pd
import yaml
from pathlib import Path

from regime_engine import RegimeEngine


class RegimeScanner:

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

        self.engine = RegimeEngine()

    def load_data(self, symbol):

        filename = symbol.replace(".", "_") + ".csv"

        filepath = self.data_path / filename

        if not filepath.exists():
            return None

        return pd.read_csv(filepath)

    def scan(self):

        results = []

        print("=" * 70)
        print("QUANTFORGE REGIME SCANNER")
        print("=" * 70)

        for symbol in self.symbols:

            try:

                data = self.load_data(symbol)

                if data is None:
                    continue

                regime = self.engine.calculate(data)

                results.append(
                    {
                        "Symbol": symbol,
                        "Regime": regime
                    }
                )

            except Exception as error:

                print(
                    f"[ERROR] {symbol}: {error}"
                )

        print()

        for result in results:

            print(
                f"{result['Symbol']:<18}"
                f"{result['Regime']}"
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

    scanner = RegimeScanner(config_path)

    scanner.scan()