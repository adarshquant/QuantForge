import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from python.data.universe_engine import UniverseEngine


CONFIG_PATH = BASE_DIR / "config" / "portfolio_config.json"


class PortfolioEngine:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.universe_engine = UniverseEngine()
        self.validate_config()

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def validate_config(self):
        required = [
            "portfolio_name",
            "initial_capital",
            "universe_type",
            "custom_universe",
            "strategy",
            "risk_profile",
            "max_position_weight",
            "min_position_weight",
            "max_positions",
            "transaction_cost",
            "slippage"
        ]

        missing = [
            key for key in required
            if key not in self.config
        ]

        if missing:
            raise ValueError(f"Missing configuration fields: {missing}")

        if float(self.config["initial_capital"]) <= 0:
            raise ValueError("Initial capital must be greater than zero")

        minimum = float(self.config["min_position_weight"])
        maximum = float(self.config["max_position_weight"])

        if minimum <= 0:
            raise ValueError("Minimum position weight must be greater than zero")

        if maximum <= 0:
            raise ValueError("Maximum position weight must be greater than zero")

        if minimum > maximum:
            raise ValueError(
                "Minimum position weight cannot exceed maximum position weight"
            )

        if int(self.config["max_positions"]) <= 0:
            raise ValueError("Maximum positions must be greater than zero")

        if float(self.config["transaction_cost"]) < 0:
            raise ValueError("Transaction cost cannot be negative")

        if float(self.config["slippage"]) < 0:
            raise ValueError("Slippage cannot be negative")

        universe_type = self.config["universe_type"].upper()

        if universe_type == "CUSTOM":
            if not self.config["custom_universe"]:
                raise ValueError("Custom universe cannot be empty")
        else:
            self.universe_engine.validate(universe_type)

    def get_universe(self):
        universe_type = self.config["universe_type"].upper()

        if universe_type == "CUSTOM":
            return list(dict.fromkeys(
                self.config["custom_universe"]
            ))

        return self.universe_engine.get_symbols(universe_type)

    def get_portfolio(self):
        universe = self.get_universe()

        return {
            "portfolio_name": self.config["portfolio_name"],
            "initial_capital": float(
                self.config["initial_capital"]
            ),
            "universe_type": self.config["universe_type"],
            "universe": universe,
            "universe_size": len(universe),
            "strategy": self.config["strategy"],
            "risk_profile": self.config["risk_profile"],
            "max_position_weight": float(
                self.config["max_position_weight"]
            ),
            "min_position_weight": float(
                self.config["min_position_weight"]
            ),
            "max_positions": int(
                self.config["max_positions"]
            ),
            "transaction_cost": float(
                self.config["transaction_cost"]
            ),
            "slippage": float(
                self.config["slippage"]
            )
        }

    def save(self, portfolio):
        with open(
            self.config_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                portfolio,
                file,
                indent=2
            )

    def summary(self):
        portfolio = self.get_portfolio()

        print("=" * 70)
        print("QUANTFORGE PORTFOLIO ENGINE")
        print("=" * 70)
        print(f"Portfolio: {portfolio['portfolio_name']}")
        print(f"Capital: ₹{portfolio['initial_capital']:,.2f}")
        print(f"Universe: {portfolio['universe_type']}")
        print(f"Universe Size: {portfolio['universe_size']}")
        print(f"Strategy: {portfolio['strategy']}")
        print(f"Risk Profile: {portfolio['risk_profile']}")
        print(f"Max Positions: {portfolio['max_positions']}")
        print(
            f"Max Position Weight: "
            f"{portfolio['max_position_weight']:.2%}"
        )
        print(
            f"Min Position Weight: "
            f"{portfolio['min_position_weight']:.2%}"
        )
        print(
            f"Transaction Cost: "
            f"{portfolio['transaction_cost']:.2%}"
        )
        print(
            f"Slippage: "
            f"{portfolio['slippage']:.2%}"
        )
        print("=" * 70)
        print("Universe Preview:")

        for symbol in portfolio["universe"][:15]:
            print(f"  {symbol}")

        if portfolio["universe_size"] > 15:
            print(
                f"  ... and "
                f"{portfolio['universe_size'] - 15} more"
            )


if __name__ == "__main__":
    engine = PortfolioEngine()
    engine.summary()