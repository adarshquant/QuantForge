import json
import os
import sys
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from python.data.universe_engine import UniverseEngine


CONFIG_PATH = BASE_DIR / "config" / "portfolio_config.json"


class QuantForgeHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            directory=str(BASE_DIR),
            **kwargs
        )

    def send_json(self, status_code, data):
        payload = json.dumps(data).encode("utf-8")

        self.send_response(status_code)
        self.send_header(
            "Content-Type",
            "application/json"
        )
        self.send_header(
            "Content-Length",
            str(len(payload))
        )
        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.end_headers()

        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.end_headers()

    def do_GET(self):

        if self.path == "/api/dashboard":

            try:

                from python.dashboard.data_api import (
                    get_dashboard_data
                )

                self.send_json(
                    200,
                    get_dashboard_data()
                )

                return

            except Exception as error:

                self.send_json(
                    500,
                    {
                        "status": "error",
                        "message": str(error)
                    }
                )

                return

        super().do_GET()


    def do_POST(self):

        if self.path != "/api/portfolio":
            self.send_json(
                404,
                {
                    "status": "error",
                    "message": "API endpoint not found"
                }
            )
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                content_length
            )

            portfolio = json.loads(
                body.decode("utf-8")
            )

            self.validate_portfolio(
                portfolio
            )

            with open(
                CONFIG_PATH,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(
                    portfolio,
                    file,
                    indent=2
                )

            self.send_json(
                200,
                {
                    "status": "success",
                    "message": "Portfolio configuration saved successfully",
                    "portfolio": portfolio
                }
            )

            print(
                f"PORTFOLIO SAVED | "
                f"{portfolio['portfolio_name']} | "
                f"{portfolio['universe_type']} | "
                f"{len(portfolio['custom_universe'])} stocks"
            )

        except Exception as error:

            print(
                f"PORTFOLIO ERROR | {error}"
            )

            self.send_json(
                400,
                {
                    "status": "error",
                    "message": str(error)
                }
            )

    def validate_portfolio(self, portfolio):

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
            key
            for key in required
            if key not in portfolio
        ]

        if missing:
            raise ValueError(
                f"Missing fields: {missing}"
            )

        if not portfolio["portfolio_name"].strip():
            raise ValueError(
                "Portfolio name cannot be empty"
            )

        if float(
            portfolio["initial_capital"]
        ) <= 0:
            raise ValueError(
                "Initial capital must be greater than zero"
            )

        if not isinstance(
            portfolio["custom_universe"],
            list
        ):
            raise ValueError(
                "Universe must be a list"
            )

        if len(
            portfolio["custom_universe"]
        ) == 0:
            raise ValueError(
                "Portfolio universe cannot be empty"
            )

        minimum = float(
            portfolio["min_position_weight"]
        )

        maximum = float(
            portfolio["max_position_weight"]
        )

        if minimum <= 0:
            raise ValueError(
                "Minimum position weight must be greater than zero"
            )

        if maximum <= 0:
            raise ValueError(
                "Maximum position weight must be greater than zero"
            )

        if minimum > maximum:
            raise ValueError(
                "Minimum position weight cannot exceed maximum position weight"
            )

        if int(
            portfolio["max_positions"]
        ) <= 0:
            raise ValueError(
                "Maximum positions must be greater than zero"
            )

        if float(
            portfolio["transaction_cost"]
        ) < 0:
            raise ValueError(
                "Transaction cost cannot be negative"
            )

        if float(
            portfolio["slippage"]
        ) < 0:
            raise ValueError(
                "Slippage cannot be negative"
            )

        universe_type = (
            portfolio["universe_type"]
            .upper()
        )

        universe_engine = UniverseEngine()

        if universe_type == "CUSTOM":

            symbols = portfolio[
                "custom_universe"
            ]

            if not symbols:
                raise ValueError(
                    "Custom universe cannot be empty"
                )

        else:

            if universe_type not in [
                "NIFTY_50",
                "NIFTY_100",
                "NIFTY_150"
            ]:
                raise ValueError(
                    f"Unsupported universe: {universe_type}"
                )

            universe_engine.validate(
                universe_type
            )


def main():

    server = ThreadingHTTPServer(
        ("0.0.0.0", int(os.environ.get("PORT", 8000))),
        QuantForgeHandler
    )

    print("=" * 70)
    print("QUANTFORGE LOCAL RESEARCH SERVER")
    print("=" * 70)
    print(
        "Dashboard: "
        "http://127.0.0.1:8000/python/dashboard/"
    )
    print(
        "API:       "
        "http://127.0.0.1:8000/api/portfolio"
    )
    print("=" * 70)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("QUANTFORGE SERVER STOPPED")
        server.server_close()


if __name__ == "__main__":
    main()