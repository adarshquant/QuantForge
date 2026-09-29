import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    BASE_DIR
    / "config"
    / "portfolio_config.json"
)

PROCESSED = (
    BASE_DIR
    / "data"
    / "processed"
)


def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_csv(filename):

    path = PROCESSED / filename

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except Exception:

        return pd.DataFrame()


def get_dashboard_data():

    config = load_config()

    performance = load_csv(
        "performance_report.csv"
    )

    equity = load_csv(
        "cpp_equity_curve.csv"
    )

    ensemble = load_csv(
        "ensemble_signals.csv"
    )

    allocation = load_csv(
        "risk_allocated_predictions.csv"
    )

    monte_carlo = load_csv(
        "monte_carlo_results.csv"
    )

    data = {
        "portfolio": {
            "name": config.get(
                "portfolio_name"
            ),
            "capital": config.get(
                "initial_capital"
            ),
            "universe": config.get(
                "universe_type"
            ),
            "risk_profile": config.get(
                "risk_profile"
            ),
            "max_positions": config.get(
                "max_positions"
            )
        },
        "performance": {},
        "equity": [],
        "signals": [],
        "allocation": [],
        "monte_carlo": {}
    }

    if not performance.empty:

        row = performance.iloc[-1]

        for column in performance.columns:

            value = row[column]

            if pd.notna(value):

                if isinstance(
                    value,
                    (int, float)
                ):

                    data["performance"][
                        column
                    ] = float(value)

                else:

                    data["performance"][
                        column
                    ] = str(value)

    if not equity.empty:

        columns = [
            column
            for column in [
                "Date",
                "Equity",
                "Daily_Return"
            ]
            if column in equity.columns
        ]

        data["equity"] = (
            equity[columns]
            .tail(500)
            .fillna(0)
            .to_dict("records")
        )

    if not ensemble.empty:

        latest_date = (
            ensemble["Date"]
            .max()
        )

        latest = ensemble[
            ensemble["Date"] == latest_date
        ].copy()

        columns = [
            column
            for column in [
                "Symbol",
                "Ensemble_Score",
                "Ensemble_Rank",
                "Ensemble_Signal"
            ]
            if column in latest.columns
        ]

        data["signals"] = (
            latest[
                columns
            ]
            .sort_values(
                "Ensemble_Rank",
                ascending=False
            )
            .head(20)
            .fillna(0)
            .to_dict("records")
        )

    if not allocation.empty:

        latest_date = (
            allocation["Date"]
            .max()
        )

        latest = allocation[
            allocation["Date"] == latest_date
        ].copy()

        columns = [
            column
            for column in [
                "Symbol",
                "Position_Weight",
                "Risk_Adjusted_Score",
                "Rank"
            ]
            if column in latest.columns
        ]

        data["allocation"] = (
            latest[
                columns
            ]
            .sort_values(
                "Position_Weight",
                ascending=False
            )
            .head(
                int(
                    config.get(
                        "max_positions",
                        10
                    )
                )
            )
            .fillna(0)
            .to_dict("records")
        )

    if not monte_carlo.empty:

        row = monte_carlo.iloc[-1]

        for column in monte_carlo.columns:

            value = row[column]

            if pd.notna(value):

                if isinstance(
                    value,
                    (int, float)
                ):

                    data["monte_carlo"][
                        column
                    ] = float(value)

                else:

                    data["monte_carlo"][
                        column
                    ] = str(value)

    return data


if __name__ == "__main__":

    result = get_dashboard_data()

    print(
        json.dumps(
            result,
            indent=2,
            default=str
        )
    )