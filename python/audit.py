import json
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = BASE_DIR / "config" / "portfolio_config.json"

ARTIFACTS = {
    "Historical Data": BASE_DIR / "data" / "historical",
    "Historical Features": BASE_DIR / "data" / "processed" / "historical_features.csv",
    "Cross Sectional Dataset": BASE_DIR / "data" / "processed" / "cross_sectional_dataset.csv",
    "ML Predictions": BASE_DIR / "data" / "processed" / "model_predictions.csv",
    "Ensemble Signals": BASE_DIR / "data" / "processed" / "ensemble_signals.csv",
    "Risk Allocation": BASE_DIR / "data" / "processed" / "risk_allocated_predictions.csv",
    "Regime Portfolio": BASE_DIR / "data" / "processed" / "regime_adjusted_backtest_input.csv",
    "Final Backtest": BASE_DIR / "data" / "processed" / "final_backtest_input.csv",
    "C++ Equity Curve": BASE_DIR / "data" / "processed" / "cpp_equity_curve.csv",
    "Performance Report": BASE_DIR / "data" / "processed" / "performance_report.csv",
    "Monte Carlo": BASE_DIR / "data" / "processed" / "monte_carlo_results.csv",
    "Walk Forward": BASE_DIR / "data" / "processed" / "walk_forward_results.csv"
}

OLD_SYMBOLS = {
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK"
}


def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def check_file(name, path):

    if path.is_dir():

        files = list(
            path.glob("*_historical.csv")
        )

        return len(files) > 0, len(files)

    return path.exists(), None


def inspect_csv(path):

    try:

        data = pd.read_csv(path)

        if data.empty:
            return False, len(data), 0, []

        symbols = []

        if "Symbol" in data.columns:

            symbols = sorted(
                data["Symbol"]
                .dropna()
                .astype(str)
                .str.upper()
                .unique()
                .tolist()
            )

        return True, len(data), len(data.columns), symbols

    except Exception:

        return False, 0, 0, []


def main():

    config = load_config()

    universe_type = (
        config["universe_type"]
        .upper()
    )

    custom = [
        symbol.upper()
        for symbol in config.get(
            "custom_universe",
            []
        )
    ]

    print("=" * 80)
    print("QUANTFORGE PRODUCTION AUDIT")
    print("=" * 80)

    print(
        f"Portfolio: {config['portfolio_name']}"
    )

    print(
        f"Universe: {universe_type}"
    )

    print(
        f"Configured Custom Symbols: {len(custom)}"
    )

    print("=" * 80)

    passed = 0
    failed = 0

    for name, path in ARTIFACTS.items():

        exists, count = check_file(
            name,
            path
        )

        if exists:

            print(
                f"[PASS] {name}"
            )

            if count is not None:

                print(
                    f"       Historical files: {count}"
                )

            passed += 1

        else:

            print(
                f"[FAIL] {name}"
            )

            failed += 1

    print()
    print("=" * 80)
    print("DATASET INSPECTION")
    print("=" * 80)

    dataset_paths = [
        (
            "Historical Features",
            ARTIFACTS["Historical Features"]
        ),
        (
            "Cross Sectional Dataset",
            ARTIFACTS["Cross Sectional Dataset"]
        ),
        (
            "ML Predictions",
            ARTIFACTS["ML Predictions"]
        ),
        (
            "Ensemble Signals",
            ARTIFACTS["Ensemble Signals"]
        ),
        (
            "Risk Allocation",
            ARTIFACTS["Risk Allocation"]
        ),
        (
            "Final Backtest",
            ARTIFACTS["Final Backtest"]
        )
    ]

    all_symbols = set()

    for name, path in dataset_paths:

        if not path.exists():
            continue

        valid, rows, columns, symbols = (
            inspect_csv(path)
        )

        print(
            f"{name}: "
            f"{rows} rows | "
            f"{columns} columns | "
            f"{len(symbols)} symbols"
        )

        all_symbols.update(
            symbols
        )

        if not valid:

            print(
                f"[FAIL] {name} is empty or unreadable"
            )

            failed += 1

    stale_symbols = (
        all_symbols.intersection(
            OLD_SYMBOLS
        )
    )

    print()
    print("=" * 80)
    print("UNIVERSE CONSISTENCY")
    print("=" * 80)

    print(
        f"Symbols present across pipeline: "
        f"{len(all_symbols)}"
    )

    if stale_symbols:

        print(
            "[INFO] Legacy symbols detected:"
        )

        print(
            ", ".join(
                sorted(stale_symbols)
            )
        )

        print(
            "These are valid stocks and are not automatically an error."
        )

    else:

        print(
            "[PASS] No legacy fixed-universe symbols detected"
        )

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        f"Checks Passed: {passed}"
    )

    print(
        f"Checks Failed: {failed}"
    )

    if failed == 0:

        print(
            "STATUS: PRODUCTION AUDIT PASSED"
        )

    else:

        print(
            "STATUS: REVIEW REQUIRED"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()