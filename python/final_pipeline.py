import os
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

PYTHON = (
    BASE_DIR
    / ".venv"
    / "Scripts"
    / "python.exe"
)

STAGES = [
    (
        "DYNAMIC HISTORICAL DATA",
        BASE_DIR
        / "python"
        / "data"
        / "historical_data.py"
    ),
    (
        "DYNAMIC HISTORICAL FEATURES",
        BASE_DIR
        / "python"
        / "features"
        / "historical_feature_engine.py"
    ),
    (
        "CROSS-SECTIONAL DATASET",
        BASE_DIR
        / "python"
        / "models"
        / "cross_sectional_dataset.py"
    ),
    (
        "ML PREDICTIONS",
        BASE_DIR
        / "python"
        / "models"
        / "generate_predictions.py"
    ),
    (
        "ML BACKTEST BRIDGE",
        BASE_DIR
        / "python"
        / "portfolio"
        / "ml_backtest_bridge.py"
    ),
    (
        "HISTORICAL MARKET REGIME",
        BASE_DIR
        / "python"
        / "features"
        / "historical_market_regime.py"
    ),
    (
        "SIGNAL ENSEMBLE",
        BASE_DIR
        / "python"
        / "models"
        / "signal_ensemble.py"
    ),
    (
        "RISK ALLOCATION",
        BASE_DIR
        / "python"
        / "portfolio"
        / "risk_allocator.py"
    ),
    (
        "REGIME ADJUSTED PORTFOLIO",
        BASE_DIR
        / "python"
        / "portfolio"
        / "regime_adjusted_builder.py"
    ),
    (
        "FINAL BACKTEST INPUT",
        BASE_DIR
        / "python"
        / "portfolio"
        / "final_backtest_builder.py"
    ),
    (
        "WALK FORWARD VALIDATION",
        BASE_DIR
        / "python"
        / "models"
        / "walk_forward_validator.py"
    ),
    (
        "MONTE CARLO RISK",
        BASE_DIR
        / "python"
        / "risk"
        / "monte_carlo_engine.py"
    )
]


def run_stage(name, script):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    if not script.exists():

        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    result = subprocess.run(
        [
            str(PYTHON),
            str(script)
        ],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"{name} failed with exit code "
            f"{result.returncode}"
        )


def main():

    print("=" * 80)
    print(
        "QUANTFORGE DYNAMIC RESEARCH PIPELINE"
    )
    print("=" * 80)

    print(
        f"Python: {PYTHON}"
    )

    if not PYTHON.exists():

        raise FileNotFoundError(
            f"Virtual environment Python not found: "
            f"{PYTHON}"
        )

    for name, script in STAGES:

        run_stage(
            name,
            script
        )

    print()
    print("=" * 80)
    print(
        "C++ HIGH-PERFORMANCE BACKTEST"
    )
    print("=" * 80)

    executable = (
        BASE_DIR
        / "cpp"
        / "backtester.exe"
    )

    if not executable.exists():

        raise FileNotFoundError(
            f"C++ executable not found: "
            f"{executable}"
        )

    cpp_env = os.environ.copy()

    cpp_env["PATH"] = (
        r"C:\msys64\ucrt64\bin"
        + os.pathsep
        + cpp_env.get("PATH", "")
    )

    result = subprocess.run(
        [
            str(executable)
        ],
        cwd=BASE_DIR,
        env=cpp_env
    )

    if result.returncode != 0:

        raise RuntimeError(
            "C++ backtester failed with "
            f"exit code {result.returncode}"
        )

    print()
    print("=" * 80)
    print(
        "PERFORMANCE REPORT"
    )
    print("=" * 80)

    performance_script = (
        BASE_DIR
        / "python"
        / "risk"
        / "performance_report.py"
    )

    run_stage(
        "PERFORMANCE REPORT",
        performance_script
    )

    print()
    print("=" * 80)
    print(
        "QUANTFORGE DYNAMIC PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)


if __name__ == "__main__":

    main()