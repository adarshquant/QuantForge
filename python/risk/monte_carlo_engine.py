from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_backtest_input.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "monte_carlo_results.csv"
)

INITIAL_CAPITAL = 1_000_000
SIMULATIONS = 5000
TRADING_DAYS = 252
BLOCK_SIZE = 5
RANDOM_SEED = 42


def load_portfolio_returns():

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "Date",
        "Symbol",
        "Return_1D",
        "Position_Weight"
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data["Return_1D"] = pd.to_numeric(
        data["Return_1D"],
        errors="coerce"
    )

    data["Position_Weight"] = pd.to_numeric(
        data["Position_Weight"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Date",
            "Return_1D",
            "Position_Weight"
        ]
    )

    data = data.sort_values(
        ["Date", "Symbol"]
    )

    data["Weighted_Return"] = (
        data["Return_1D"]
        * data["Position_Weight"]
    )

    portfolio_returns = (
        data.groupby("Date")["Weighted_Return"]
        .sum()
        .sort_index()
    )

    return portfolio_returns


def generate_block_sample(
    returns,
    rng
):

    values = returns.to_numpy()

    if len(values) < BLOCK_SIZE:
        raise ValueError(
            "Not enough observations for block bootstrap."
        )

    block_count = int(
        np.ceil(TRADING_DAYS / BLOCK_SIZE)
    )

    starts = rng.integers(
        0,
        len(values) - BLOCK_SIZE + 1,
        size=block_count
    )

    blocks = [
        values[
            start:start + BLOCK_SIZE
        ]
        for start in starts
    ]

    sample = np.concatenate(blocks)

    return sample[:TRADING_DAYS]


def calculate_max_drawdown(returns):

    equity = np.cumprod(
        1.0 + returns
    )

    peaks = np.maximum.accumulate(
        equity
    )

    drawdowns = (
        equity / peaks
    ) - 1.0

    return drawdowns.min()


def run_monte_carlo(returns):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    terminal_values = np.zeros(
        SIMULATIONS
    )

    maximum_drawdowns = np.zeros(
        SIMULATIONS
    )

    for simulation in range(SIMULATIONS):

        sampled_returns = generate_block_sample(
            returns,
            rng
        )

        equity_curve = (
            INITIAL_CAPITAL
            * np.cumprod(
                1.0 + sampled_returns
            )
        )

        terminal_values[simulation] = (
            equity_curve[-1]
        )

        maximum_drawdowns[simulation] = (
            calculate_max_drawdown(
                sampled_returns
            )
        )

    return (
        terminal_values,
        maximum_drawdowns
    )


def calculate_statistics(
    terminal_values,
    maximum_drawdowns
):

    losses = (
        terminal_values
        < INITIAL_CAPITAL
    )

    sorted_values = np.sort(
        terminal_values
    )

    var_index = int(
        np.floor(
            0.05 * len(sorted_values)
        )
    )

    var_value = (
        sorted_values[var_index]
    )

    tail = sorted_values[
        :var_index + 1
    ]

    cvar_value = tail.mean()

    statistics = {
        "Initial_Capital": INITIAL_CAPITAL,
        "Simulations": SIMULATIONS,
        "Horizon_Days": TRADING_DAYS,
        "Mean_Terminal_Wealth": terminal_values.mean(),
        "Median_Terminal_Wealth": np.median(
            terminal_values
        ),
        "P05_Terminal_Wealth": np.percentile(
            terminal_values,
            5
        ),
        "P25_Terminal_Wealth": np.percentile(
            terminal_values,
            25
        ),
        "P75_Terminal_Wealth": np.percentile(
            terminal_values,
            75
        ),
        "P95_Terminal_Wealth": np.percentile(
            terminal_values,
            95
        ),
        "Worst_Terminal_Wealth": terminal_values.min(),
        "Best_Terminal_Wealth": terminal_values.max(),
        "Probability_of_Loss": losses.mean(),
        "VaR_95_Terminal_Wealth": var_value,
        "CVaR_95_Terminal_Wealth": cvar_value,
        "Mean_Max_Drawdown": maximum_drawdowns.mean(),
        "Median_Max_Drawdown": np.median(
            maximum_drawdowns
        ),
        "Worst_Max_Drawdown": maximum_drawdowns.min(),
        "Probability_Drawdown_20pct": (
            maximum_drawdowns <= -0.20
        ).mean(),
        "Probability_Drawdown_30pct": (
            maximum_drawdowns <= -0.30
        ).mean()
    }

    return statistics


def main():

    print()
    print("=" * 70)
    print("QUANTFORGE MONTE CARLO RISK ENGINE")
    print("=" * 70)

    returns = load_portfolio_returns()

    print()
    print(
        f"Historical observations: {len(returns)}"
    )

    print(
        f"Simulation paths: {SIMULATIONS}"
    )

    print(
        f"Simulation horizon: {TRADING_DAYS} trading days"
    )

    print(
        f"Bootstrap block size: {BLOCK_SIZE} days"
    )

    terminal_values, maximum_drawdowns = (
        run_monte_carlo(returns)
    )

    statistics = calculate_statistics(
        terminal_values,
        maximum_drawdowns
    )

    results = pd.DataFrame(
        [statistics]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)

    print(
        f"Mean Terminal Wealth: "
        f"₹{statistics['Mean_Terminal_Wealth']:,.2f}"
    )

    print(
        f"Median Terminal Wealth: "
        f"₹{statistics['Median_Terminal_Wealth']:,.2f}"
    )

    print(
        f"5th Percentile Wealth: "
        f"₹{statistics['P05_Terminal_Wealth']:,.2f}"
    )

    print(
        f"95th Percentile Wealth: "
        f"₹{statistics['P95_Terminal_Wealth']:,.2f}"
    )

    print(
        f"Worst Simulated Wealth: "
        f"₹{statistics['Worst_Terminal_Wealth']:,.2f}"
    )

    print(
        f"Best Simulated Wealth: "
        f"₹{statistics['Best_Terminal_Wealth']:,.2f}"
    )

    print(
        f"Probability of Loss: "
        f"{statistics['Probability_of_Loss']:.2%}"
    )

    print(
        f"95% VaR Wealth: "
        f"₹{statistics['VaR_95_Terminal_Wealth']:,.2f}"
    )

    print(
        f"95% CVaR Wealth: "
        f"₹{statistics['CVaR_95_Terminal_Wealth']:,.2f}"
    )

    print(
        f"Mean Maximum Drawdown: "
        f"{statistics['Mean_Max_Drawdown']:.2%}"
    )

    print(
        f"Worst Maximum Drawdown: "
        f"{statistics['Worst_Max_Drawdown']:.2%}"
    )

    print(
        f"Probability Drawdown > 20%: "
        f"{statistics['Probability_Drawdown_20pct']:.2%}"
    )

    print(
        f"Probability Drawdown > 30%: "
        f"{statistics['Probability_Drawdown_30pct']:.2%}"
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print(
        "Monte Carlo simulation completed successfully."
    )


if __name__ == "__main__":

    main()