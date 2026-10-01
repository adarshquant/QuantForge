from pathlib import Path
import json
import sys
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
PROCESSED = BASE_DIR / "data" / "processed"
CONFIG_PATH = BASE_DIR / "config" / "portfolio_config.json"
INPUT_PATH = PROCESSED / "final_backtest_input.csv"


def resolve_symbols(config):
    universe_type = config.get("universe_type", "NIFTY_50")

    if universe_type == "CUSTOM":
        return set(config.get("custom_universe", []))

    from python.data.universe_engine import UniverseEngine

    engine = UniverseEngine()
    symbols = engine.get_universe(universe_type)
    return set(symbols)


def apply_weight_limits(weights, min_weight, max_weight):
    weights = weights.copy()

    if weights.empty:
        return weights

    weights = weights.clip(lower=min_weight, upper=max_weight)

    for _ in range(20):
        total = weights.sum()

        if total <= 0:
            weights[:] = 1.0 / len(weights)
            break

        weights = weights / total

        over = weights > max_weight
        under = weights < min_weight

        if not over.any() and not under.any():
            break

        fixed = weights.copy()
        fixed[over] = max_weight

        remaining_symbols = ~(over | under)

        if remaining_symbols.any():
            remaining = 1.0 - fixed[over].sum() - min_weight * under.sum()
            if remaining > 0:
                base = weights[remaining_symbols]
                fixed[remaining_symbols] = base / base.sum() * remaining

        fixed[under] = min_weight
        weights = fixed

    return weights / weights.sum()


def run_portfolio_backtest():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Symbol"] = df["Symbol"].astype(str)

    numeric_columns = [
        "Return_1D",
        "Predicted_Alpha",
        "Prediction_Rank",
        "Position_Weight"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", "Symbol", "Return_1D", "Prediction_Rank"])

    selected_symbols = resolve_symbols(config)

    df = df[df["Symbol"].isin(selected_symbols)].copy()

    if df.empty:
        raise ValueError("Selected portfolio universe has no matching symbols in final_backtest_input.csv")

    max_positions = int(config.get("max_positions", 10))
    min_weight = float(config.get("min_position_weight", 0.05))
    max_weight = float(config.get("max_position_weight", 0.35))
    transaction_cost = float(config.get("transaction_cost", 0.0005))
    slippage = float(config.get("slippage", 0.0005))
    initial_capital = float(config.get("initial_capital", 1000000))

    rows = []
    previous_weights = {}

    for date, day in df.groupby("Date", sort=True):
        day = day.sort_values(
            ["Prediction_Rank", "Predicted_Alpha"],
            ascending=[True, False]
        ).head(max_positions).copy()

        if day.empty:
            continue

        raw_weights = day["Position_Weight"].fillna(0).clip(lower=0)

        if raw_weights.sum() <= 0:
            raw_weights = pd.Series(
                1.0,
                index=day.index
            )

        weights = pd.Series(
            raw_weights.values,
            index=day["Symbol"].values,
            dtype=float
        )

        weights = apply_weight_limits(
            weights,
            min_weight,
            max_weight
        )

        current_weights = weights.to_dict()

        all_symbols = set(previous_weights) | set(current_weights)

        turnover = sum(
            abs(
                current_weights.get(symbol, 0.0)
                - previous_weights.get(symbol, 0.0)
            )
            for symbol in all_symbols
        )

        weight_map = day["Symbol"].map(current_weights).fillna(0)

        gross_return = float(
            (weight_map * day["Return_1D"]).sum()
        )

        cost = turnover * transaction_cost
        slip = turnover * slippage
        net_return = gross_return - cost - slip

        rows.append({
            "Date": date,
            "Gross_Return": gross_return,
            "Transaction_Cost": cost,
            "Slippage": slip,
            "Net_Return": net_return,
            "Turnover": turnover,
            "Positions": len(current_weights)
        })

        previous_weights = current_weights

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("Backtest produced no trading sessions")

    result = result.sort_values("Date").reset_index(drop=True)

    result["Equity"] = initial_capital * (
        1.0 + result["Net_Return"]
    ).cumprod()

    running_peak = result["Equity"].cummax()

    result["Drawdown"] = (
        result["Equity"] / running_peak - 1.0
    )

    result["Cumulative_Return"] = (
        result["Equity"] / initial_capital - 1.0
    )

    result.to_csv(
        PROCESSED / "cpp_equity_curve.csv",
        index=False
    )

    monthly = (
        result.set_index("Date")["Net_Return"]
        .resample("ME")
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        .reset_index()
    )

    monthly.columns = ["Month", "Return"]

    monthly.to_csv(
        PROCESSED / "monthly_returns.csv",
        index=False
    )

    total_return = result["Equity"].iloc[-1] / initial_capital - 1.0

    years = max(
        (result["Date"].iloc[-1] - result["Date"].iloc[0]).days / 365.25,
        1 / 365.25
    )

    cagr = (
        (result["Equity"].iloc[-1] / initial_capital) ** (1 / years)
        - 1.0
    )

    volatility = result["Net_Return"].std(ddof=1) * np.sqrt(252)

    sharpe = (
        result["Net_Return"].mean()
        / result["Net_Return"].std(ddof=1)
        * np.sqrt(252)
        if result["Net_Return"].std(ddof=1) > 0
        else 0.0
    )

    downside = result.loc[
        result["Net_Return"] < 0,
        "Net_Return"
    ].std(ddof=1)

    sortino = (
        result["Net_Return"].mean()
        / downside
        * np.sqrt(252)
        if pd.notna(downside) and downside > 0
        else 0.0
    )

    max_drawdown = result["Drawdown"].min()

    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown < 0
        else 0.0
    )

    winning = result.loc[
        result["Net_Return"] > 0,
        "Net_Return"
    ]

    losing = result.loc[
        result["Net_Return"] < 0,
        "Net_Return"
    ]

    profit_factor = (
        winning.sum() / abs(losing.sum())
        if losing.sum() != 0
        else 0.0
    )

    win_rate = (
        (result["Net_Return"] > 0).mean()
    )

    metrics = pd.DataFrame([{
        "Metric": "Initial Capital",
        "Value": initial_capital
    }, {
        "Metric": "Final Capital",
        "Value": result["Equity"].iloc[-1]
    }, {
        "Metric": "Total Return",
        "Value": total_return
    }, {
        "Metric": "CAGR",
        "Value": cagr
    }, {
        "Metric": "Annualized Volatility",
        "Value": volatility
    }, {
        "Metric": "Sharpe Ratio",
        "Value": sharpe
    }, {
        "Metric": "Sortino Ratio",
        "Value": sortino
    }, {
        "Metric": "Maximum Drawdown",
        "Value": max_drawdown
    }, {
        "Metric": "Calmar Ratio",
        "Value": calmar
    }, {
        "Metric": "Win Rate",
        "Value": win_rate
    }, {
        "Metric": "Profit Factor",
        "Value": profit_factor
    }, {
        "Metric": "Sessions",
        "Value": len(result)
    }, {
        "Metric": "Total Turnover",
        "Value": result["Turnover"].sum()
    }, {
        "Metric": "Transaction Costs",
        "Value": result["Transaction_Cost"].sum()
    }, {
        "Metric": "Slippage",
        "Value": result["Slippage"].sum()
    }])

    metrics.to_csv(
        PROCESSED / "performance_report.csv",
        index=False
    )

    return {
        "sessions": len(result),
        "final_capital": float(result["Equity"].iloc[-1]),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "positions": max_positions,
        "universe_symbols": len(selected_symbols)
    }


if __name__ == "__main__":
    output = run_portfolio_backtest()
    print(json.dumps(output, indent=2))
