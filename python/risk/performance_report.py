from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "cpp_equity_curve.csv"
REPORT_PATH = BASE_DIR / "data" / "processed" / "performance_report.csv"
MONTHLY_PATH = BASE_DIR / "data" / "processed" / "monthly_returns.csv"

INITIAL_CAPITAL = 1_000_000.0


def max_drawdown(equity):
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return drawdown.min()


def calculate_metrics(df):
    returns = df["Net_Return"].astype(float)
    equity = df["Equity"].astype(float)

    total_return = equity.iloc[-1] / INITIAL_CAPITAL - 1.0

    periods = len(df)

    if periods > 1:
        cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (252.0 / periods) - 1.0
    else:
        cagr = 0.0

    volatility = returns.std(ddof=0) * np.sqrt(252.0)

    if returns.std(ddof=0) > 0:
        sharpe = returns.mean() / returns.std(ddof=0) * np.sqrt(252.0)
    else:
        sharpe = 0.0

    downside = returns[returns < 0]

    if len(downside) > 0 and downside.std(ddof=0) > 0:
        sortino = returns.mean() / downside.std(ddof=0) * np.sqrt(252.0)
    else:
        sortino = 0.0

    mdd = max_drawdown(equity)

    if mdd != 0:
        calmar = cagr / abs(mdd)
    else:
        calmar = 0.0

    winning = returns[returns > 0]
    losing = returns[returns < 0]

    win_rate = len(winning) / len(returns) if len(returns) else 0.0

    gross_profit = winning.sum()
    gross_loss = abs(losing.sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 0.0
    )

    best_day = returns.max()
    worst_day = returns.min()

    total_turnover = df["Turnover"].sum()

    transaction_cost_rate = (
        df["Transaction_Cost"].sum()
    )

    slippage_rate = (
        df["Slippage_Cost"].sum()
    )

    metrics = {
        "Initial Capital": INITIAL_CAPITAL,
        "Final Equity": equity.iloc[-1],
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": mdd,
        "Calmar Ratio": calmar,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Best Day": best_day,
        "Worst Day": worst_day,
        "Trading Sessions": len(df),
        "Total Turnover": total_turnover,
        "Transaction Cost Rate": transaction_cost_rate,
        "Slippage Rate": slippage_rate
    }

    return metrics


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing C++ equity curve: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "Date",
        "Gross_Return",
        "Turnover",
        "Transaction_Cost",
        "Slippage_Cost",
        "Net_Return",
        "Equity",
        "Drawdown"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in C++ equity curve: {missing}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    numeric_columns = [
        "Gross_Return",
        "Turnover",
        "Transaction_Cost",
        "Slippage_Cost",
        "Net_Return",
        "Equity",
        "Drawdown"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_columns
    ).reset_index(drop=True)

    df["Peak_Equity"] = df["Equity"].cummax()
    df["Drawdown"] = (
        df["Equity"] / df["Peak_Equity"] - 1.0
    )

    metrics = calculate_metrics(df)

    report = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        }
    )

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    monthly = (
        df.set_index("Date")["Net_Return"]
        .resample("ME")
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Return"
    ]

    monthly.to_csv(
        MONTHLY_PATH,
        index=False
    )

    equity_output = BASE_DIR / "data" / "processed" / "equity_curve.csv"

    df[
        [
            "Date",
            "Gross_Return",
            "Turnover",
            "Transaction_Cost",
            "Slippage_Cost",
            "Net_Return",
            "Equity",
            "Drawdown"
        ]
    ].to_csv(
        equity_output,
        index=False
    )

    print("=" * 80)
    print("QUANTFORGE PERFORMANCE ANALYTICS")
    print("=" * 80)

    print()
    print(f"Initial Capital       : ₹{metrics['Initial Capital']:,.2f}")
    print(f"Final Equity          : ₹{metrics['Final Equity']:,.2f}")
    print(f"Total Return          : {metrics['Total Return']:.2%}")
    print(f"CAGR                  : {metrics['CAGR']:.2%}")
    print(f"Annualized Volatility : {metrics['Annualized Volatility']:.2%}")
    print(f"Sharpe Ratio          : {metrics['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio         : {metrics['Sortino Ratio']:.2f}")
    print(f"Maximum Drawdown      : {metrics['Maximum Drawdown']:.2%}")
    print(f"Calmar Ratio          : {metrics['Calmar Ratio']:.2f}")
    print(f"Win Rate              : {metrics['Win Rate']:.2%}")
    print(f"Profit Factor         : {metrics['Profit Factor']:.2f}")
    print(f"Best Day              : {metrics['Best Day']:.2%}")
    print(f"Worst Day             : {metrics['Worst Day']:.2%}")
    print(f"Trading Sessions      : {metrics['Trading Sessions']}")
    print(f"Total Turnover        : {metrics['Total Turnover']:.2%}")
    print(
        f"Transaction Cost Rate : "
        f"{metrics['Transaction Cost Rate']:.2%}"
    )
    print(
        f"Slippage Rate         : "
        f"{metrics['Slippage Rate']:.2%}"
    )

    print()
    print(f"Equity Curve          : {equity_output}")
    print(f"Monthly Data          : {MONTHLY_PATH}")
    print(f"Report                : {REPORT_PATH}")

    print()
    print("PERFORMANCE ANALYTICS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()