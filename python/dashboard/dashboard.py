from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

EQUITY_PATH = DATA_DIR / "equity_curve.csv"
MONTHLY_PATH = DATA_DIR / "monthly_returns.csv"
REPORT_PATH = DATA_DIR / "performance_report.csv"
CPP_EQUITY_PATH = DATA_DIR / "cpp_equity_curve.csv"

st.set_page_config(
    page_title="QuantForge | Quantitative Research",
    page_icon="Q",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        padding: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        background: rgba(128,128,128,0.04);
    }

    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .subtitle {
        color: #777;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.22);
        padding: 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None

equity = load_csv(EQUITY_PATH)
monthly = load_csv(MONTHLY_PATH)
report = load_csv(REPORT_PATH)
cpp_equity = load_csv(CPP_EQUITY_PATH)

st.title("QuantForge")
st.markdown(
    '<div class="subtitle">Autonomous Quantitative Market Intelligence & Trading Research Engine</div>',
    unsafe_allow_html=True
)

st.markdown("---")

if report.empty and equity.empty and cpp_equity.empty:
    st.error("No performance data found. Run the QuantForge research pipeline first.")
    st.stop()

metrics = {}

if not report.empty:
    if len(report.columns) >= 2:
        for _, row in report.iterrows():
            key = str(row.iloc[0]).strip()
            value = row.iloc[1]
            metrics[key] = value

def get_metric(names, default="N/A"):
    for name in names:
        if name in metrics:
            return metrics[name]
    return default

initial_capital = get_metric(["Initial Capital"])
final_equity = get_metric(["Final Equity"])
total_return = get_metric(["Total Return"])
cagr = get_metric(["CAGR"])
volatility = get_metric(["Annualized Volatility"])
sharpe = get_metric(["Sharpe Ratio"])
sortino = get_metric(["Sortino Ratio"])
max_dd = get_metric(["Maximum Drawdown"])
win_rate = get_metric(["Win Rate"])
profit_factor = get_metric(["Profit Factor"])
turnover = get_metric(["Total Turnover"])
transaction_cost = get_metric(["Transaction Costs"])
slippage_cost = get_metric(["Slippage Costs"])

st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Initial Capital", str(initial_capital))

with c2:
    st.metric("Final Equity", str(final_equity))

with c3:
    st.metric("Total Return", str(total_return))

with c4:
    st.metric("CAGR", str(cagr))

c5, c6, c7, c8 = st.columns(4)

with c5:
    st.metric("Annualized Volatility", str(volatility))

with c6:
    st.metric("Sharpe Ratio", str(sharpe))

with c7:
    st.metric("Maximum Drawdown", str(max_dd))

with c8:
    st.metric("Profit Factor", str(profit_factor))

st.markdown('<div class="section-title">Equity Curve</div>', unsafe_allow_html=True)

equity_source = cpp_equity if not cpp_equity.empty else equity

if not equity_source.empty:
    date_col = find_column(equity_source, ["Date", "date"])
    value_col = find_column(
        equity_source,
        ["Equity", "equity", "Portfolio_Value", "portfolio_value", "Capital"]
    )

    if date_col and value_col:
        equity_source[date_col] = pd.to_datetime(equity_source[date_col], errors="coerce")
        equity_source[value_col] = pd.to_numeric(
            equity_source[value_col],
            errors="coerce"
        )
        equity_source = equity_source.dropna(subset=[date_col, value_col])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=equity_source[date_col],
                y=equity_source[value_col],
                mode="lines",
                name="Portfolio Equity",
                line=dict(width=2)
            )
        )

        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-title">Risk & Execution Profile</div>', unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric("Win Rate", str(win_rate))

with r2:
    st.metric("Turnover", str(turnover))

with r3:
    st.metric("Transaction Costs", str(transaction_cost))

with r4:
    st.metric("Slippage Costs", str(slippage_cost))

st.markdown('<div class="section-title">Monthly Performance</div>', unsafe_allow_html=True)

if not monthly.empty:
    date_col = find_column(monthly, ["Date", "date", "Month", "month"])
    return_col = find_column(
        monthly,
        ["Return", "Monthly_Return", "monthly_return", "Portfolio_Return"]
    )

    if date_col and return_col:
        monthly[date_col] = pd.to_datetime(monthly[date_col], errors="coerce")
        monthly[return_col] = pd.to_numeric(
            monthly[return_col],
            errors="coerce"
        )
        monthly = monthly.dropna(subset=[date_col, return_col])

        fig_monthly = go.Figure()

        fig_monthly.add_trace(
            go.Bar(
                x=monthly[date_col],
                y=monthly[return_col],
                name="Monthly Return"
            )
        )

        fig_monthly.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Month",
            yaxis_title="Return",
            template="plotly_white"
        )

        st.plotly_chart(fig_monthly, use_container_width=True)

st.markdown('<div class="section-title">Performance Report</div>', unsafe_allow_html=True)

if not report.empty:
    display_report = report.copy()
    display_report.columns = ["Metric", "Value"] + list(
        display_report.columns[2:]
    )
    st.dataframe(
        display_report,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#777;font-size:13px;">
    QuantForge Research Engine · Python + C++ · ML · Regime Detection ·
    Risk Allocation · Monte Carlo · Walk-Forward Validation
    </div>
    """,
    unsafe_allow_html=True
)