# QuantForge

## Autonomous Quantitative Market Intelligence & Trading Research Engine

QuantForge is a quantitative research and portfolio intelligence platform that combines machine learning, technical alpha, market regime detection, portfolio construction, risk management, Monte Carlo simulation, walk-forward validation, and high-performance C++ backtesting into a unified research pipeline.

The system uses a dynamic investment universe, allowing research to be performed across NIFTY 50, NIFTY 100, NIFTY 150, or a custom stock universe.

---

## Architecture

```text
Portfolio Builder
        ↓
Dynamic Universe Engine
        ↓
Historical Market Data
        ↓
Feature Engineering
        ↓
Cross-Sectional Alpha Dataset
        ↓
Machine Learning Alpha Model
        ↓
Signal Ensemble
        ↓
Market Regime Detection
        ↓
Risk Allocation
        ↓
Portfolio Construction
        ↓
Walk-Forward Validation
        ↓
Monte Carlo Risk Simulation
        ↓
C++ High-Performance Backtester
        ↓
Performance Analytics
        ↓
Quantitative Research Dashboard
```

---

## Core Features

### Dynamic Universe Engine

QuantForge supports:

- NIFTY 50
- NIFTY 100
- NIFTY 150
- Custom stock universes

The selected universe is propagated through the research pipeline instead of relying on a fixed stock list.

### Historical Market Data

The historical data engine retrieves market data using Yahoo Finance and stores it locally for research.

Features include:

- Multi-year historical data
- Daily market data
- Dynamic universe resolution
- Automatic symbol processing
- Missing-data handling

### Feature Engineering

QuantForge generates quantitative features including:

- 1D, 5D, 10D, 20D and 60D returns
- RSI
- MACD
- MACD signal
- MACD histogram
- SMA 10/20/50/100/200
- EMA 12/26
- Volatility
- Volume ratio
- Price relative to moving averages
- Momentum
- Trend strength
- Rolling high/low distance
- Price z-score
- Drawdown
- High-low range

### Cross-Sectional Alpha

Instead of predicting absolute returns alone, QuantForge evaluates securities relative to their cross-sectional peers.

The research dataset calculates:

- Future returns
- Cross-sectional returns
- Excess returns
- Alpha ranking
- Alpha classification

### Machine Learning Alpha Model

The current alpha model uses a Random Forest classifier.

The model includes:

- Time-based train/validation/test separation
- Class balancing
- Cross-sectional probability scoring
- Stock-level alpha ranking
- Model persistence using Joblib

### Signal Ensemble

The final research signal combines multiple components:

```text
Machine Learning Alpha
          +
Momentum
          +
Trend
          +
Volume
          +
Volatility Adjustment
          ↓
    Ensemble Score
```

The resulting signals are classified as:

- STRONG BUY
- BUY
- WATCH
- SELL
- STRONG SELL

### Market Regime Detection

QuantForge detects broad market conditions using:

- Market trend breadth
- Cross-sectional momentum
- Market volatility

Supported regimes:

- BULL
- SIDEWAYS
- BEAR
- HIGH VOLATILITY

Portfolio exposure can be adjusted according to the detected regime.

### Risk Management

The risk layer includes:

- Volatility-adjusted position sizing
- Position limits
- Portfolio exposure control
- Historical VaR
- CVaR
- Parametric risk
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Risk contribution
- Concentration analysis
- Stress testing

### Monte Carlo Simulation

QuantForge uses block-bootstrap Monte Carlo simulation to analyze:

- Terminal wealth distribution
- Downside scenarios
- Probability of loss
- Maximum drawdown distribution
- Tail risk
- Value-at-Risk
- Conditional Value-at-Risk

### Walk-Forward Validation

The model can be evaluated using rolling historical windows instead of relying only on a single static train/test split.

This provides sequential out-of-sample evaluation across different market periods.

### C++ Backtesting Engine

Performance-sensitive portfolio backtesting is implemented in C++17.

The engine models:

- Portfolio returns
- Position changes
- Turnover
- Transaction costs
- Slippage
- Equity compounding
- Maximum drawdown
- Sharpe ratio
- Win rate
- Profit factor

Python handles research orchestration while C++ handles the high-performance backtesting layer.

---

## Dashboard

QuantForge includes a browser-based quantitative research dashboard.

Dashboard sections include:

- Dashboard
- Portfolio Builder
- Performance
- Risk Analysis
- Monthly Returns
- Execution
- Research Pipeline
- Report

The dashboard receives generated QuantForge outputs through a local API.

---

## Technology Stack

### Programming

- Python
- C++17
- JavaScript
- HTML
- CSS

### Python Libraries

- pandas
- NumPy
- scikit-learn
- SciPy
- yfinance
- Joblib
- PyYAML

### Machine Learning

- Random Forest
- Cross-sectional ranking
- Time-based validation
- Walk-forward validation

### Quantitative Finance

- Alpha modeling
- Technical analysis
- Portfolio construction
- Risk management
- Monte Carlo simulation
- Market regime detection
- Transaction-cost modeling
- Slippage modeling

### Visualization

- Plotly
- Browser-based dashboard

---

## Project Structure

```text
QuantForge/
│
├── config/
│   └── portfolio_config.json
│
├── cpp/
│   ├── include/
│   └── src/
│       └── backtester.cpp
│
├── data/
│   ├── historical/
│   ├── processed/
│   └── raw/
│
├── python/
│   ├── alerts/
│   ├── dashboard/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── portfolio/
│   ├── risk/
│   ├── signals/
│   ├── audit.py
│   ├── final_pipeline.py
│   └── main.py
│
├── tests/
│
├── .gitignore
└── README.md
```

Generated market data and research artifacts are excluded from version control.

---

## Pipeline Outputs

The research pipeline generates artifacts such as:

```text
historical_features.csv
cross_sectional_dataset.csv
model_predictions.csv
ensemble_signals.csv
risk_allocated_predictions.csv
regime_adjusted_backtest_input.csv
final_backtest_input.csv
cpp_equity_curve.csv
performance_report.csv
monte_carlo_results.csv
walk_forward_results.csv
```

These generated datasets are excluded from Git version control through `.gitignore`.

---

## Installation

### Clone the repository

```powershell
git clone <YOUR_REPOSITORY_URL>
cd QuantForge
```

### Create virtual environment

```powershell
python -m venv .venv
```

### Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install yfinance pandas pyyaml scikit-learn scipy joblib
```

---

## Configuration

Portfolio configuration is stored in:

```text
config/portfolio_config.json
```

Example configuration:

```json
{
  "portfolio_name": "My QuantForge Portfolio",
  "initial_capital": 1000000,
  "universe_type": "NIFTY_100",
  "custom_universe": [],
  "strategy": "quantforge_alpha",
  "risk_profile": "balanced",
  "max_position_weight": 0.35,
  "min_position_weight": 0.05,
  "max_positions": 10,
  "transaction_cost": 0.0005,
  "slippage": 0.0005
}
```

Supported universe types:

```text
NIFTY_50
NIFTY_100
NIFTY_150
CUSTOM
```

---

## Running the Pipeline

Run the complete research pipeline:

```powershell
.\.venv\Scripts\python.exe .\python\final_pipeline.py
```

The pipeline automatically executes:

```text
Historical Data
        ↓
Feature Engineering
        ↓
Cross-Sectional Dataset
        ↓
ML Predictions
        ↓
Signal Ensemble
        ↓
Risk Allocation
        ↓
Market Regime
        ↓
Portfolio Construction
        ↓
Walk-Forward Validation
        ↓
Monte Carlo
        ↓
C++ Backtest
        ↓
Performance Report
```

---

## Production Audit

QuantForge includes an automated production audit:

```powershell
.\.venv\Scripts\python.exe .\python\audit.py
```

The audit checks the presence and integrity of the major research artifacts.

Current project audit:

```text
12 checks
12 passed
0 failed
```

---

## Dashboard

Start the local dashboard server:

```powershell
.\.venv\Scripts\python.exe .\python\dashboard\server.py
```

Open:

```text
http://127.0.0.1:8000/python/dashboard/
```

The dashboard exposes:

- Portfolio configuration
- Performance analytics
- Equity curve
- Signal information
- Risk allocation
- Research pipeline status
- Backtesting information

---

## Backtesting

The C++ engine performs risk-aware portfolio backtesting with:

- Daily portfolio returns
- Dynamic position weights
- Turnover calculation
- Transaction costs
- Slippage
- Equity curve generation

The resulting equity curve is consumed by the Python performance analytics layer.

---

## Research Methodology

QuantForge is designed around a research workflow emphasizing:

1. Dynamic universe selection
2. Historical market data
3. Feature engineering
4. Cross-sectional alpha research
5. Machine learning
6. Signal combination
7. Market regime analysis
8. Risk-aware allocation
9. Out-of-sample validation
10. Monte Carlo risk analysis
11. Transaction-cost-aware backtesting
12. Performance analysis

The system is intended to evaluate a strategy under historical and simulated conditions rather than assume that a model will perform consistently in the future.

---

## Validation

The current integrated system has been successfully validated across the complete pipeline.

```text
Dynamic Historical Data       PASS
Dynamic Feature Engine        PASS
Cross-Sectional Dataset       PASS
Machine Learning              PASS
Signal Ensemble               PASS
Risk Allocation               PASS
Regime Adjustment             PASS
Final Backtest                PASS
Walk-Forward Validation       PASS
Monte Carlo Simulation        PASS
C++ Backtesting               PASS
Production Audit              PASS
```

Production audit result:

```text
12 Passed
0 Failed
```

---

## Disclaimer

QuantForge is an educational and quantitative research project.

It is not financial advice.

Historical backtests, simulations, model predictions, and research results do not guarantee future investment performance.

---

## Author

**Adarsh Srivastava**

Quantitative Research • Machine Learning • Financial Modeling • Python • C++

---

## License

This project is currently intended as a personal research and portfolio project.