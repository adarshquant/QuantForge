import numpy as np
import pandas as pd
from pathlib import Path


class RiskEngine:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.confidence = 0.95
        self.trading_days = 252
        self.initial_capital = 1000000

    def load_data(self):

        data = pd.read_csv(self.dataset_path)

        if "Return_1D" not in data.columns:
            raise ValueError("Return_1D column missing")

        data["Date"] = pd.to_datetime(data["Date"])

        returns = data.pivot_table(
            index="Date",
            columns="Symbol",
            values="Return_1D"
        )

        returns = returns.replace(
            [np.inf, -np.inf],
            np.nan
        )

        returns = returns.dropna(
            how="all"
        )

        returns = returns.fillna(0)

        return data, returns

    def build_weights(self, data, symbols):

        latest = (
            data.sort_values("Date")
            .groupby("Symbol")
            .tail(1)
            .set_index("Symbol")
        )

        alpha = latest["Alpha_Rank"].reindex(symbols).fillna(0)

        volatility = (
            latest["Volatility_20"]
            .reindex(symbols)
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(
                latest["Volatility_20"].median()
            )
        )

        score = (
            alpha
            / volatility.replace(0, np.nan)
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)

        if score.sum() == 0:
            weights = pd.Series(
                1 / len(symbols),
                index=symbols
            )
        else:
            weights = score / score.sum()

        weights = weights.clip(
            upper=0.35
        )

        weights = weights / weights.sum()

        return weights

    def portfolio_returns(self, returns, weights):

        aligned = returns.reindex(
            columns=weights.index
        )

        return aligned.dot(weights)

    def historical_var(self, portfolio_returns):

        var = np.percentile(
            portfolio_returns,
            (1 - self.confidence) * 100
        )

        return abs(var)

    def historical_cvar(self, portfolio_returns):

        var = np.percentile(
            portfolio_returns,
            (1 - self.confidence) * 100
        )

        tail = portfolio_returns[
            portfolio_returns <= var
        ]

        if tail.empty:
            return 0.0

        return abs(tail.mean())

    def parametric_var(self, portfolio_returns):

        mean = portfolio_returns.mean()
        std = portfolio_returns.std()

        z = 1.645

        var = mean - z * std

        return abs(var)

    def annualized_volatility(self, portfolio_returns):

        return (
            portfolio_returns.std()
            * np.sqrt(self.trading_days)
        )

    def sharpe_ratio(self, portfolio_returns):

        annual_return = (
            portfolio_returns.mean()
            * self.trading_days
        )

        annual_volatility = (
            portfolio_returns.std()
            * np.sqrt(self.trading_days)
        )

        if annual_volatility == 0:
            return 0.0

        return annual_return / annual_volatility

    def maximum_drawdown(self, portfolio_returns):

        wealth = (
            1 + portfolio_returns
        ).cumprod()

        peak = wealth.cummax()

        drawdown = (
            wealth / peak
        ) - 1

        return abs(drawdown.min())

    def concentration_risk(self, weights):

        hhi = (
            weights ** 2
        ).sum()

        effective_positions = (
            1 / hhi
            if hhi > 0
            else 0
        )

        return hhi, effective_positions

    def risk_contribution(
        self,
        returns,
        weights
    ):

        covariance = (
            returns.cov()
            * self.trading_days
        )

        vector = weights.values

        portfolio_variance = (
            vector
            @ covariance.values
            @ vector
        )

        portfolio_volatility = np.sqrt(
            portfolio_variance
        )

        marginal = (
            covariance.values
            @ vector
        )

        contribution = (
            vector
            * marginal
        )

        contribution = (
            contribution
            / portfolio_variance
        )

        return pd.Series(
            contribution,
            index=weights.index
        )

    def stress_test(
        self,
        returns,
        weights
    ):

        scenarios = {
            "Market Crash": -0.20,
            "Severe Crash": -0.30,
            "Banking Shock": -0.15,
            "Technology Shock": -0.15,
            "Broad Correction": -0.10
        }

        results = {}

        for name, shock in scenarios.items():

            stressed = (
                returns
                .tail(20)
                .mean()
            )

            stressed_return = (
                stressed * 0.5
                + shock * 0.5
            )

            portfolio_loss = (
                stressed_return
                * weights
            ).sum()

            results[name] = portfolio_loss

        return results

    def run(self):

        print("=" * 70)
        print("QUANTFORGE RISK ENGINE")
        print("=" * 70)

        data, returns = self.load_data()

        symbols = list(returns.columns)

        weights = self.build_weights(
            data,
            symbols
        )

        portfolio = self.portfolio_returns(
            returns,
            weights
        )

        var = self.historical_var(
            portfolio
        )

        cvar = self.historical_cvar(
            portfolio
        )

        parametric_var = self.parametric_var(
            portfolio
        )

        volatility = self.annualized_volatility(
            portfolio
        )

        sharpe = self.sharpe_ratio(
            portfolio
        )

        drawdown = self.maximum_drawdown(
            portfolio
        )

        hhi, effective_positions = (
            self.concentration_risk(
                weights
            )
        )

        contributions = self.risk_contribution(
            returns,
            weights
        )

        stress = self.stress_test(
            returns,
            weights
        )

        print()
        print("PORTFOLIO WEIGHTS")

        for symbol, weight in weights.items():

            print(
                f"{symbol:<18}"
                f"{weight * 100:>8.2f}%"
            )

        print()
        print("RISK METRICS")

        print(
            f"Historical VaR (95%): "
            f"{var * 100:.2f}%"
        )

        print(
            f"Historical CVaR (95%): "
            f"{cvar * 100:.2f}%"
        )

        print(
            f"Parametric VaR (95%): "
            f"{parametric_var * 100:.2f}%"
        )

        print(
            f"Annualized Volatility: "
            f"{volatility * 100:.2f}%"
        )

        print(
            f"Sharpe Ratio: "
            f"{sharpe:.3f}"
        )

        print(
            f"Maximum Drawdown: "
            f"{drawdown * 100:.2f}%"
        )

        print(
            f"HHI Concentration: "
            f"{hhi:.4f}"
        )

        print(
            f"Effective Positions: "
            f"{effective_positions:.2f}"
        )

        print()
        print("RISK CONTRIBUTION")

        for symbol, contribution in contributions.sort_values(
            ascending=False
        ).items():

            print(
                f"{symbol:<18}"
                f"{contribution * 100:>8.2f}%"
            )

        print()
        print("STRESS TESTS")

        for scenario, result in stress.items():

            print(
                f"{scenario:<22}"
                f"{result * 100:>8.2f}%"
            )

        print()
        print(
            f"VaR Capital Impact: "
            f"₹{self.initial_capital * var:,.0f}"
        )

        print(
            f"CVaR Capital Impact: "
            f"₹{self.initial_capital * cvar:,.0f}"
        )

        print()
        print("=" * 70)


if __name__ == "__main__":

    base_dir = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    dataset_path = (
        base_dir
        / "data"
        / "processed"
        / "cross_sectional_dataset.csv"
    )

    engine = RiskEngine(
        dataset_path
    )

    engine.run()