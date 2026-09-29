import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


class PortfolioOptimizer:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.min_weight = 0.02
        self.max_weight = 0.35
        self.target_volatility = 0.18

    def load_data(self):

        data = pd.read_csv(self.dataset_path)

        data["Datetime"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        return data

    def prepare(self, data):

        latest = (
            data.sort_values("Datetime")
            .groupby("Symbol")
            .tail(1)
            .copy()
        )

        returns = (
            data.pivot(
                index="Datetime",
                columns="Symbol",
                values="Return_20D"
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna(how="all")
        )

        returns = returns.fillna(0)

        covariance = returns.cov() * 252

        symbols = latest["Symbol"].tolist()

        covariance = covariance.reindex(
            index=symbols,
            columns=symbols
        ).fillna(0)

        alpha = latest["Alpha_Rank"].values

        volatility = (
            latest["Volatility_20"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(latest["Volatility_20"].median())
            .values
        )

        return symbols, alpha, volatility, covariance.values

    def optimize(self, alpha, volatility, covariance):

        n = len(alpha)

        alpha = np.nan_to_num(alpha)
        volatility = np.nan_to_num(volatility)

        alpha = (
            alpha - alpha.mean()
        ) / (
            alpha.std() + 1e-8
        )

        def objective(weights):

            portfolio_return = np.dot(
                weights,
                alpha
            )

            portfolio_variance = (
                weights @ covariance @ weights
            )

            portfolio_volatility = np.sqrt(
                max(portfolio_variance, 0)
            )

            concentration = np.sum(
                weights ** 2
            )

            return (
                -portfolio_return
                + 0.35 * portfolio_volatility
                + 0.20 * concentration
            )

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            }
        ]

        bounds = [
            (
                self.min_weight,
                self.max_weight
            )
            for _ in range(n)
        ]

        initial = np.ones(n) / n

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 1000,
                "ftol": 1e-9
            }
        )

        if not result.success:
            return initial

        return result.x

    def build_portfolio(self):

        data = self.load_data()

        symbols, alpha, volatility, covariance = (
            self.prepare(data)
        )

        weights = self.optimize(
            alpha,
            volatility,
            covariance
        )

        portfolio = pd.DataFrame(
            {
                "Symbol": symbols,
                "Alpha": alpha,
                "Volatility": volatility,
                "Weight": weights
            }
        )

        portfolio = portfolio.sort_values(
            "Weight",
            ascending=False
        )

        portfolio["Allocation"] = (
            portfolio["Weight"] * 100
        ).round(2)

        return portfolio

    def run(self):

        print("=" * 70)
        print("QUANTFORGE PORTFOLIO OPTIMIZER")
        print("=" * 70)

        portfolio = self.build_portfolio()

        print()
        print("OPTIMIZED PORTFOLIO")
        print()

        for _, row in portfolio.iterrows():

            print(
                f"{row['Symbol']:<18}"
                f"Alpha: {row['Alpha']:>8.3f}   "
                f"Vol: {row['Volatility']:>8.3f}   "
                f"Weight: {row['Allocation']:>6.2f}%"
            )

        print()
        print(
            f"Total allocation: "
            f"{portfolio['Allocation'].sum():.2f}%"
        )

        print("=" * 70)

        return portfolio


if __name__ == "__main__":

    base_dir = (
        Path(__file__).resolve().parents[2]
    )

    dataset_path = (
        base_dir
        / "data"
        / "processed"
        / "cross_sectional_dataset.csv"
    )

    optimizer = PortfolioOptimizer(
        dataset_path
    )

    optimizer.run()