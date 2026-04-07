"""
Monte Carlo Simulation for Stock Price Forecasting.

This module provides functionality to perform Monte Carlo simulations for stock price
predictions using Geometric Brownian Motion (GBM). The simulation generates multiple
potential price paths based on historical return distributions and calculates
statistical percentiles for future price predictions.

The module includes:
- Statistical analysis of return distributions (Shapiro-Wilk test)
- Multiple simulation runs using random walk with drift
- Percentile-based confidence intervals (5th, 16th, 84th, 95th)
- Visualization of Monte Carlo results
- Graph generation and export functionality

Typical usage:
    price_df, monte_carlo_df = monte_carlo_analysis(
        seed_number=42,
        stock_data_df=historical_data,
        forecast_df=forecast_data,
        years=5,
        sim_amount=1000
    )
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('future.no_silent_downcasting', True)

def monte_carlo_analysis(seed_number, stock_data_df, forecast_df, years, sim_amount):
    """
    Perform Monte Carlo simulation for stock price forecasting using Geometric Brownian Motion.
    
    This function simulates multiple potential future price paths based on historical return
    distributions. It uses the Shapiro-Wilk test to determine the distribution characteristics
    and calculates drift and shock parameters for the GBM model. The results include percentile-
    based confidence intervals and visualization of the simulation outcomes.
    
    Args:
        seed_number (int): Random seed for reproducibility of simulation results.
        stock_data_df (pd.DataFrame): Historical stock data containing at least closing prices
            in the 5th column (index 4) and a 'ticker' column.
        forecast_df (pd.DataFrame): Forecast data containing a 'close_Price' column used to
            calculate return statistics.
        years (int): Number of years to forecast into the future.
        sim_amount (int): Number of Monte Carlo simulation runs to perform.
    
    Returns:
        tuple: A tuple containing:
            - price_df (pd.DataFrame): Transposed DataFrame where each row represents a
              simulation run and columns represent daily prices over the forecast period.
            - monte_carlo_df (pd.DataFrame): DataFrame indexed by year containing statistical
              percentiles (5th, 16th, Mean, 84th, 95th) of simulated prices.
    
    Raises:
        FileNotFoundError: If the generated graph cannot be saved to the specified path.
    
    Notes:
        - Uses 252 trading days per year for calculations.
        - Applies Shapiro-Wilk test (α=0.05) to determine if returns are Gaussian.
        - Generates and saves a visualization graph to 'generated_graphs' directory.
        - Progress messages are printed every 250 simulation runs.
    """
    # print("seed_number", seed_number)
    # print("stock_data_df", stock_data_df)
    # print("forecast_df", forecast_df)
    # print("years", years)
    # print("sim_amount", sim_amount)
    np.random.seed(seed=seed_number)
    price_df = pd.DataFrame()
    # calculate amount of days into x future of years
    days = years * 252
    
    # Ensure we're working with a Series, not DataFrame
    close_prices = forecast_df["close_Price"]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]  # Take first column if DataFrame
    
    returns = np.log(1 + close_prices.pct_change().dropna()).infer_objects(copy=False)
    stat, p = stats.shapiro(returns)
    # print(stat, p)
    alpha = 0.05
    if p > alpha:
        mu = float(returns.mean())  # Ensure scalar
        print("Sample looks Gaussian (fail to reject H0)")
    else:
        mu = float(returns.median())  # Ensure scalar
        print("Sample does not look Gaussian (reject H0)")

    sigma = float(returns.std())  # Ensure scalar
    dt = years / days
    drift = (mu - (0.5 * sigma**2)) * dt
    shock = sigma * np.sqrt(dt)
    for run in range(sim_amount):
        price = np.zeros(days)
        price[0] = stock_data_df.iloc[-1][stock_data_df.columns[4]]
        for day in range(1, days):
            price[day] = price[day - 1] * np.exp(drift + shock * np.random.normal())

        price_sim_df = pd.DataFrame(price, columns=[f"Run_{run}"])
        price_df = pd.concat([price_df, price_sim_df], axis=1)
        if run % 250 == 0:
            print(f"Processed {run} runs, out of {sim_amount} runs.")

    price_df = price_df.transpose()
    year = [0]
    mean = [price_df[0].mean()]
    lower_percentile_2nd = [price_df[0].quantile(0.05)]
    lower_percentile_1st = [price_df[0].quantile(0.16)]
    upper_percentile_1st = [price_df[0].quantile(0.84)]
    upper_percentile_2nd = [price_df[0].quantile(0.95)]
    for i in range(len(price_df.columns)):
        i += 1
        if i % 252 == 0:
            year.append(int(i / 252))
            mean.append(price_df[i - 1].mean())
            lower_percentile_2nd.append(price_df[i - 1].quantile(0.05))
            lower_percentile_1st.append(price_df[i - 1].quantile(0.16))
            upper_percentile_1st.append(price_df[i - 1].quantile(0.84))
            upper_percentile_2nd.append(price_df[i - 1].quantile(0.95))

    monte_carlo_df = pd.DataFrame({
        "5th Percentile": lower_percentile_2nd, "16th Percentile": lower_percentile_1st,
        "Mean": mean, "84th Percentile": upper_percentile_1st, "95th Percentile": upper_percentile_2nd
    }, index=year)
    # print(monte_carlo_df)
    plt.figure(figsize=(18, 8))
    plt.plot(monte_carlo_df)
    plt.legend(monte_carlo_df, loc="best")
    plt.xlabel("Year")
    plt.ylabel("Opening price")
    plt.title("Monte Carlo Analysis for Stock opening price")
    stock_data_df = stock_data_df.replace({"ticker": [" ", "/"]}, {"ticker": "_"}, regex=True)
    stock_name = stock_data_df.iloc[0]["ticker"]
    graph_name = str(f"Monte_Carlo_Sim_of_{stock_name}.png")
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    # Save the graph
    try:
        plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
        plt.clf()
        plt.close("all")
    except FileNotFoundError:
        raise FileNotFoundError("The graph could not be saved. Please check the file name or path.")

    return price_df, monte_carlo_df


def portfolio_monte_carlo(
    weights,
    mean_returns,
    cov_matrix,
    initial_investment=100000,
    years=5,
    num_simulations=5000,
    seed=42
):
    """
    Run Monte Carlo simulation on an optimized portfolio using correlated GBM.

    Uses Cholesky decomposition to generate correlated random returns for all
    assets simultaneously, then applies portfolio weights to get portfolio-level
    value paths over the investment horizon.

    Args:
        weights: Array of portfolio weights (must sum to 1.0)
        mean_returns: Annualized mean log returns per asset (Series or array)
        cov_matrix: Annualized covariance matrix (DataFrame or 2D array)
        initial_investment: Starting portfolio value in currency units
        years: Number of years to simulate
        num_simulations: Number of simulation paths to generate
        seed: Random seed for reproducibility

    Returns:
        dict with keys:
            - 'portfolio_values': DataFrame (simulations x days) of portfolio values
            - 'yearly_stats': DataFrame indexed by year with percentile statistics
            - 'final_return_mean': Mean annualized return across simulations
            - 'final_return_p5': 5th percentile total return
            - 'final_return_p95': 95th percentile total return
            - 'final_value_mean': Mean final portfolio value
            - 'final_value_p5': 5th percentile final value
            - 'final_value_p95': 95th percentile final value
            - 'num_simulations': Number of simulations run
            - 'years': Investment horizon
    """
    np.random.seed(seed)

    weights = np.array(weights)
    mean_returns = np.array(mean_returns)
    cov_matrix = np.array(cov_matrix)

    num_assets = len(weights)
    days = years * 252
    dt = 1 / 252

    # Cholesky decomposition for correlated random variables
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If cov_matrix is not positive-definite, add small regularization
        print("[WARN] Covariance matrix not positive-definite, adding regularization")
        epsilon = 1e-6
        cov_matrix = cov_matrix + np.eye(num_assets) * epsilon
        L = np.linalg.cholesky(cov_matrix)

    # Calculate drift per asset per day
    drift = (mean_returns - 0.5 * np.diag(cov_matrix)) * dt

    print(f"\n[PORTFOLIO MC] Running {num_simulations:,} simulations over {years} years ({days} days)...")

    # Pre-allocate portfolio value matrix
    portfolio_values = np.zeros((num_simulations, days + 1))
    portfolio_values[:, 0] = initial_investment

    for sim in range(num_simulations):
        # Generate correlated random shocks for all days at once
        Z = np.random.standard_normal((days, num_assets))
        correlated_shocks = Z @ L.T  # Apply correlation structure

        # Daily returns for each asset
        daily_returns = drift + correlated_shocks * np.sqrt(dt)

        # Portfolio daily returns (weighted sum)
        portfolio_daily_returns = daily_returns @ weights

        # Compound the returns to get portfolio values
        cumulative_returns = np.exp(np.cumsum(portfolio_daily_returns))
        portfolio_values[sim, 1:] = initial_investment * cumulative_returns

        if sim % 1000 == 0 and sim > 0:
            print(f"   Simulated {sim:,} / {num_simulations:,} paths...")

    print(f"   Completed all {num_simulations:,} simulations.")

    # Build yearly statistics DataFrame
    yearly_data = []
    for yr in range(years + 1):
        day_idx = yr * 252 if yr > 0 else 0
        day_idx = min(day_idx, days)
        values_at_year = portfolio_values[:, day_idx]

        yearly_data.append({
            'Year': yr,
            '5th Percentile': np.percentile(values_at_year, 5),
            '16th Percentile': np.percentile(values_at_year, 16),
            'Mean': np.mean(values_at_year),
            '84th Percentile': np.percentile(values_at_year, 84),
            '95th Percentile': np.percentile(values_at_year, 95)
        })

    yearly_stats = pd.DataFrame(yearly_data).set_index('Year')

    # Final portfolio value statistics
    final_values = portfolio_values[:, -1]
    final_returns = (final_values / initial_investment - 1)

    result = {
        'portfolio_values': pd.DataFrame(portfolio_values),
        'yearly_stats': yearly_stats,
        'final_return_mean': float(np.mean(final_returns)),
        'final_return_p5': float(np.percentile(final_returns, 5)),
        'final_return_p95': float(np.percentile(final_returns, 95)),
        'final_value_mean': float(np.mean(final_values)),
        'final_value_p5': float(np.percentile(final_values, 5)),
        'final_value_p95': float(np.percentile(final_values, 95)),
        'num_simulations': num_simulations,
        'years': years
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"PORTFOLIO MONTE CARLO RESULTS ({years}-year horizon)")
    print(f"{'=' * 60}")
    print(f"Initial Investment:  {initial_investment:>12,.0f}")
    print(f"Mean Final Value:    {result['final_value_mean']:>12,.0f}")
    print(f"5th Percentile:      {result['final_value_p5']:>12,.0f}")
    print(f"95th Percentile:     {result['final_value_p95']:>12,.0f}")
    print(f"Mean Total Return:   {result['final_return_mean']:>11.2%}")
    print(f"5th Pct Return:      {result['final_return_p5']:>11.2%}")
    print(f"95th Pct Return:     {result['final_return_p95']:>11.2%}")
    print(f"{'=' * 60}")

    print(f"\nYearly Statistics:")
    print(yearly_stats.to_string())

    # Plot portfolio MC results
    _plot_portfolio_monte_carlo(yearly_stats, initial_investment, years)

    return result


def _plot_portfolio_monte_carlo(yearly_stats, initial_investment, years):
    """Generate and save portfolio Monte Carlo visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))

    x = yearly_stats.index

    # Fill confidence bands
    ax.fill_between(
        x, yearly_stats['5th Percentile'], yearly_stats['95th Percentile'],
        alpha=0.15, color='blue', label='5th-95th Percentile'
    )
    ax.fill_between(
        x, yearly_stats['16th Percentile'], yearly_stats['84th Percentile'],
        alpha=0.25, color='blue', label='16th-84th Percentile'
    )

    # Plot mean line
    ax.plot(x, yearly_stats['Mean'], 'b-', linewidth=2, label='Mean')

    # Plot initial investment reference
    ax.axhline(y=initial_investment, color='gray', linestyle='--', alpha=0.5, label='Initial Investment')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.set_title(f'Portfolio Monte Carlo Simulation ({years}-Year Horizon)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    graph_name = f"Portfolio_Monte_Carlo_{years}yr.png"

    try:
        plt.savefig(
            os.path.join(path, "generated_graphs", graph_name),
            bbox_inches="tight", pad_inches=0.5, dpi=150,
            transparent=False, format="png"
        )
        print(f"\n[GRAPH] Saved: generated_graphs/{graph_name}")
    except FileNotFoundError:
        print("[WARN] Could not save portfolio MC graph")
    finally:
        plt.clf()
        plt.close("all")
