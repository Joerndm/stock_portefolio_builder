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
