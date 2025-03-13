import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf

def efficient_frontier_sim(price_df):
    """
    This function calculates the efficient frontier of a portfolio based on the input price dataframe.
    The function calculates the log returns of the price dataframe, the mean of log returns, and the covariance matrix of log returns.
    It then runs 750,000 simulations to generate random weights for the portfolio and calculate the portfolio return and volatility.
    The function plots the efficient frontier and saves the graph as a PNG file.
    If there are more than 2 columns in the price dataframe, the function performs additional processing to reduce the efficient frontier.
    The function returns a dataframe containing the portfolio number, weights, returns, and volatilities.

    :param price_df: A pandas dataframe containing the stock prices.
    :return: A pandas dataframe containing the portfolio number, weights, returns, and volatilities.
    """
    # Print the input price dataframe
    print("price_df")
    print(price_df)
    
    # Calculate log returns of the price dataframe
    log_returns_df = np.log(1 + price_df.pct_change(1).dropna())
    
    # Calculate mean of log returns and annualize it
    log_returns_mean = log_returns_df.mean() * 252
    log_returns_mean = log_returns_mean.to_frame()
    log_returns_mean = log_returns_mean.rename(columns={0: "Mean"})
    log_returns_mean = log_returns_mean.transpose()
    
    # Ensure log returns dataframe only contains columns present in log_returns_mean
    log_returns_df = log_returns_df[log_returns_mean.columns]
    
    # Calculate covariance matrix of log returns and annualize it
    log_returns_cov = log_returns_df.cov() * 252
    
    # Initialize lists to store simulation results
    portefolio_number = []
    portefolio_weight = []
    portfolio_returns = []
    portfolio_volatilities = []
    
    print("Starting simulation...")
    # Run 750,000 simulations
    for sim in range(750000):
        portefolio_number.append(sim)
        
        # Generate random weights for the portfolio
        weights = np.random.random(len(log_returns_mean.columns))
        weights /= np.sum(weights)
        
        # Store the weights and calculate portfolio return and volatility
        portefolio_weight.append(weights)
        portfolio_returns.append(np.sum(weights * np.array(log_returns_mean)))
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns_cov, weights))))
        
        # Print progress every 25,000 simulations
        if sim % 25000 == 0:
            print(f"Simulations: {sim}")
                    
    # Create dataframes for portfolio numbers, weights, returns, and volatilities
    portefolio_number_df = pd.DataFrame(portefolio_number, columns=["Portefolio number"])
    portefolio_weight_df = pd.DataFrame(portefolio_weight, columns=log_returns_mean.columns)
    portfolio_returns_df = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatilities})
    
    # Concatenate all dataframes into a single dataframe
    portefolio_df = pd.concat([portefolio_number_df, portefolio_weight_df, portfolio_returns_df], axis=1)
    
    # Sort the dataframe by volatility in ascending order
    portefolio_df = portefolio_df.sort_values(by="Volatility", ascending=True)
    
    # Plot the efficient frontier
    portefolio_df.plot(x='Volatility', y='Return', kind='scatter', figsize=(18,10))
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    
    # Define the path and name for the graph
    graph_name = str(f"Efficient_frontier.png")
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    
    # Save the graph
    try:
        plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
        plt.clf()
        plt.close("all")
    except FileNotFoundError:
        raise FileNotFoundError("The graph could not be saved. Please check the file name or path.")
    
    # If there are more than 2 columns in the price dataframe, perform additional processing
    if len(price_df.columns) > 2:
        # Round volatility to 3 decimal places
        portefolio_df["Volatility"] = portefolio_df["Volatility"].round(3)
        
        # Group by volatility and keep the maximum return for each volatility
        portefolio_df = portefolio_df.groupby("Volatility").max("Return")
        
        # Reset index and reorder columns
        columns = list(portefolio_df.columns)
        columns.append("Volatility")
        portefolio_df = portefolio_df.reset_index()
        portefolio_df = portefolio_df[columns]
        
        # Remove return-volatility pairs that are lower than the previous pair
        loop = True
        while loop:
            x = 0
            drop_list = []
            for index, row in portefolio_df.iterrows():
                if index > 0:
                    if row["Return"] < portefolio_df.iloc[index-1]["Return"]:
                        drop_list.append(index)
                        x += 1
            
            portefolio_df = portefolio_df.drop(drop_list)
            portefolio_df = portefolio_df.reset_index(drop=True)
            if x == 0:
                loop = False
        
        # Plot the reduced efficient frontier
        portefolio_df.plot(x="Volatility", y="Return", kind="scatter", figsize=(18,10))
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        
        # Define the path and name for the reduced graph
        graph_name = str(f"Efficient_frontier_reduced.png")
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        
        # Save the reduced graph
        try:
            plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
            plt.clf()
            plt.close("all")
        except FileNotFoundError:
            raise FileNotFoundError("The graph could not be saved. Please check the file name or path.")
    
    return portefolio_df

if __name__ == "__main__":
    # Download stock prices for specified symbols
    prices = yf.download(["AMBU-B.CO", "ROCK-B.CO", "TRYG.CO", "DEMANT.CO", "GN.CO", "JYSK.CO", "RBREW.CO"])["Open"]
    
    # Run the efficient frontier simulation
    portefolio_df = efficient_frontier_sim(prices)
    
    # Print completion message
    print("Done.")
