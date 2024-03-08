import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf


def efficient_frontier_sim(price_df):
    log_returns_df = np.log(1 + price_df.pct_change(1).dropna())
    log_returns_mean = log_returns_df.mean() * 252
    log_returns_mean = log_returns_mean.to_frame()
    log_returns_mean = log_returns_mean.rename(columns={0: "Mean"})
    log_returns_mean = log_returns_mean.transpose()
    log_returns_df = log_returns_df[log_returns_mean.columns]
    log_returns_cov = log_returns_df.cov() * 252
    portefolio_number = []
    portefolio_weight = []
    portfolio_returns = []
    portfolio_volatilities = []
    print("Starting simulation...")
    for sim in range(750000):
        portefolio_number.append(sim)
        weights = np.random.random(len(log_returns_mean.columns))
        weights /= np.sum(weights)
        portefolio_weight.append(weights)
        portfolio_returns.append(np.sum(weights * np.array(log_returns_mean)))
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns_cov, weights))))
        if sim % 25000 == 0:
            print(f"Simulations: {sim}")
                    

    portefolio_number_df = pd.DataFrame(portefolio_number, columns=["Portefolio number"])
    portefolio_weight_df = pd.DataFrame(portefolio_weight, columns=log_returns_mean.columns)
    portfolio_returns_df = pd.DataFrame({'Return': portfolio_returns, 'Volatility':portfolio_volatilities})
    portefolio_df = pd.concat([portefolio_number_df, portefolio_weight_df, portfolio_returns_df], axis=1)
    # print("Portefolio df: ")
    # print(portefolio_df)
    # portefolio_df = portefolio_df.loc[portefolio_df['Return'] > 0]
    # print("Portefolio df: ")
    # print(portefolio_df)
    portefolio_df = portefolio_df.sort_values(by="Volatility", ascending=True)
    portefolio_df.plot(x='Volatility', y='Return', kind='scatter', figsize=(18,10))
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
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

    if len(price_df.columns) > 2:
        # Reduced deciamals in Volatility to 3
        portefolio_df["Volatility"] = portefolio_df["Volatility"].round(3)
        portefolio_df = portefolio_df.groupby("Volatility").max("Return")
        columns = list(portefolio_df.columns)
        columns.append("Volatility")
        portefolio_df = portefolio_df.reset_index()
        portefolio_df = portefolio_df[columns]
        # Remove Return Volatility pairs that are lower than the the previous pair
        loop = True
        while loop == True:
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
            

        portefolio_df.plot(x="Volatility", y="Return", kind="scatter", figsize=(18,10))
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        graph_name = str(f"Efficient_frontier_reduced.png")
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        # Save the graph
        try:
            plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
            plt.clf()
            plt.close("all")


        except FileNotFoundError:
                raise FileNotFoundError("The graph could not be saved. Please check the file name or path.")
        

    return portefolio_df
    

if __name__ == "__main__":
    # Import stock symbols from a CSV file
    # Load financial data into a Pandas dataframe
    prices = yf.download(["AMBU-B.CO", "ROCK-B.CO", "TRYG.CO", "DEMANT.CO", "GN.CO", "JYSK.CO", "RBREW.CO"])["Open"]
    portefolio_df = efficient_frontier_sim(prices)
    print("Done.")
