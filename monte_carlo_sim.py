import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
from dateutil.relativedelta import relativedelta

def monte_carlo_analysis_1(seed_number, stock_data_df, forecast_df, days, sim_amount):
  np.random.seed(seed=seed_number)
  log_returns = np.log(1 + stock_data_df[stock_data_df.columns[1]].pct_change())
  sns.displot(log_returns.iloc[1:])
  plt.xlabel("Daily Return")
  plt.ylabel("Frequency")
  plt.title("Log Returns Distribution")
  plt.show()
  mean = log_returns.mean()
  print(mean)
  var = log_returns.var()
  print(var)
  drift = mean - (0.5 * var)
  print(drift)
  std_dev = log_returns.std()
  print(std_dev)
  z = np.random.normal(loc=0, scale=1, size=(days, sim_amount))
  daily_returns = np.exp(drift + std_dev * z)
  S0 = stock_data_df[stock_data_df.columns[1]].iloc[-1]
  price_list = np.zeros_like(daily_returns)
  price_list[0] = S0
  for t in range(1, days):
    price_list[t] = price_list[t - 1] * daily_returns[t]


  plt.figure(figsize=(18, 8))
  plt.plot(price_list)
  plt.xlabel("Days")
  plt.ylabel("Price")
  plt.title("Monte Carlo Analysis for Stock Price")
  plt.show()


def monte_carlo_analysis_2(seed_number, stock_data_df, forecast_df, days, sim_amount):
  np.random.seed(seed=seed_number)
  dt = 1 / days
  sigma = stock_data_df[stock_data_df.columns[1]].std()
  mu = stock_data_df[stock_data_df.columns[1]].mean()
  price = np.zeros(days)
  price[0] = stock_data_df[stock_data_df.columns["Price"]].iloc[-1]
  print(price[0])
  for run in range(sim_amount):
    shock = np.zeros(days)
    for day in range(1, days):
      shock = np.random.normal(loc = mu * dt, scale = sigma * np.sqrt(dt))
      price[day] = (mu * price[day - 1] * dt) + (sigma * price[day - 1] * shock)


def monte_carlo_analysis_3(seed_number, stock_data_df, forecast_df, years, sim_amount):
  np.random.seed(seed=seed_number)
  monte_carlo_df = pd.DataFrame()
  # calculate amopunt of days into x future of years
  days = years * 252
  exp_returns = forecast_df[forecast_df.columns[1]].pct_change().dropna()
  sigma = exp_returns.std()
  print(sigma)
  mu = exp_returns.mean()
  print(mu)
  dt = 1 / days
  print(dt)
  for run in range(sim_amount):
    price = np.zeros(days)
    price[0] = forecast_df.loc[0, forecast_df.columns[1]]
    # print(price)
    # print(price[0])
    shock = np.zeros(days)
    drift = np.zeros(days)
    for day in range(1, days):
      shock[day] = np.random.normal(loc = mu * dt, scale = sigma * np.sqrt(dt))
      drift[day] = mu * dt
      price[day] = price[day - 1] + (price[day - 1] * (drift[day] + shock[day]))
      # if day < 10:
      #   print(day)
      #   print(shock[day])
      #   print(drift[day])
      #   print(price[day - 1])
      #   print((drift[day] + shock[day]))
      #   print(price[day])


    price_df = pd.DataFrame(price, columns=[f"Run_{run}"])
    monte_carlo_df = pd.concat([monte_carlo_df, price_df], axis=1)

    # Create print statement per 250 index processed
    if run % 250 == 0:
        print(f"Processed {run} rows, out of {sim_amount} runs.")


  legend_list = []
  for column in monte_carlo_df.columns:
      legend_list.append(column)

  plt.figure(figsize=(18, 8))
  plt.plot(monte_carlo_df)
  # plt.legend(legend_list, loc="best")
  plt.xlabel("Days")
  plt.ylabel("Price")
  plt.title("Monte Carlo Analysis for Stock Price")
  # plt.show()
  stock_data_df = stock_data_df.replace({"Name": [" ", "/"]}, {"Name": "_"}, regex=True)
  stock_name = stock_data_df.iloc[0]["Name"]
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
      

  return monte_carlo_df


# days = 365
# dt = 1 / days 
# sigma = stock_data_df["Price"].std()
# mu = stock_data_df["Price"].mean()
# np.random.seed(0)
# def monte_carlo_analysis(starting_price, days, mu, sigma):
#     # print(f"Starting price: {starting_price}")
#     # print(f"Days: {days}")
#     # print(f"Mu: {mu}")
#     # print(f"Sigma: {sigma}")
#     price = numpy.zeros(days)
#     price[0] = starting_price
#     shock = numpy.zeros(days)
#     drift = numpy.zeros(days)
#     for day in range(1, days):

#         shock[day] = numpy.random.normal(loc = mu * dt, scale = sigma * numpy.sqrt(dt))
#         drift[day] = mu * dt
#         price[day] = price[day - 1] + (price[day - 1] * (drift[day] + shock[day]))


#     return price

# for run in range(1000):
#     price = monte_carlo_analysis(starting_price, days, mu, sigma)
#     # print(price)
#     pyplot.plot(price)
#     pyplot.xlabel("Days")
#     pyplot.ylabel("Price")

# pyplot.show()