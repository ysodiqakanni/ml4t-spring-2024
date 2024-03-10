
"""
Code implementing your indicators as functions that operate on DataFrames.
There is no defined API for indicators.py, but when it runs,
the main method should generate the charts that will illustrate your indicators in the report.
"""
import numpy as np

from util import get_data
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def author():
    return 'syusuff3'

def get_sma(symbols, start_date, end_date, lookback=14):
    """
    Implement Simple Moving Average Indicator. If you have close prices 30,40,50,60,70,80,90,100,
    the sma over 4 days is calculated as: SMA = 30+40+50+60/4, 40+50+60+70/4, 50+60+70+80/4,...
    """
    # load prices for the symbols
    df_prices_all = get_data(symbols, pd.date_range(start_date, end_date))
    # now exclude SPY
    df_prices = df_prices_all[symbols]

    sma = df_prices.copy()
    # set the values of the df to zero and we can accumulate new values into it
    for day in range(df_prices.shape[0]):
        for sym in symbols:
            sma.ix[day, sym] = 0

    # loop over all days
    for day in range(df_prices.shape[0]):
        if day < lookback-1:
            # this day is too early to calculate the full SMA so we set the values to nan across symbols
            for sym in symbols:
                sma.ix[day, sym] = np.nan
            continue

        for sym in symbols:
            # if lookback =14 for example, we want to go back 14 days and loop from then to now
            # to get to current day, right boundary = day+1
            # left boundary = day-14 +1, (+1 to avoid starting from day 0). We want [1,14]
            for prev_day in range(day-lookback+1, day+1):
                # sum the values
                sma.ix[day, sym] += df_prices.ix[prev_day, sym]
            # now divide by the lookback to get the simple moving average for this day
            sma.ix[day, sym] /= lookback
    return df_prices, sma

def plot_sma(prices, sma):
    # get the prices
    plt.figure()
    ax = prices.plot(color='r')
    sma.plot(color='g', ax=ax)

    plt.grid(True)
    plt.title("SMA vs Adjusted closing price for JPM")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(["Closed prices", "SMA"])
    plt.show()
def run(symbols, start_date, end_date, lookback=14):
    df_prices, df_sma = get_sma(symbols, start_date, end_date, lookback)

    # now plot sma over price
    plot_sma(df_prices, df_sma)

if __name__ == '__main__':
    print("running...")
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbols = ['JPM']
    run(symbols, sd, ed)
    print("done")