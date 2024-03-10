
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
import math

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

def plot_sma(prices, sma, sma_long_term):
    # get the prices
    plt.figure()
    ax = prices.plot(color='r')
    sma.plot(color='g', ax=ax)
    sma_long_term.plot(color='b', ax=ax)

    plt.grid(True)
    plt.title("SMA vs Adjusted closing price for JPM")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(["Closed prices", "SMA", "SMA long term"])
    plt.show()

def generate_bollinger_bands(symbols, sma, start_date, end_date, df_prices=None, lookback=14):
    if df_prices is None:
        df_prices_all = get_data(symbols, pd.date_range(start_date, end_date))
        df_prices = df_prices_all[symbols]

    bbp = df_prices.copy()
    bbp[:,:] = 0
    for day in range(df_prices.shape[0]):
        for sym in symbols:
            # calculate the standard deviation over the lookback period. std = sum(x - mean)^2
            for prev_day in range(day-lookback+1, day+1):
                bbp.ix[day, sym] += math.pow(df_prices.ix[prev_day, sym] - sma.ix[day, sym], 2)

            # complete the standard deviation calculation
            bbp.ix[day,sym] = math.sqrt(bbp.ix[day,sym]/(lookback-1))

            # complete the calculation of BB% for this day and symbol
            bottom_band = sma.ix[day,sym] - (2 * bbp.ix[day,sym])
            top_band = sma.ix[day, sym] + (2 * bbp.ix[day, sym])

            bbp.ix[day,sym] = (df_prices.ix[day, sym] - bottom_band) / (top_band - bottom_band)

def run(symbols, start_date, end_date, lookback=14):
    df_prices, df_sma = get_sma(symbols, start_date, end_date, 14)
    df_prices, df_sma_long_term = get_sma(symbols, start_date, end_date, 50)

    # now plot sma over price
    plot_sma(df_prices, df_sma,df_sma_long_term)


if __name__ == '__main__':
    print("running...")
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbols = ['JPM']
    run(symbols, sd, ed)
    print("done")