"""
Code implementing a TheoreticallyOptimalStrategy (details below).
It should implement testPolicy(), which returns a trades data frame (see below).

"""

"""
Instructions:
Allowable positions are 1000 shares long, 1000 shares short, 0 shares. (You may trade up to 2000 shares at a time as long as your positions are 1000 shares long or 1000 shares short.)  
Benchmark: The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM, and holding that position.  
Transaction costs for TheoreticallyOptimalStrategy:  
Commission: $0.00
Impact: 0.00. 

"""

import datetime as dt
import pandas as pd
from util import get_data
def author():
    return 'syusuff3'

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    # returns a trades dataframe
    # get the stock prices for the date range
    # loop through the prices and determine whether to buy,sell or do nothing

    # the below line loads the adjusted price for all selected stocks including SPY
    df_prices_all = get_data([symbol], pd.date_range(sd, ed))
    # now exclude SPY
    df_prices = df_prices_all[symbol]

    # create trades dataframe from df_prices so they can have the same dimension and index
    df_trades = df_prices.copy()
    # set all the values to 0
    df_trades[:] = 0

    dates_index = df_prices.index
    net_holding = 0
    for i in range(len(dates_index) - 1):
        # if next price is > current, we buy today.
        # if it's less, then we sell today
        # if it's same then we take previous action taken
        # for all 3 above, net holdings must be -1000 or 1000 or 0

        # holdings = previous holding + today's action
        # eg, -1000 (yesterday) + 1000 today.
        # 0 yesterday + 1000 today
        position = 0
        if df_prices[dates_index[i+1]] > df_prices[dates_index[i]]:
            # tomorrow's price is higher, so we buy!. You can only buy if holdings is 0 or -1000
            position = 1000
        elif df_prices[dates_index[i+1]] < df_prices[dates_index[i]]:
            # sell!
            position = -1000
        if position + net_holding < -1000 or position + net_holding > 1000:
            # position goes out of the allowed constraints, so do nothing!
            position = 0

        df_trades[dates_index[i]] = position
        net_holding += position

    return df_trades