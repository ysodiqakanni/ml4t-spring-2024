
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

def get_stock_data(symbols, start_date, end_date, data="Adj Close"):
    df_prices_all = get_data(symbols, pd.date_range(start_date, end_date), colname=data)
    # now exclude SPY
    df_prices = df_prices_all[symbols]
    return df_prices

def get_smaOld(symbols, start_date, end_date, lookback=14):
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

            # basically sum data from [day-lb+1 to day+1]
            lookback_mean = df_prices.ix[day-lookback+1 :day+1, sym].mean()
            sma.ix[day, sym] = lookback_mean
            # for prev_day in range(day-lookback+1, day+1):
            #     # sum the values
            #     sma.ix[day, sym] += df_prices.ix[prev_day, sym]
            # # now divide by the lookback to get the simple moving average for this day
            # compare_with = sma.ix[day,sym] / lookback
            # sma.ix[day, sym] /= lookback
    return df_prices, sma

def bollinger_bands_indicatorOld(symbols, start_date, end_date, lookback=14):
    df_prices, sma = get_smaOld(symbols, start_date, end_date, lookback)
    #
    # if df_prices is None:
    #     df_prices_all = get_data(symbols, pd.date_range(start_date, end_date))
    #     df_prices = df_prices_all[symbols]

    bbp = df_prices.copy()
    bbp[:] = 0
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

    #lines = [(df_prices, "Adj Close"), (sma, "SMA")]
    #line = [(bbp, "BBP")]
    #title = "SMA and Adj Close Price for JPM"
    #generate_chart(lines, title, "Date", "Value", "BolingerBands")
    #generate_chart(line, "Bollinger Band Percentage for JPM", "Date", "Value", "Bbp")
    return bbp

# Fully vectorized sma
def get_sma(symbols, start_date, end_date, lookback=14):
    prices = get_stock_data(symbols, start_date, end_date)
    sma = prices.cumsum()
    sma.values[lookback:,:] = (sma.values[lookback:,:] - sma.values[:-lookback, :]) / lookback
    sma.ix[:lookback,:] = np.nan
    return sma


# Vectorized version below
def bollinger_bands_indicator(symbols, start_date, end_date, lookback=14):
    symbol = symbols[0]
    sma = get_sma(symbols, start_date, end_date, lookback)
    prices = get_stock_data(symbols, start_date, end_date)
    rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()
    upper_std = sma + 2*rolling_std
    lower_std = sma - 2*rolling_std

    bbp = (prices-lower_std) / (upper_std - lower_std)
    return bbp
def rsi_indicator_old(symbols, start_date, end_date, lookback):
    df_prices = get_stock_data(symbols, start_date, end_date, "Adj Close")
    df_rsi = df_prices.copy()
    df_rsi.ix[:,:] = 0

    for curr_day in range(df_prices.shape[0]):
        if curr_day < lookback - 1:
            continue
        for sym in symbols:
            up_gain = 0
            down_loss = 0

            # iterate over lookback from current day and calculate gain on up days, and loss on down days
            for prev_day in range(curr_day-lookback+1, curr_day+1):
                price_change = df_prices.ix[prev_day, sym] - df_prices.ix[prev_day-1, sym]

                if price_change > 0:
                    up_gain += price_change
                else:
                    down_loss -= price_change

            # Recall, RSI = 100 - 100/(1+RS)
            # RS = Avg gain/Avg loss
            # if the total loss is 0, then RS is infinite, and RSI = 100
            if down_loss == 0:
                df_rsi.ix[curr_day, sym] = 100
            else:
                rs = (up_gain/lookback) / (down_loss/lookback)  # ratio of average gain and average loss
                df_rsi.ix[curr_day, sym] = 100 - (100 / (1+rs))

    # now let's generate a plot of rsi and adj close prices
    #generate_chart([(df_prices, "Adj Close"), (df_rsi, "RSI")], "The chart of Adjusted price and RSI for JPM", "Date", "Value", "RsiIndicator")
    return df_rsi

# 4/16/2024 11:56am
def rsi_indicator(symbols, start_date, end_date, lookback):
    df_prices = get_stock_data(symbols, start_date, end_date, "Adj Close")
    df_rsi = pd.DataFrame(index=df_prices.index, columns=df_prices.columns)

    delta = df_prices.diff()    # daily price difference
    up_gains = (delta.where(delta > 0, 0)).rolling(window=lookback).mean()
    down_losses = (-delta.where(delta < 0, 0)).rolling(window=lookback).mean()

    rs = up_gains / down_losses
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN

    rsi = 100 - (100 / (1 + rs))

    # Handle the case where loss is zero
    rsi[down_losses == 0] = 100

    df_rsi.iloc[lookback-1:] = rsi.iloc[lookback-1:]

    return df_rsi
def rsi_indicator_buggy(symbols, start_date, end_date, lookback):
    df_prices = get_stock_data(symbols, start_date, end_date, "Adj Close")
    df_rsi = df_prices.copy()
    df_rsi.ix[:,:] = 0

    price_differences = df_prices.diff(1)
    gains = price_differences.where(price_differences > 0, 0)
    losses = price_differences.where(price_differences < 0, 0)

    up_gains = gains.rolling(window=lookback, min_periods=1).mean()
    down_losses = losses.rolling(window=lookback, min_periods=1).mean()

    rs = up_gains / down_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi

def stochastic_indicator(symbols, start_date, end_date, lookback=14):
    # the formular:
    # %K = (currentClose - LowestLow) / (HighestHigh - LowestLow) * 100
    # %D = 3-day SMA of %K
    # so we need adj close, low and high
    df_prices = get_stock_data(symbols, start_date, end_date, "Adj Close")
    df_high_prices = get_stock_data(symbols, start_date, end_date, "High")
    df_low_prices = get_stock_data(symbols, start_date, end_date, "Low")

    df_stochastic = df_prices.copy()
    df_stochastic.ix[:, :] = 0
    # Todo: set a 0 default value for all %k columns to avoid nan
    # we only need the %k values as they're more sensitive to price changes

    for curr_day in range(df_prices.shape[0]):
        if curr_day < lookback-1:
            continue
        # calculate for each symbol
        for sym in symbols:
            # get the highest of the past 14 data. that is data[day-14,day]
            highs_14 = df_high_prices.ix[curr_day-lookback+1:curr_day, sym]
            low_14 = df_low_prices.ix[curr_day-lookback+1:curr_day, sym]
            # st the percentage k value for this symbol on this day
            kcol = f"{sym}_PK"
            df_stochastic.ix[curr_day, sym] = (df_prices.ix[curr_day, sym] - min(low_14)) / (max(highs_14) - min(low_14)) * 100
            # calculate the %D which is just a 3-day SMA of %k.
            # we need to have seen loopback + 3 columns
            #if curr_day >= lookback-1 + 3-1:
            #    df_stochastic.ix[curr_day, sym] = sum(df_stochastic.ix[curr_day-2:curr_day+1, kcol])/3

    # let's generate the plots
    #lines = [(df_stochastic[["JPM"]], "14-day Stochastic"), (df_prices[["JPM"]], "Price")]
    #title = "Stochastic oscillator vs price for JPM"
    #generate_chart(lines, title, "Date", "Value", "StochasticIndicator")
    return df_stochastic

def rate_of_change_indicator(symbols, start_date, end_date, lookback=12):
    # Rate of change is a momentum indicator
    # formular: ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
    # lookback will be n in our formular
    df_prices = get_stock_data(symbols, start_date, end_date)
    df_roc = df_prices.copy()
    df_roc.ix[:,:] = 0

    for curr_day in range(df_prices.shape[0]):
        prev_day = curr_day-lookback-1
        if prev_day < 0:
            continue
        for sym in symbols:
            df_roc.ix[curr_day, sym] = ((df_prices.ix[curr_day, sym] - df_prices.ix[prev_day, sym]) / df_prices.ix[prev_day, sym]) * 100

    lines = [(df_roc[["JPM"]], "12-day Rate of Change"), (df_prices[["JPM"]], "Adj Close")]
    title = "Rate of Change Indicator for JPM"
    generate_chart(lines, title, "Date", "Value", "RateOfChangeIndicator")
    return df_roc

def golden_cross_indicator(symbols, start_date, end_date):
    df_prices, df_sma_short_term = get_sma(symbols, start_date, end_date, 14)
    df_prices, df_sma_long_term = get_sma(symbols, start_date, end_date, 50)
    # for project 8, we identify the golden cross and death crosses and return signals -1,0,1
    # for now we just return the 2 vectors

    lines = [(df_prices, "Adj Close"), (df_sma_short_term, "Short Term SMA"), (df_sma_long_term, "Long Term SMA")]
    title = "Long term and short term SMA plotted against Adj Close price"
    generate_chart(lines, title, "Date", "Value", "GoldenCross")
    return [df_sma_short_term, df_sma_long_term]

# lines represent a list of tuple of lines to plot on the same chart
# each tuple looks like (dataframe, label)
def generate_chart(lines, title, xlabel, ylabel, file_name):
    plt.figure()
    # plot the first line
    ax = lines[0][0].plot()
    for l in lines[1:]:
        l[0].plot(ax=ax)

    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([line[1] for line in lines])
    plt.savefig("{}.png".format(str(file_name)))

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    #daily_returns[1:] = (df[1:] / df[:-1].values) - 1 # compute daily returns for row 1 onwards
    daily_returns = (df / df.shift(1)) - 1  # much easier with Pandas!
    daily_returns[0] = 0
    #daily_returns.iloc[0, :] = 0  # Pandas leaves the 0th row full of Nans
    return daily_returns

def getStatistics(port_vals):
    """
    computes and returns cummulative returns, std and mean of daily returns
    """
    dailyReturns = compute_daily_returns(port_vals)
    cr = (port_vals[-1]/port_vals[0]) -1
    adr = dailyReturns.mean()
    sddr = dailyReturns.std()

    return [round(cr,6), round(sddr,6), round(adr,6), round(port_vals[-1],6)]

def run(symbols, start_date, end_date, lookback=14):
    #bbp = bollinger_bands_indicator(symbols, start_date, end_date, lookback=14)
    rsi = rsi_indicator_old(symbols, start_date, end_date, lookback=14)
    rsi2 = rsi_indicator_today(symbols, start_date, end_date, lookback=14)
    print("RSI shapes: ", rsi.shape, rsi2.shape)
    #stochastic = stochastic_indicator(symbols, start_date, end_date, lookback=14)
    #roc = rate_of_change_indicator(symbols, start_date, end_date, lookback=12)
    #gcross = golden_cross_indicator(symbols, start_date, end_date)


if __name__ == '__main__':
    print("running...")
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbols = ['JPM']
    run(symbols, sd, ed)
    print("done")