"""
An improved version of your marketsim code accepts a “trades” DataFrame (instead of a file).
More info on the trades data frame is below. It is OK not to submit this file if you
have subsumed its functionality into one of your other required code files.
This file has a different name and a slightly different setup than your previous project.
However, that solution can be used with several edits for the new requirements.
"""

from util import get_data
import pandas as pd

def author():
    return 'syusuff3'

# def simulate_market(trades_df):
#     # performs market simulation
#     pass


def compute_portvals(
        df_trades,
        symbol,
        start_val=100000,
        commission=0.0,
        impact=0.0,
):
    # My pseudocode
    # read in the order file to get all the symbols
    # also get the start and end dates from the file


    start_date = df_trades.index[0]
    end_date = df_trades.index[-1]
    symbols = [symbol]

    # the below line loads the adjusted price for all selected stocks including SPY
    df_prices_all = get_data(symbols, pd.date_range(start_date, end_date))
    # now exclude SPY
    df_prices = df_prices_all[symbols]
    # now we add an extra column (CASH) with values 1
    df_prices["CASH"] = 1
    # then create a dataframe to hold the trade data. It should look like the price df (in terms of shape)
    df_trades["CASH"] = 0

    # now we iterate over the daily orders and update the dataframes
    # Note: charge commission (on each transaction) as a deduction from cash balance
    # Also charge a percentage of impact.
    for idx, order_item in df_trades.iterrows():
        #sym, order, shares = order_item["Symbol"], order_item["Order"], order_item["Shares"]
        position, shares = order_item[symbol], abs(order_item[symbol])
        market_impact = impact * shares
        if position > 0:
            # buying
            #df_trades.loc[idx, symbol] += shares  # add the volume
            df_trades.loc[idx, "CASH"] -= (shares * df_prices.loc[idx, symbol] + commission + market_impact)
        else:
            #df_trades.loc[idx, symbol] -= shares
            df_trades.loc[idx, "CASH"] += (shares * df_prices.loc[idx, symbol] - commission - market_impact)

    # let's calculate the holdings => for each row i, add df_trades[i] + holdings[i-1]
    # so that means we can initialize the holdings df to df_tades, and them do a cummulative sum of holdings
    # adding previous data to current
    df_holdings = df_trades.copy()
    # add the capital to the first cash value
    first_date_idx = df_holdings.index[0]
    df_holdings.loc[first_date_idx, "CASH"] += start_val
    df_holdings = df_holdings.cumsum()

    # create the values dataframe
    df_values = df_prices * df_holdings

    portvals = df_values.sum(axis=1)

    return portvals
