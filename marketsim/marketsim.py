""""""  		  	   		 	   			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Sodiq Yusuff		  	   		 	   			  		 			     			  	 
GT User ID: syusuff3		  	   		 	   			  		 			     			  	 
GT ID: 903953477	  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt  		  	   		 	   			  		 			     			  	 
import os  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import pandas as pd  		  	   		 	   			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 


def author():
    return 'syusuff3'
def compute_portvals(
    orders_file="./orders/orders.csv",  		  	   		 	   			  		 			     			  	 
    start_val=1000000,  		  	   		 	   			  		 			     			  	 
    commission=9.95,  		  	   		 	   			  		 			     			  	 
    impact=0.005,  		  	   		 	   			  		 			     			  	 
):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    Computes the portfolio values.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 	   			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 	   			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
    :type start_val: int  		  	   		 	   			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	   			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
    """
    # this is the function the autograder will call to test your code  		  	   		 	   			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	   			  		 			     			  	 
    # code should work correctly with either input  		  	   		 	   			  		 			     			  	 
    # TODO: Your code here

    # My pseudocode
    # read in the order file to get all the symbols
    # also get the start and end dates from the file

    df_orders = pd.read_csv(
        orders_file,
        index_col="Date",
        parse_dates=True,
        na_values=["nan"],
    )

    df_orders.sort_values(by="Date", inplace=True)
    symbols = df_orders["Symbol"].unique() # ["IBM","GOOG"]
    # load in the data in a dataframe

    start_date = df_orders.index[0] # dt.datetime(2008,1,1)
    end_date = df_orders.index[-1] # dt.datetime(2008,6,1)

    # the below line loads the adjusted price for all selected stocks including SPY
    df_prices_all = get_data(symbols, pd.date_range(start_date, end_date))
    # now exclude SPY
    df_prices = df_prices_all[symbols]
    # now we add an extra column (CASH) with values 1
    df_prices["CASH"] = 1
    # then create a dataframe to hold the trade data. It should look like the price df (in terms of shape)
    df_trades = df_prices.copy()
    df_trades.iloc[:, :] = 0
    # create a df to record holdings

    # now we iterate over the daily orders and update the dataframes
    # Note: charge commission (on each transaction) as a deduction from cash balance
    # Also charge a percentage of impact.
    for idx, order_item in df_orders.iterrows():
        sym, order, shares = order_item["Symbol"], order_item["Order"], order_item["Shares"]
        market_impact = impact * df_prices.loc[idx, sym] * shares
        if order == "BUY":
            df_trades.loc[idx, sym] += shares  # add the volume
            df_trades.loc[idx, "CASH"] -= (shares * df_prices.loc[idx, sym] + commission + market_impact)
        else:
            df_trades.loc[idx, sym] -= shares
            df_trades.loc[idx, "CASH"] += (shares * df_prices.loc[idx, sym] - commission - market_impact)

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

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    #
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.value
    return portvals
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
def test_code():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    Helper function to test code  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 	   			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 	   			  		 			     			  	 
    # Define input parameters  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    of = "./orders/orders-short.csv"
    sv = 1000000  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # Process orders  		  	   		 	   			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	   			  		 			     			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		 	   			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	   			  		 			     			  	 
    else:  		  	   		 	   			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # Get portfolio stats  		  	   		 	   			  		 			     			  	 
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		 	   			  		 			     			  	 
    start_date = dt.datetime(2008, 1, 1)  		  	   		 	   			  		 			     			  	 
    end_date = dt.datetime(2008, 6, 1)  		  	   		 	   			  		 			     			  	 
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		 	   			  		 			     			  	 
        0.2,  		  	   		 	   			  		 			     			  	 
        0.01,  		  	   		 	   			  		 			     			  	 
        0.02,  		  	   		 	   			  		 			     			  	 
        1.5,  		  	   		 	   			  		 			     			  	 
    ]  		  	   		 	   			  		 			     			  	 
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		 	   			  		 			     			  	 
        0.2,  		  	   		 	   			  		 			     			  	 
        0.01,  		  	   		 	   			  		 			     			  	 
        0.02,  		  	   		 	   			  		 			     			  	 
        1.5,  		  	   		 	   			  		 			     			  	 
    ]  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		 	   			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	   			  		 			     			  	 
    print()  		  	   		 	   			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	   			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	   			  		 			     			  	 
    print()  		  	   		 	   			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	   			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	   			  		 			     			  	 
    print()  		  	   		 	   			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	   			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	   			  		 			     			  	 
    print()  		  	   		 	   			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	   			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	   			  		 			     			  	 
    print()  		  	   		 	   			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    test_code()  		  	   		 	   			  		 			     			  	 
