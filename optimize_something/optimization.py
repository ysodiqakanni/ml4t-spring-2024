""""""
import math

"""MC1-P2: Optimize a portfolio.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
  		  	   		 	   			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import matplotlib.pyplot as plt  		  	   		 	   			  		 			     			  	 
import pandas as pd  		  	   		 	   			  		 			     			  	 
from util import get_data, plot_data
import scipy.optimize as spo
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
# This is the function that will be tested by the autograder  		  	   		 	   			  		 			     			  	 
# The student must update this code to properly implement the functionality  		  	   		 	   			  		 			     			  	 
def optimize_portfolio(  		  	   		 	   			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1,0,0),
    ed=dt.datetime(2009, 1, 1,0,0),
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	   			  		 			     			  	 
    gen_plot=False,  		  	   		 	   			  		 			     			  	 
):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	   			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	   			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	   			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	   			  		 			     			  	 
    statistics.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
    :type sd: datetime  		  	   		 	   			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
    :type ed: datetime  		  	   		 	   			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	   			  		 			     			  	 
        symbol in the data directory)  		  	   		 	   			  		 			     			  	 
    :type syms: list  		  	   		 	   			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	   			  		 			     			  	 
        code with gen_plot = False.  		  	   		 	   			  		 			     			  	 
    :type gen_plot: bool  		  	   		 	   			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	   			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	   			  		 			     			  	 
    :rtype: tuple  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	   			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		 	   			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	   			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # find the allocations for the optimal portfolio
    # assume we start with equal allocations. That's 1/4 each

    # we normalize the prices by dividing by the first price  ==> normed
    normalizedPrices = prices / prices.iloc[0]
    # provide an allocation guess
    guessedAllocations = [1.0 / normalizedPrices.shape[1] for i in range(normalizedPrices.shape[1])]

    # equality constraint defines a function that must be 0. => sum(allocs) - 1 = 0 => sum(allocs) = 1
    # inequality constraint defines fn that must be >= 0. And we want x >= 0.
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, 1)] * len(guessedAllocations)
    # also, there's a need to set the bounds to ensure the upper and lower limit are explicitly set
    ret = spo.minimize(sharpe_ratio_fn, guessedAllocations, args=(normalizedPrices,), method='SLSQP',
                              options={'disp':False}, constraints=constraints, bounds=bounds)
    statistics = getStatistics(ret.x, normalizedPrices)

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	   			  		 			     			  	 
    if gen_plot:  		  	   		 	   			  		 			     			  	 
        # add code to plot here
        # Now we plot the daily returns of our portfolio against SPY
        portVal = getPortfolioValue(ret.x, normalizedPrices)
        spyNormalized = prices_SPY / prices_SPY.iloc[0]

        # merge the 2 series into a single dataframe
        dfBoth = pd.concat([portVal, spyNormalized], keys=["Portfolio", "SPY"], axis=1)

        dfBoth.plot(title="Daily Portfolio Value vs SPY")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.grid(True)
        plt.savefig("images/{}.png".format(str("Figure1")))
 
    return statistics

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = df.copy()
    # daily_returns[1:] = (df[1:] / df[:-1].values) - 1 # compute daily returns for row 1 onwards
    daily_returns = (df / df.shift(1)) - 1  # much easier with Pandas!
    daily_returns[0] = 0   # to take care of the Nan value in the 0th row
    #daily_returns.iloc[0, :] = 0  # Pandas leaves the 0th row full of Nans
    return daily_returns

def sharpe_ratio_fn(allocations, normPrices):
    # we use k= 252 since we're taking daily samples
    # SR = sqrt(252) *  [mean(dailyReturns - dailyRiskFreeRate) / std(dailyReturns)]
    allocatedPrices = normPrices * allocations
    # assuming we're starting investment with $1
    investmentBudget = 1
    positionValues = investmentBudget * allocatedPrices
    dfPortfolioValue = positionValues.sum(axis=1)

    dailyReturns = compute_daily_returns(dfPortfolioValue)
    sr = math.sqrt(252) * dailyReturns.mean() / dailyReturns.std()

    # since the optimizer is a minimizer, and we're really interested in maximizing the sharpe ratio
    # a good trick is to minimize the -ve sharpe ratio
    return -1 * sr


def getPortfolioValue(allocs, normalizedPrices):
    allocatedPrices = normalizedPrices * allocs
    # assuming we're starting investment with $1
    investmentBudget = 1
    positionValues = investmentBudget * allocatedPrices
    dfPortfolioValue = positionValues.sum(axis=1)

    return dfPortfolioValue

def getStatistics(allocations, normPrices):
    allocatedPrices = normPrices * allocations
    # assuming we're starting investment with $1
    investmentBudget = 1
    positionValues = investmentBudget * allocatedPrices
    dfPortfolioValue = positionValues.sum(axis=1)

    dailyReturns = compute_daily_returns(dfPortfolioValue)
    cr = (dfPortfolioValue[-1]/dfPortfolioValue[0]) -1  # cummulative return
    adr = dailyReturns.mean()
    sddr = dailyReturns.std()
    sr = math.sqrt(252) * dailyReturns.mean() / dailyReturns.std()
    return (allocations, cr, adr, sddr, sr)

def test_code():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]
    #symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]
  		  	   		 	   			  		 			     			  	 
    # Assess the portfolio  		  	   		 	   			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	   			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False
    )  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # Print statistics  		  	   		 	   			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		 	   			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		 	   			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		 	   			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		 	   			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		 	   			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	   			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		 	   			  		 			     			  	 
    print(f"Cumulative Return: {cr}")  		  	   		 	   			  		 			     			  	 

if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		 	   			  		 			     			  	 
    # Do not assume that it will be called  		  	   		 	   			  		 			     			  	 
    test_code()
    #personalTest()
    #optimizeRosenbrock()
