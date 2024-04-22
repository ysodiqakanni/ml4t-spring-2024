""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publiclyD viewable websites including repositories  		  	   		 	   			  		 			     			  	 
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
import random  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import pandas as pd  		  	   		 	   			  		 			     			  	 
import util as ut
import indicators
import numpy as np
from util import get_data
import RTLearner as rtl
import BagLearner as bgl

def author():
    return 'syusuff3'

class StrategyLearner(object):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    # constructor  		  	   		 	   			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.verbose = verbose  		  	   		 	   			  		 			     			  	 
        self.impact = impact  		  	   		 	   			  		 			     			  	 
        self.commission = commission
        random.seed(903953477)
        #self.learner = rtl.RTLearner(leaf_size=5)
        # let's use bagLearner
        self.learner = bgl.BagLearner(rtl.RTLearner, kwargs={"leaf_size": 10}, bags=30)
        self.lookback = 10

  		  	   		 	   			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 	   			  		 			     			  	 

    def get_stock_data(self, symbols, start_date, end_date, data="Adj Close"):
        df_prices_all = get_data(symbols, pd.date_range(start_date, end_date), colname=data)
        # now exclude SPY
        df_prices = df_prices_all[symbols]
        return df_prices

    def add_evidence(self, symbol="AAPL",
                     sd=dt.datetime(2008,1,1),
                     ed=dt.datetime(2009,12,31),
                     sv=100000):
        """  		  	   		 	   			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 

        # before doing the learning, we need to first create the dataset.
        # we will use the indicator values for the features columns
        # and we will build the y colum as follows:
            # set a value N, now iterate over days, at day i, check the price at day[i+N]
            # based on that, set y[i] to -1, 0 or 1.
        symbols = [symbol]
        #lookback = 3
        rsi = indicators.rsi_indicator(symbols, sd, ed, self.lookback)
        bbp = indicators.bollinger_bands_indicator(symbols, sd, ed, 50)
        stochastic = indicators.stochastic_indicator(symbols, sd, ed, self.lookback)
        merged_indicators = pd.concat([rsi, bbp, stochastic], axis=1)
        # add an extra column filled with zeros
        merged_indicators["Y"] = 0
        prices = self.get_stock_data(symbols, sd, ed, "Adj Close")
        # next we create a numpy array with 3 columns being the rsi, bbp and stochastic vals.
        merged_indicators.fillna(0, inplace=True)
        data = merged_indicators.values # np.random.rand((252, 4))
        rows = rsi.shape[0]     # total number of trading days/rows. Must be equal for all indicators
        N = 10      # set the lookahead period
        for day in range(self.lookback, rows-N-1):
            # compare current price and price at N
            ret = (prices.ix[day+N, symbol] / prices.ix[day, symbol]) - 1
            # check the price difference. And it has to be > (commission + impact)
            if prices.ix[day+N, symbol] < prices.ix[day, symbol] and ret < (-0.015 - self.impact):
                # y column is the third
                data[day, 3] = -1
            elif prices.ix[day+N, symbol] > prices.ix[day, symbol] and ret > (0.015 + self.impact):
                data[day, 3] = 1
            else:
                data[day, 3] = 0

        # add your code to do learning here
        # Now at this point call the DT or RT learner
        self.learner.add_evidence(data[:, 0:3], data[:, -1])

        return
  		  	   		 	   			  		 			     			  	 
        # example usage of the old backward compatible util function  		  	   		 	   			  		 			     			  	 
        syms = [symbol]  		  	   		 	   			  		 			     			  	 
        dates = pd.date_range(sd, ed)  		  	   		 	   			  		 			     			  	 
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
        prices = prices_all[syms]  # only portfolio symbols  		  	   		 	   			  		 			     			  	 
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	   			  		 			     			  	 
        if self.verbose:  		  	   		 	   			  		 			     			  	 
            print(prices)  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        # example use with new colname  		  	   		 	   			  		 			     			  	 
        volume_all = ut.get_data(  		  	   		 	   			  		 			     			  	 
            syms, dates, colname="Volume"  		  	   		 	   			  		 			     			  	 
        )  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
        volume = volume_all[syms]  # only portfolio symbols  		  	   		 	   			  		 			     			  	 
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later  		  	   		 	   			  		 			     			  	 
        if self.verbose:  		  	   		 	   			  		 			     			  	 
            print(volume)  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		 	   			  		 			     			  	 
    def testPolicy(self, symbol="AAPL",
                   sd=dt.datetime(2010,1,1),
                   ed=dt.datetime(2011,12,31),
                   sv=100000):
        """  		  	   		 	   			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 

        # here we call the query function of RTLearner
        # first we need to get the data points [the x feature - indicators]
        symbols = [symbol]
        #lookback = 14

        bbp = indicators.bollinger_bands_indicator(symbols, sd, ed, 50)
        rsi = indicators.rsi_indicator(symbols, sd, ed, self.lookback)
        stochastic = indicators.stochastic_indicator(symbols, sd, ed, self.lookback)

        merged_indicators = pd.concat([rsi, bbp, stochastic], axis=1)
        merged_indicators.fillna(0, inplace=True)
        points = merged_indicators.values
        result = self.learner.query(points)

        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol, ]].copy()  # only portfolio symbols
        trades.values[:, :] = 0  # set them all to nothing

        position = 0
        action = 0
        for day in range(1, result.shape[0]):
            if result[day] == 1:
                # sell
                action = 1000 - position
            elif result[day] == -1:
                action = -1000 - position
            else:
                action = -position
            trades.ix[day, symbol] = action

            position += action
        
        return trades

        # here we build a fake set of trades
        # your code should return the same sort of data  		  	   		 	   			  		 			     			  	 
        dates = pd.date_range(sd, ed)  		  	   		 	   			  		 			     			  	 
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
        trades = prices_all[[symbol,]]  # only portfolio symbols  		  	   		 	   			  		 			     			  	 
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	   			  		 			     			  	 
        trades.values[:, :] = 0  # set them all to nothing  		  	   		 	   			  		 			     			  	 
        trades.values[0, :] = 1000  # add a BUY at the start  		  	   		 	   			  		 			     			  	 
        trades.values[40, :] = -1000  # add a SELL  		  	   		 	   			  		 			     			  	 
        trades.values[41, :] = 1000  # add a BUY  		  	   		 	   			  		 			     			  	 
        trades.values[60, :] = -2000  # go short from long  		  	   		 	   			  		 			     			  	 
        trades.values[61, :] = 2000  # go long from short  		  	   		 	   			  		 			     			  	 
        trades.values[-1, :] = -1000  # exit on the last day  		  	   		 	   			  		 			     			  	 
        if self.verbose:  		  	   		 	   			  		 			     			  	 
            print(type(trades))  # it better be a DataFrame!  		  	   		 	   			  		 			     			  	 
        if self.verbose:  		  	   		 	   			  		 			     			  	 
            print(trades)  		  	   		 	   			  		 			     			  	 
        if self.verbose:  		  	   		 	   			  		 			     			  	 
            print(prices_all)  		  	   		 	   			  		 			     			  	 
        return trades  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    print("One does not simply think up a strategy")
    #sl = StrategyLearner()
    #ab = sl.add_evidence()
    #my_trades = sl.testPolicy()
    impact, commission = 0.005, 9.95
    symbol = "JPM"
    start_val = 100000
    start_date, end_date = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)

    strategy_learner = StrategyLearner(verbose=False, impact=impact, commission=commission)
    strategy_learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date)
    trades1 = strategy_learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)

    strategy_learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date)
    trades2 = strategy_learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)
    #trades3 = strategy_learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)


    isEqual = trades1.values == trades2.values
    print("received trades")
