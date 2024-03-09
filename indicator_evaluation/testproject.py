import TheoreticallyOptimalStrategy as tos
import datetime as dt

"""
instructions
Add an author() function to each file.  
For your report, use only the symbol JPM.  
Use the time period January 1, 2008, to December 31, 2009.
Starting cash is $100,000.  
"""
def author():
    return 'syusuff3'


if "__main__" == "__main__":
    # generate plots by making the following calls:
    # call testpolicy
    # call indicators
    # call marketsimcode
    print("getting started")

    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print(df_trades.values)
