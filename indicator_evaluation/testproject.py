import TheoreticallyOptimalStrategy as tos
import datetime as dt
from marketsimcode import compute_portvals

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
    # Let's get a benchmark: The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM,
    df_benchmark_trades = df_trades.copy()
    df_benchmark_trades[:] = 0  # set all the values to zero
    df_benchmark_trades[df_benchmark_trades.index[0]] = 1000

    optimized_portfolio = compute_portvals(df_trades, symbol="JPM", start_val=100000,commission=0,impact=0)
    benchmark_portfolio = compute_portvals(df_benchmark_trades, symbol="JPM", start_val=100000,commission=0,impact=0)

    print(df_trades.values)
