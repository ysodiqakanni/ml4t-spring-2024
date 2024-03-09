import TheoreticallyOptimalStrategy as tos
import datetime as dt
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

"""
instructions
Add an author() function to each file.  
For your report, use only the symbol JPM.  
Use the time period January 1, 2008, to December 31, 2009.
Starting cash is $100,000.  
"""
def author():
    return 'syusuff3'

def generate_tos_plots(optimized, benchmark):
    # normalize by dividing by the first values
    optimized = optimized/optimized.iloc[0]
    benchmark = benchmark/benchmark.iloc[0]
    plt.figure()
    optimized.plot(color='r')
    benchmark.plot(color='purple')
    # plt.plot(optimized, color='red')
    # plt.plot(benchmark, color='purple')
    plt.legend(['Optimized', 'Benchmark'])
    plt.title("Theoretically optimal strategy vs benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized value")
    plt.grid(True)
    plt.show()


if "__main__" == "__main__":
    # generate plots by making the following calls:
    # call testpolicy
    # call indicators
    # call marketsimcode
    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # Let's get a benchmark: The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM,
    df_benchmark_trades = df_trades.copy()
    df_benchmark_trades[:] = 0  # set all the values to zero
    # now invest in 1000 shares and hold that position
    df_benchmark_trades.loc[df_benchmark_trades.index[0], "JPM"] = 1000

    optimized_portfolio = compute_portvals(df_trades, symbol="JPM", start_val=100000,commission=0,impact=0)
    benchmark_portfolio = compute_portvals(df_benchmark_trades, symbol="JPM", start_val=100000,commission=0,impact=0)
    generate_tos_plots(optimized_portfolio, benchmark_portfolio)

    #print(df_trades.values)
