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
    file_name = "tos_chart"
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

    return [round(cr,6), round(adr,6), round(sddr,6), port_vals[-1]]


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

    tos_statistics = getStatistics(optimized_portfolio)
    benchmark_statistics = getStatistics(benchmark_portfolio)
    #print("Benchmark CR, Mean and Std: ", benchmark_statistics[:-1])
    #print("Optimized CR, Mean and Std: ", tos_statistics[:-1])

    xval = 232
