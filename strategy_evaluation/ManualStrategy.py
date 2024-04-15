import datetime as dt
import pandas as pd
import util as ut
import indicators
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

class ManualStrategy(object):
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose


    def testPolicy(self, symbol = "AAPL",
                   sd=dt.datetime(2010, 1, 1),
                   ed=dt.datetime(2011,12,31),
                   sv=100000):
        # First trade would be start_date + 1
        # for simple case (just 1 indicator)
        # using the symbol and date range, call the indicator fn and get the signal values as a single vector
        # now create a for loop of days
        # use some logic to create an order for the day
        # add to the orders array
        # return the orders array.
        # done!!
        # Theoretically optimal = $678610

        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol, ]]  # only portfolio symbols
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later
        trades.values[:, :] = 0  # set them all to nothing

        symbols = [symbol]
        lookback = 14
        rsi_vals = indicators.rsi_indicator(symbols, sd, ed, lookback)
        bbp = indicators.bollinger_bands_indicator(symbols,sd, ed, 14)
        stochastic = indicators.stochastic_indicator(symbols, sd, ed, lookback)

        # Note that: Allowable positions are 1000 shares long, 1000 shares short, 0 shares.
        position = 0
        action = 0
        # use the index of rsi to create the loop
        for day in range(rsi_vals.shape[0]):
            if position > 2000 or position < -2000:
                print("danger here! pos=", position)
            if rsi_vals.ix[day, symbol] < 30 or bbp.ix[day, symbol] < 0.2 and stochastic.ix[day, symbol] < 20:
                # BUY
                #trades.append([rsi_vals.index[day].date(), symbol, "BUY", 1000])
                #trades.values[day, :] = 1000 - position
                action = 1000 - position
                trades.values[day, :] = action
            elif rsi_vals.ix[day, symbol] > 70 or bbp.ix[day, symbol] > 0.8 and stochastic.ix[day, symbol] > 80:
                #trades.append([rsi_vals.index[day].date(), symbol, "SELL", 1000])
                #trades.values[day, :] = -1000 - position
                action = -1000 - position
                trades.values[day, :] = action
            else:
                #print("what do we do here?")
                # here we do nothing. So if we're currently at 1000, we go -1000
                action = -position
            position += action

        return trades


def generate_plots(optimized, benchmark):
    # normalize by dividing by the first values
    optimized = optimized/optimized.iloc[0]
    benchmark = benchmark/benchmark.iloc[0]
    plt.figure()
    optimized.plot(color='r')
    benchmark.plot(color='purple')
    plt.legend(['Manual Strategy', 'Benchmark'])
    plt.title("Manual strategy vs benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized value")
    plt.grid(True)
    file_name = "ManualStrategy"
    plt.savefig("{}.png".format(str(file_name)))

if __name__ == "__main__":
    print("One does not simply think up a strategy")
    sym = "JPM"
    ms = ManualStrategy()
    orders = ms.testPolicy(symbol=sym, sd=dt.datetime(2008,1,1),
                           ed=dt.datetime(2009, 12, 31))

    df_benchmark_trades = orders.copy()
    df_benchmark_trades[:] = 0  # set all the values to zero
    # now invest in 1000 shares and hold that position
    df_benchmark_trades.loc[df_benchmark_trades.index[0], sym] = 1000

    optimized_portfolio = compute_portvals(orders, symbol=sym, start_val=100000, commission=9.95, impact=0.005)
    benchmark_portfolio = compute_portvals(df_benchmark_trades, symbol=sym, start_val=100000,commission=9.95,impact=0.005)
    generate_plots(optimized_portfolio, benchmark_portfolio)
    stats_manual = indicators.getStatistics(optimized_portfolio)
    stats_benchmark = indicators.getStatistics(benchmark_portfolio)
    print("Manual statistics: CR, SD, Mean, Portfolio", stats_manual)
    print("Benchmark statistics: CR, SD, Mean, Portfolio", stats_benchmark)
    print("Done running")
