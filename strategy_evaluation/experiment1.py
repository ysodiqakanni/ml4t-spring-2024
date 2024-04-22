import pandas as pd

import ManualStrategy as ms
import StrategyLearner as sl
import datetime as dt
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

def author():
    return 'syusuff3'

def compute_and_plot(start_date, end_date, file_name):
    impact, commission = 0.005, 9.95
    symbol = "JPM"
    start_val = 100000

    manual_learner = ms.ManualStrategy(verbose=False, impact=impact, commission=commission)
    manual_trades = manual_learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)
    benchmark_trades = manual_trades.copy()
    benchmark_trades[:] = 0
    benchmark_trades.loc[benchmark_trades.index[0], symbol] = 1000
    manual_portvals = compute_portvals(manual_trades, symbol, start_val, commission, impact)

    strategy_learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    strategy_learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date)
    strategy_trades = strategy_learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)
    strategy_portvals = compute_portvals(strategy_trades, symbol, start_val=start_val, commission=commission,
                                         impact=impact)

    #benchmark_trades = pd.DataFrame(columns=[symbol], index=[manual_trades.index[0]])  # manual_trades.copy()
    #benchmark_trades.ix[manual_trades.index[0], symbol] = 1000
    benchmark_portvals = compute_portvals(benchmark_trades, symbol, start_val=start_val, commission=commission, impact=impact)

    manual_normalized = manual_portvals/manual_portvals.iloc[0]
    strategy_normalized = strategy_portvals/strategy_portvals.iloc[0]
    benchmark_normalized = benchmark_portvals/benchmark_portvals.iloc[0]

    plt.figure()
    manual_normalized.plot(color='r')
    strategy_normalized.plot(color='g')
    benchmark_normalized.plot(color='purple')

    plt.legend(['Manual', 'Strategy', 'Benchmark'])
    plt.title("Manual vs strategy vs benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized value")
    plt.grid(True)
    #file_name = "ManualStrategy"
    plt.savefig("{}.png".format(str(file_name)))

    return strategy_normalized

def run_experiment():
    # comparing manual and strategy learners
    # A. using in-sample JPM, get the trades df from manual and strategy
    # period is from January 1, 2008, to December 31, 2009.
    # Commission: $9.95, Impact: 0.005

    # in-sample
    indata = compute_and_plot(dt.datetime(2008,1,1), dt.datetime(2009,12,31), "InSample")
    # out-sample
    outdata = compute_and_plot(dt.datetime(2010,1,1), dt.datetime(2011,12,31), "OutSample")


    return indata, outdata


    impact, commission = 0.005, 9.95
    symbol = "JPM"
    start_val = 100000
    start_date, end_date = dt.datetime(2008,1,1), dt.datetime(2009,12,31)
    manual_learner = ms.ManualStrategy(verbose=False, impact=impact, commission=commission)
    manual_trades = manual_learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)
    manual_portvals = compute_portvals(manual_trades,symbol,start_val, commission, impact)

    strategy_learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    strategy_learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date)
    strategy_trades = strategy_learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)
    strategy_portvals = compute_portvals(strategy_trades, symbol, start_val=start_val, commission=commission, impact=impact)

    benchmark_trades_in_sample = pd.DataFrame(columns=[symbol], index=[manual_trades.index[0]])  # manual_trades.copy()
    benchmark_trades_in_sample.ix[manual_trades.index[0], symbol] = 1000
    #benchmark_trades_in_sample.append({symbol:1000})

    # Now let's test out of sample
    os_start_date, os_end_date = dt.datetime(2010,1,1), dt.datetime(2011,12,31)
    os_manual_trades = manual_learner.testPolicy(symbol=symbol, sd=os_start_date, ed=os_end_date)
    os_port_vals = compute_portvals(os_manual_trades, symbol, start_val, commission, impact)

    os_strategy_trades = strategy_learner.testPolicy(symbol=symbol, sd=os_start_date, ed=os_end_date)
    os_strategy_port_vals = compute_portvals(os_strategy_trades, symbol, start_val, commission, impact)
    # get portfolio values of each
    # plot portfolio values of manual vs strategy vs benchmark

    # B. Use out-sample JPM
    # out-of-sample/testing period is from January 1, 2010, to December 31, 2011.
    # Commission: $9.95, Impact: 0.005
    # get trades from manual and strategy by passing out-sample dates
    # calculate the port_vals out_sample for manual, strategy and benchmark
    # plot the port_vals in a single chart

    return