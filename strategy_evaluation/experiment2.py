import StrategyLearner as sl
import datetime as dt
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
def author():
    return 'syusuff3'

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    # daily_returns[1:] = (df[1:] / df[:-1].values) - 1 # compute daily returns for row 1 onwards
    daily_returns = (df / df.shift(1)) - 1  # much easier with Pandas!
    daily_returns[0] = 0
    # daily_returns.iloc[0, :] = 0  # Pandas leaves the 0th row full of Nans
    return daily_returns

def getStatistics(port_vals):
    """
    computes and returns cummulative returns, std and mean of daily returns
    """
    dailyReturns = compute_daily_returns(port_vals)
    cr = (port_vals[-1] / port_vals[0]) - 1
    adr = dailyReturns.mean()
    sddr = dailyReturns.std()

    return [round(cr, 6), round(sddr, 6), round(adr, 6), round(port_vals[-1],6)]

def run_experiment():
    # use JPM in sample
    # set 3 different impact values
    # commission = 0
    # use in-sample
    # get 3 different trades
    # select 3 metrics and run exp 3 times or more
    # generate charts
    commission = 0
    start_val = 100000
    symbol = "JPM"
    start_date, end_date = dt.datetime(2008,1,1), dt.datetime(2009,12,31)
    impacts = [0.008, 0.02, 0.1, 0.2]

    plt.figure()
    for impact in impacts[0:4]:
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
        learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date)
        trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)
        portvals = compute_portvals(trades, symbol, start_val=start_val, commission=commission,
                                             impact=impact)
        strategy_normalized = portvals / portvals.iloc[0]
        plt.plot(strategy_normalized)
        #stats = getStatistics(portvals)
        #print(stats)

    plt.legend(impacts)
    plt.title("Experiment 2 - Varying impacts in-sample")
    plt.xlabel("Date")
    plt.ylabel("Normalized value")
    plt.grid(True)
    file_name = "Experiment2"
    plt.savefig("{}.png".format(str(file_name)))

