import pandas as pd
import os
import matplotlib.pyplot as plt


def print_spy():
    df = pd.read_csv("../data/SPY.csv")
    print(df)


def symbol_to_path(symbol, base_dir="../data"):
    """get the path to the symbol"""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


# using a utility function to merge symbols
def get_data(symbols, dates):
    """read stock data (adj close) for the symbols for the given dates"""
    df = pd.DataFrame(index=dates)
    if "SPY" not in symbols:
        symbols.insert(0, "SPY")
    for symbol in symbols:
        # get each symbol data and merge
        df_temp = pd.read_csv(symbol_to_path(symbol),
                              index_col="Date", parse_dates=True,
                              usecols=["Date", "Adj Close"],
                              na_values=["nan"])

        df_temp = df_temp.rename(columns=({"Adj Close": symbol}))
        df = df.join(df_temp)
        if symbol == "SPY":     # drop dates that SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def normalize_data(df):
    """Normalize stock prices using the first row of the dataframe."""
    return df / df.ix[0, :]


def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    plot_data(df.ix[start_index:end_index, columns], title="Selected data")


def test_run():
    # Define a date range
    dates = pd.date_range('2010-01-01', '2010-12-31')
    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']  # SPY will be added in get_data()

    # Get stock data
    df = get_data(symbols, dates)

	# Slice by row range (dates) using DataFram.ix[] selector
    #print(df.ix['2010-01-01':'2010-01-31'])  # the month of January

    #  Slice by column (symbols)
    #print(df['GOOG']) # a single label selects a single column
    #print(df[['IBM', 'GLD']]) # a list of labels selects multiple columns

    # Slice by row and column
    #print(df.ix['2010-03-01':'2010-03-15', ['SPY', 'IBM']])
    #plot_data(df)

    dfj = df.ix['2010-01-01':'2010-01-31']  # for january
    print(dfj)
    print(f"mean = \n{dfj.mean()}")
    print(f"standard deviation = \n{dfj.std()}")


def personal_run():
    symbols = ["IBM", "GOOG", "GLD"]
    start_date = '2010-01-22'
    end_date = '2010-01-26'
    dates = pd.date_range(start_date, end_date)
    dfm = get_data(symbols, dates)
    # extract only some parts for goog and gld
    dfe = dfm.ix["2010-01-25": "2010-01-26", ["GOOG", "GLD"]]
    print(dfm)
    print(dfe)

if __name__ == "__main__":
    test_run()