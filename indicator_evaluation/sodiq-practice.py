
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Simple Moving Average (SMA)
def calculate_sma(df, window=20):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    return df

# Function to identify buy and sell signals
def identify_signals(df):
    df['Signal'] = 0  # 0 represents no signal

    # Golden Cross (Buy Signal)
    df.loc[df['Close'] > df['SMA'], 'Signal'] = 1

    # Death Cross (Sell Signal)
    df.loc[df['Close'] < df['SMA'], 'Signal'] = -1

    return df

def dummy_sma():
    # Sample data (replace with your actual financial data)
    data = {'Date': pd.date_range(start='2022-01-01', periods=100),
            'Close': np.random.rand(100) * 10 + 50}

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    # Calculate Simple Moving Average (SMA)
    df = calculate_sma(df)

    # Identify Buy and Sell signals
    df = identify_signals(df)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='black')
    plt.plot(df.index, df['SMA'], label='SMA', color='blue')

    # Buy signals (Golden Cross)
    buy_signals = df[df['Signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal')

    # Sell signals (Death Cross)
    sell_signals = df[df['Signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal')

    plt.title('Stock Price and SMA - Buy and Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dummy_sma()