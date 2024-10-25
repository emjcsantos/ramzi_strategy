import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np

# Fetch Apple stock data
ticker = "CTXR"
data = yf.download(ticker, start="2024-04-01", end="2024-10-15")

# Calculate 20-day EMA
data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()

# Calculate Average True Range (ATR)
data['TR'] = np.maximum(data['High'] - data['Low'], 
                        np.maximum(abs(data['High'] - data['Close'].shift()), 
                                   abs(data['Low'] - data['Close'].shift())))
data['ATR'] = data['TR'].rolling(window=10).mean()

# Calculate Upper Keltner Channels
data['Upper_KC_3'] = data['EMA21'] + (3 * data['ATR'])
data['Upper_KC_4'] = data['EMA21'] + (4 * data['ATR'])

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.plot(data.index, data['EMA21'], label='21-day EMA', color='red')
plt.plot(data.index, data['Upper_KC_3'], label='Upper Keltner Channel (3x)', color='green')
plt.plot(data.index, data['Upper_KC_4'], label='Upper Keltner Channel (4x)', color='purple')

plt.title(f'{ticker} Stock Price with Upper Keltner Channels')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()