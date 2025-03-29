import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Download stock data for Tesla (TSLA)
data = yf.download("TSLA", start="2020-01-01", end="2024-03-04")

# Compute Moving Averages
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Compute Relative Strength Index (RSI)
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Fill missing values
data.fillna(method='bfill', inplace=True)

# 1️⃣ Stock Price Trend Over Time 📈
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Close'], label="Closing Price", color="blue")
plt.title("Tesla (TSLA) Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# 2️⃣ Volume vs. Closing Price Scatter Plot 📊
plt.figure(figsize=(10,5))
plt.scatter(data['Volume'], data['Close'], alpha=0.5, color='green')
plt.xlabel("Trading Volume")
plt.ylabel("Stock Price")
plt.title("Trading Volume vs. Closing Price (TSLA)")
plt.show()

# 3️⃣ Moving Averages Overlaid on Closing Price 📉
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Close'], label="Closing Price", color="black")
plt.plot(data.index, data['SMA_10'], label="10-Day SMA", color="red", linestyle="dashed")
plt.plot(data.index, data['SMA_50'], label="50-Day SMA", color="blue", linestyle="dashed")
plt.title("Tesla Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# 4️⃣ RSI Over Time 📊
plt.figure(figsize=(12,5))
plt.plot(data.index, data['RSI'], label="RSI", color="purple")
plt.axhline(70, linestyle="dashed", color="red", label="Overbought (70)")
plt.axhline(30, linestyle="dashed", color="green", label="Oversold (30)")
plt.title("Relative Strength Index (RSI) for Tesla")
plt.xlabel("Date")
plt.ylabel("RSI Value")
plt.legend()
plt.show()