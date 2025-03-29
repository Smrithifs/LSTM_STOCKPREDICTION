import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

# Download historical stock data for Tesla (TSLA)
data = yf.download("TSLA", interval="1h", start="2024-02-01", end="2024-03-04")

# Feature Engineering
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(14).mean() /
                                 -data['Close'].diff(1).clip(upper=0).rolling(14).mean())))
data.fillna(method='bfill', inplace=True)
data = data[['Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI']]

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare sequences
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting Close Price
    return np.array(X), np.array(y)

seq_length = 50  
X, y = create_sequences(data_scaled, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build the LSTM model
model = Sequential([
    Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(seq_length, X.shape[2]))),
    Dropout(0.3),
    Bidirectional(LSTM(units=128, return_sequences=True)),
    Dropout(0.3),
    LSTM(units=100, return_sequences=True),
    Dropout(0.3),
    LSTM(units=100, return_sequences=False),
    Dropout(0.3),
    Dense(units=1)
])

# Compile & train model
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict future hourly prices (next 5 days)
future_input = X[-1].reshape(1, seq_length, X.shape[2])
hourly_predictions = []
for _ in range(5 * 24):  # 5 days * 24 hours
    pred = model.predict(future_input)
    hourly_predictions.append(pred[0, 0])
    future_input = np.roll(future_input, -1, axis=1)
    future_input[0, -1, 0] = pred

# Convert predictions back to original scale
hourly_predictions = scaler.inverse_transform(
    np.hstack((np.array(hourly_predictions).reshape(-1, 1), np.zeros((120, data.shape[1] - 1)))))[:, 0]

# Print predicted values
for i, price in enumerate(hourly_predictions):
    print(f"Hour {i+1}: ${price:.2f}")

# Plot the hourly predictions
plt.figure(figsize=(12, 6))
plt.plot(range(1, 121), hourly_predictions, color='red', linestyle='dashed', label="Predicted Prices")
plt.title("Tesla (TSLA) Hourly Stock Price Prediction for Next 5 Days")
plt.xlabel("Future Hours")
plt.ylabel("Predicted Stock Price")
plt.legend()
plt.show()