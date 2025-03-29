import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from datetime import datetime, timedelta

# Download past historical stock data for Reliance (NSE: RELIANCE)
ticker = "RELIANCE.NS"
now = datetime.now()
start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')  # Use last 30 days
data = yf.download(ticker, interval="5m", start=start_date)

# Check if data is empty
if data.empty:
    print("No data retrieved. Check ticker symbol or market hours.")
    exit()

# Feature Engineering
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data.bfill(inplace=True)  # Fill missing values

data = data[['Close', 'Volume', 'SMA_10', 'SMA_50']]

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

# Split Data (80% Train, 20% Test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[2]))),
    Dropout(0.3),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict on test data
y_pred = model.predict(X_test)
y_pred_actual = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), data.shape[1]-1)))))[:, 0]
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), data.shape[1]-1)))))[:, 0]

# Calculate Metrics
r2 = r2_score(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

print(f"\nModel Accuracy Metrics:")
print(f"✅ R² Score: {r2:.4f}")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}\n")

# Predict prices for today minute-by-minute
total_minutes_today = (16 - now.hour) * 60 - now.minute  # Until 4 PM IST
future_input = X[-1].reshape(1, seq_length, X.shape[2])
future_predictions = []

for _ in range(total_minutes_today):
    pred = model.predict(future_input)
    future_predictions.append(pred[0, 0])
    future_input = np.roll(future_input, -1, axis=1)
    future_input[0, -1, 0] = pred  # Update Close price

# Convert predictions back to actual scale
future_predictions = scaler.inverse_transform(np.hstack((
    np.array(future_predictions).reshape(-1,1),
    np.zeros((total_minutes_today, data.shape[1]-1))
)))[:, 0]


# Print predicted prices for today
print("\n📈 Predicted Prices for Today:")
for i, price in enumerate(future_predictions):
    print(f"Minute {i+1}: ₹{price:.2f}")

# Plot predicted prices
plt.figure(figsize=(12,6))
plt.plot(range(1, total_minutes_today + 1), future_predictions, color='red', linestyle='dashed', label='Predicted Prices')
plt.title("Reliance Industries (NSE: RELIANCE) Minute-by-Minute Stock Price Prediction")
plt.xlabel("Minutes from Now")
plt.ylabel("Stock Price (₹)")
plt.legend()
plt.show()