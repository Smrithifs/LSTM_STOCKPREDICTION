import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

# Define the stock ticker (Change for other companies)
ticker = "AAPL"

# Download stock data
data = yf.download(ticker, start="2020-01-01", end="2024-03-04")

# Compute moving averages & RSI
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(14).mean() /
-data['Close'].diff(1).clip(upper=0).rolling(14).mean())))

# Fill missing values
data.fillna(method='bfill', inplace=True)

# Select relevant features
data = data[['Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Define sequence length
seq_length = 50

# Function to create sequences
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length, 0])  # Predicting Close Price
    return np.array(sequences), np.array(labels)

# Split data into training and testing sets
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Create sequences
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Model filename
model_filename = "30days.h5"

# Check if model already exists
if os.path.exists(model_filename):
    print("✅ Loading existing model...")
    model = load_model(model_filename)
else:
    print("🚀 Training a new LSTM model...")

    # Build LSTM Model
    model = Sequential([
        Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(units=128, return_sequences=True)),
        Dropout(0.3),
        LSTM(units=100, return_sequences=True),
        Dropout(0.3),
        LSTM(units=100, return_sequences=False),
        Dropout(0.3),
        Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

    # Save the model
    model.save(model_filename)
    print(f"✅ Model saved as '{model_filename}'!")

# Make predictions
predicted_prices = model.predict(X_test)

# Convert back to original scale
predicted_prices = scaler.inverse_transform(np.hstack((predicted_prices, np.zeros((predicted_prices.shape[0], data.shape[1]-1)))))[:, 0]
actual_prices = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((y_test.shape[0], data.shape[1]-1)))))[:, 0]

# Compute R² Score
r2 = r2_score(actual_prices, predicted_prices)
print(f"\n📊 Model Efficiency: R² Score = {r2:.4f} (Higher is better)")

# Compute error metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

print(f"\n📉 Model Performance Metrics:")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"✅ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 30-Day Future Prediction
future_input = test_data[-seq_length:]  # Last 50 days of test data
future_predictions = []

for _ in range(30):
    future_input_reshaped = np.array(future_input).reshape(1, seq_length, data.shape[1])
    next_pred = model.predict(future_input_reshaped)
    
    # Store prediction
    future_predictions.append(next_pred[0, 0])
    
    # Update input sequence
    future_input = np.vstack([future_input[1:], np.hstack([next_pred, np.zeros((1, data.shape[1]-1))])])

# Convert future predictions back to original scale
future_predictions = scaler.inverse_transform(np.hstack((np.array(future_predictions).reshape(-1,1), 
                                                          np.zeros((30, data.shape[1]-1)))))[:, 0]

# Print predicted stock prices for 30 days
print("\n📅 Predicted Stock Prices for the Next 30 Days:\n")
for i, price in enumerate(future_predictions, 1):
    print(f"Day {i}: ${price:.2f}")

# Plot 1: Actual vs Predicted Prices (Test Data) - Red & Blue
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label="Actual Prices", color='blue')
plt.plot(predicted_prices, label="Predicted Prices", color='red', linestyle='dashed')
plt.title(f"{ticker} Stock Price Prediction (Test Data)")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.show()

# Plot 2: 30-Day Future Predictions - Green & Blue
plt.figure(figsize=(12, 6))
plt.plot(range(len(data)), data['Close'], label="Historical Prices", color='blue')
plt.plot(range(len(data), len(data) + 30), future_predictions, label="30-Day Forecast", color='green', linestyle='dashed')
plt.title(f"📈 30-Day Stock Price Prediction for {ticker}")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.show()