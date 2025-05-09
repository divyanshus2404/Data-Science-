import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import math
import itertools

# Load dataset from Yahoo Finance
ticker = 'AAPL'
df = yf.download(ticker, start='2010-01-01', end='2024-01-01')
df = df[['Close']]
df.dropna(inplace=True)

# Plot stock price
df.plot(figsize=(12,6), title=f'{ticker} Stock Price')
plt.show()

# Check stationarity
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] > 0.05:
        print('Series is non-stationary')
    else:
        print('Series is stationary')

adf_test(df['Close'])

# Differencing to make stationary
df_diff = df['Close'].diff().dropna()
adf_test(df_diff)

# Train-test split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Function to determine optimal p, d, q
def optimize_arima(train):
    best_aic = float("inf")
    best_order = None
    best_model = None
    p_values = range(0, 4)
    d_values = range(0, 3)
    q_values = range(0, 4)
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train, order=(p, d, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, d, q)
                best_model = model
        except:
            continue
    return best_order, best_model

# Get best ARIMA parameters
best_order, darima = optimize_arima(train)
pred_arima = darima.forecast(len(test))
mse_arima = mean_squared_error(test, pred_arima)
rmse_arima = math.sqrt(mse_arima)
r2_arima = r2_score(test, pred_arima) * 100

# SARIMA Model
sarima = SARIMAX(train, order=best_order, seasonal_order=(1,1,1,12)).fit()
pred_sarima = sarima.forecast(len(test))
mse_sarima = mean_squared_error(test, pred_sarima)
rmse_sarima = math.sqrt(mse_sarima)
r2_sarima = r2_score(test, pred_sarima) * 100

# LSTM Model
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)
train_scaled, test_scaled = df_scaled[:train_size], df_scaled[train_size:]

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Predictions
pred_lstm = model.predict(X_test)
pred_lstm = scaler.inverse_transform(pred_lstm)
mse_lstm = mean_squared_error(test[seq_length:], pred_lstm)
rmse_lstm = math.sqrt(mse_lstm)
r2_lstm = r2_score(test[seq_length:], pred_lstm) * 100

# Compare Accuracy
print(f'Best ARIMA Order: {best_order}')
print(f'RMSE ARIMA: {rmse_arima}, Accuracy: {r2_arima:.2f}%')
print(f'RMSE SARIMA: {rmse_sarima}, Accuracy: {r2_sarima:.2f}%')
print(f'RMSE LSTM: {rmse_lstm}, Accuracy: {r2_lstm:.2f}%')

# Plot results
plt.figure(figsize=(12,6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, pred_arima, label='ARIMA')
plt.plot(test.index, pred_sarima, label='SARIMA')
plt.plot(test.index[seq_length:], pred_lstm, label='LSTM')
plt.legend()
plt.show()