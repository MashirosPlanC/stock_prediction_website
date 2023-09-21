import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(model, X_train, y_train, epochs, batch_size, learning_rate):
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size, :, :]
            y_batch = y_train[i : i + batch_size, :]
            outputs = model(X_batch)
            optimizer.zero_grad()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

def predict_stock_prices_lstm(stock_prices, days_to_predict):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_stock_prices = scaler.fit_transform(stock_prices.values.reshape(-1, 1))

    # Prepare the training data
    X_train = []
    y_train = []
    window_size = 60
    for i in range(window_size, len(scaled_stock_prices)):
        X_train.append(scaled_stock_prices[i - window_size:i, 0])
        y_train.append(scaled_stock_prices[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)

    # Train the LSTM model
    input_dim = 1
    hidden_dim = 50
    num_layers = 2
    output_dim = 1
    lstm_model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    train_lstm_model(lstm_model, X_train, y_train, epochs, batch_size, learning_rate)

    # Prepare the test data
    X_test = []
    for i in range(len(stock_prices) - window_size, len(stock_prices) + days_to_predict - window_size):
        X_test.append(scaled_stock_prices[i - window_size:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_test = torch.FloatTensor(X_test)

    # Predict the stock prices
    lstm_model.eval()
    with torch.no_grad():
        predicted_stock_prices = lstm_model(X_test).numpy()
        predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)

    prediction_dates = pd.date_range(stock_prices.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
    prediction_dates = [date.strftime("%Y-%m-%d") for date in prediction_dates]
    result = list(zip(prediction_dates, predicted_stock_prices.flatten()))
    return result

# 使用方法如下, result 是一个 list, 其中的内容是 (日期，价格)

# # Fetch stock prices
# stock_symbol = "GOOGL"
# start_date = "2020-01-01"
# end_date = "2021-12-31"
# stock_prices = fetch_stock_prices(stock_symbol, start_date, end_date)

# # Predict future stock prices
# days_to_predict = 30
# result = predict_stock_prices_lstm(stock_prices, days_to_predict)