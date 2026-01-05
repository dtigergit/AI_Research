import os
import torch
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# Get previous business days to train and evaluate the model
def get_prev_n_business_days(n, from_date=None):
    if from_date is None:
        from_date = datetime.datetime.today()
        
    business_days = pd.bdate_range(end=from_date, periods=n).to_pydatetime().tolist()
    business_days = [d.date() for d in business_days]
    
    return business_days  # Exclude the 'from_date' itself business_days[:-1] 

# The pipeline to train and evaluate the model
def run_pipeline(symbol='GDX', look_back_days=2000, sequence_length=50):

    print(f'Using LSTM model to predict price for symbol = [{symbol}]')
    
    # Look back days for training
    print(f'look_back_days = {look_back_days}')
    
    # Create sequences for LSTM (e.g. use past 50 days to predict next day)
    print(f'sequence_length = {sequence_length}')

    # Step 1: Download stock data
    stock_symbol = symbol

    bus_days = get_prev_n_business_days(look_back_days)
    start_date = bus_days[0].strftime('%Y-%m-%d')
    end_date = bus_days[-1].strftime('%Y-%m-%d')
    print(f'start_date = {start_date}, end_date = {end_date}')

    data = yf.download(stock_symbol, start=start_date, end=end_date)
    close_prices = data['Close'].values.reshape(-1, 1)  # Use closing prices

    # Step 2: Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences for LSTM (use past sequence_length days to predict next day)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM [samples, timesteps, features]

    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to PyTorch tensors
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to PyTorch tensors and move to device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, move to device, define loss and optimizer
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #train_dataset = TensorDataset(X_train, y_train)
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Step 5: Train the model
    epochs = 100
    print(f'Training for {epochs} epochs')
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Step 6: Make predictions
    model.eval()
    with torch.no_grad():
        predicted = model(X_test)
        predicted = scaler.inverse_transform(predicted.cpu().numpy())  # Move to CPU for numpy conversion
        y_test_actual = scaler.inverse_transform(y_test.cpu().numpy())  # Move to CPU for numpy conversion

    # Step 7: Evaluate and plot
    rmse = np.sqrt(mean_squared_error(y_test_actual, predicted))
    print(f'Root Mean Squared Error: {rmse:.2f}')

    # Create date range for test period
    test_start_idx = train_size + sequence_length  # Account for sequence_length offset
    test_dates = data.index[test_start_idx:test_start_idx + len(y_test_actual)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_actual, label='Actual Stock Price', linewidth=2)
    plt.plot(test_dates, predicted, label='Predicted Stock Price', linewidth=2)
    plt.title(f'{stock_symbol} Stock Price Prediction using LSTM model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    
    # Format x-axis to show dates properly
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the chart to a file since a non-interactive backend is used
    os.makedirs('plots', exist_ok=True)
    fig_path = os.path.join('plots', f"{stock_symbol}_lstm_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {fig_path}")
    
    return model, scaler, sequence_length


def predict_tomorrow_price(symbol='GLD', model=None, scaler=None, sequence_length=50):
    """
    Predict tomorrow's stock close price using a trained LSTM model
    """
    
    if model is None or scaler is None:
        print("Error: Model and scaler must be provided. Please train a model first.")
        return None
    
    print(f"\nPredicting tomorrow's close price for {symbol}...")
    
    # Get the most recent data (need sequence_length + 1 days to predict tomorrow)
    bus_days = get_prev_n_business_days(sequence_length + 10)  # Get extra days to ensure we have enough data
    start_date = bus_days[0].strftime('%Y-%m-%d')
    
    # Use next business day as end (exclusive) to include the latest business day
    next_bday_after_last = pd.bdate_range(start=bus_days[-1], periods=2)[-1]
    end_date = next_bday_after_last.strftime('%Y-%m-%d')
    
    # Download recent data
    recent_data = yf.download(symbol, start=start_date, end=end_date)
    if recent_data.empty:
        print(f"Error: Could not download data for {symbol}")
        return None
    
    # Get the most recent closing prices
    recent_close_prices = recent_data['Close'].values.reshape(-1, 1)
    
    # Scale the recent data using the same scaler
    recent_scaled = scaler.transform(recent_close_prices)
    
    # Get the last sequence_length days for prediction
    last_sequence = recent_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    
    # Convert to PyTorch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(last_sequence_tensor)
        predicted_price = scaler.inverse_transform(predicted_scaled.cpu().numpy())
    
    # Show today's actual close price and get tomorrow's predicted price
    today_close = recent_close_prices[-1][0]
    today_date = recent_data.index[-1].date()
    predicted_tomorrow = predicted_price[0][0]
    
    print(f"Today's close price ({today_date:%m-%d-%Y}): ${today_close:.2f}")
    print(f"Predicted tomorrow's close price: ${predicted_tomorrow:.2f}")
    print(f"Predicted change: ${predicted_tomorrow - today_close:.2f} ({((predicted_tomorrow - today_close) / today_close * 100):.2f}%)")
    
    return predicted_tomorrow


if __name__ == "__main__":
    
    symbol = 'GLD'
    
    # Train the model and get the trained components
    model, scaler, sequence_length = run_pipeline(symbol=symbol)
    
    # Predict tomorrow's price
    predict_tomorrow_price(symbol, model, scaler, sequence_length)