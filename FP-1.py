
"""
Created on Thu May  9 15:42:14 2024

@author: Mr. Marcus.Becker & GreenTech Wealth Innovations

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns 
from pandas_datareader import data
import yfinance as yfin
yfin.pdr_override() 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.optimize import minimize

def getData(ticker, start_date, end_date):
    
    
    ticker_list = list(ticker.Ticker)
    
    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    panel_data =  data.get_data_yahoo(ticker_list,
                                     start = start_date,
                                     end = end_date)


    # Create a Panda DatFrame from multicolumn data
    df = pd.DataFrame()

    for t in ticker.Ticker:
        #print(t)
        df[t] = pd.Series(panel_data[('Adj Close',  t)].iloc[:].values, 
                       index = panel_data.index)

    # Use Ticker Labels as Column Name for easier reckognition
    df.columns = ticker.Label
    
    return df


def cleanData(data, method = 'forward'):
    
    
    if method == 'forward':
        
        # Use forward fill to replace missing data (Default)
        # If present ETF price is unknown, use last known price
        # Fundamental Theorem  S_t = E_Q[S_t+1/1+rf] => Martingale Property
        # If rf approx 0 exp. asset prices of tomorrow are similiar to todays prices!    
        cleanedData = data.ffill()
        
    elif method == 'backward':
        
        #Use future known price
        cleanedData = data.bfill()
        
    elif method == 'interpolate':
        
        # Interpolate between last known prices (before and after missing values)
        cleanedData = data.interpolate()
        
    elif method == 'dropna':
        
        # Drop missing price rows (reduces the dataset if nan values are prevelant)
        cleanedData = data.dropna()     
    
    else:
        
        print("Unknown cleaning method. Use forward, backward, interpolate or dropna as option.")
        
    return cleanedData
    



# =============================================================================
#  Import Data
# =============================================================================
    
# Get the ticker list including labels of the Tickers
ticker = pd.read_excel("E:/Annaconda/.spyder-py3/Dataset/Green ETF Selection.xlsx",sheet_name = "ETF_Universe")

start_date = '2020-01-01'
end_date = '2024-03-15'

# Ge the pricing Data (needs internet connection)
df = getData(ticker, start_date, end_date)


# =============================================================================
# Clean Data
# =============================================================================

# Resampling dataset with mean intraday prices
df_cleaned = df.resample('D').mean()

# use forward/backward propagation for missing prices 
df_cleaned = cleanData(df_cleaned, method = 'backward') # if  very first entry is missing
df_cleaned = cleanData(df_cleaned, method = 'forward')  # if  very last entry is missing

# =============================================================================
# ANN Model For Forecasting 
# =============================================================================

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_cleaned)

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.7)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Prepare training and testing datasets
def prepare_data(df_cleaned):
    X, y = [], []
    for i in range(len(df_cleaned) - 1):
        X.append(df_cleaned[i])
        y.append(df_cleaned[i + 1])
    return np.array(X), np.array(y)

X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)


# Define the ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(7)  # Output layer with 7 neurons for 7 ETFs
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Inverse scaling for predictions
predictions_unscaled = scaler.inverse_transform(predictions)

# Define a list of colors
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan']

plt.figure(figsize=(14, 7))
for i in range(len(df_cleaned.columns)):  
    color = colors[i % len(colors)]  
    plt.plot(df_cleaned.index[train_size+1:], df_cleaned.iloc[train_size+1:, i], label=f"Actual {df_cleaned.columns[i]} Price", color=color)
    plt.plot(df_cleaned.index[train_size+1:], predictions_unscaled[:, i], label=f"Predicted {df_cleaned.columns[i]} Price", linestyle='dashed', color=color)
plt.title('ETF Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()


# Number of trading days in 3 months
num_days_3m = 66

# Use the last available data point as initial features for prediction
X_pred_3m = np.array([test_data[-1]])

# Make predictions for 3 months ahead
predictions_3m = []
for _ in range(num_days_3m):
    # Predict next day's price
    pred_next_day = model.predict(X_pred_3m[-1].reshape(1, -1))
    predictions_3m.append(pred_next_day[0])
    # Update X_pred_3m for next iteration
    X_pred_3m = np.append(X_pred_3m, pred_next_day, axis=0)

# Inverse scaling for predictions
predictions_3m_unscaled = scaler.inverse_transform(predictions_3m)

# Create a date range for the next 6 months
next_3m_dates = pd.date_range(df_cleaned.index[-1], periods=num_days_3m + 1)[1:]

# Plot actual vs. predicted prices for the next 3 months
plt.figure(figsize=(14, 7))
for i, etf in enumerate(df_cleaned.columns):
    plt.plot(df_cleaned.index[train_size+1:], df_cleaned.iloc[train_size+1:, i], label=f"Actual {etf} Price", color=colors[i % len(colors)])
    plt.plot(next_3m_dates, predictions_3m_unscaled[:, i], label=f"Predicted {etf} Price", linestyle='dashed', color=colors[i % len(colors)])
plt.title('ETF Price Prediction for the Next 3 Months')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

# Number of trading days in 6 months
num_days_6m = 132

# Use the last available data point as initial features for prediction
X_pred_6m = np.array([test_data[-1]])

# Make predictions for 6 months ahead
predictions_6m = []
for _ in range(num_days_6m):
    # Predict next day's price
    pred_next_day = model.predict(X_pred_6m[-1].reshape(1, -1))
    predictions_6m.append(pred_next_day[0])
    # Update X_pred_6m for next iteration
    X_pred_6m = np.append(X_pred_6m, pred_next_day, axis=0)

# Inverse scaling for predictions
predictions_6m_unscaled = scaler.inverse_transform(predictions_6m)

# Create a date range for the next 6 months
next_6m_dates = pd.date_range(df_cleaned.index[-1], periods=num_days_6m + 1)[1:]

# Plot actual vs. predicted prices for the next 6 months
plt.figure(figsize=(14, 7))
for i, etf in enumerate(df_cleaned.columns):
    plt.plot(df_cleaned.index[train_size+1:], df_cleaned.iloc[train_size+1:, i], label=f"Actual {etf} Price", color=colors[i % len(colors)])
    plt.plot(next_6m_dates, predictions_6m_unscaled[:, i], label=f"Predicted {etf} Price", linestyle='dashed', color=colors[i % len(colors)])
plt.title('ETF Price Prediction for the Next 6 Months')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

# Calculate Returns
def calculate_returns(predictions, start_price):
    return (predictions - start_price) / start_price

# Calculate cumulative returns
cumulative_returns_3m = (predictions_3m_unscaled - predictions_3m_unscaled[0]) / predictions_3m_unscaled[0]
cumulative_returns_6m = (predictions_6m_unscaled - predictions_6m_unscaled[0]) / predictions_6m_unscaled[0]

# Plot actual vs. predicted prices and cumulative returns for the next 3 months
plt.figure(figsize=(14, 7))
for i, etf in enumerate(df_cleaned.columns):
    plt.plot(df_cleaned.index[train_size+1:], df_cleaned.iloc[train_size+1:, i], label=f"Actual {etf} Price", color=colors[i % len(colors)])
    plt.plot(next_3m_dates, predictions_3m_unscaled[:, i], label=f"Predicted {etf} Price", linestyle='dashed', color=colors[i % len(colors)])
    plt.plot(next_3m_dates, cumulative_returns_3m[:, i] + 1, label=f"Cumulative Returns {etf}", linestyle='dotted', color=colors[i % len(colors)])
plt.title('ETF Price Prediction and Cumulative Returns for the Next 3 Months')
plt.xlabel('Date')
plt.ylabel('Price and Returns')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()


# Plot actual vs. predicted prices and cumulative returns for the next 6 months
plt.figure(figsize=(14, 7))
for i, etf in enumerate(df_cleaned.columns):
    plt.plot(df_cleaned.index[train_size+1:], df_cleaned.iloc[train_size+1:, i], label=f"Actual {etf} Price", color=colors[i % len(colors)])
    plt.plot(next_6m_dates, predictions_6m_unscaled[:, i], label=f"Predicted {etf} Price", linestyle='dashed', color=colors[i % len(colors)])
    plt.plot(next_6m_dates, cumulative_returns_6m[:, i] + 1, label=f"Cumulative Returns {etf}", linestyle='dotted', color=colors[i % len(colors)])
plt.title('ETF Price Prediction and Cumulative Returns for the Next 6 Months')
plt.xlabel('Date')
plt.ylabel('Price and Returns')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

# Calculate returns for 3 months and 6 months
returns_3m = calculate_returns(predictions_3m_unscaled[-1], df_cleaned.iloc[train_size, :].values)
returns_6m = calculate_returns(predictions_6m_unscaled[-1], df_cleaned.iloc[train_size, :].values)

# Print the returns
print("Returns for the next 3 months:")
for etf, ret in zip(df_cleaned.columns, returns_3m):
    print(f"{etf}: {ret:.2%}")

print("\nReturns for the next 6 months:")
for etf, ret in zip(df_cleaned.columns, returns_6m):
    print(f"{etf}: {ret:.2%}")

# Calculating Accuracy
train_loss = model.evaluate(X_train, y_train)
test_loss = model.evaluate(X_test, y_test)
train_accuracy = (1 - train_loss) * 100
test_accuracy = (1 - test_loss) * 100

print("Training Accuracy:", train_accuracy, "%")
print("Testing Accuracy:", test_accuracy, "%")

# Correlation Analysis
correlation_matrix = df_cleaned.corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix')
plt.xticks(range(len(df_cleaned.columns)), df_cleaned.columns, rotation=45)
plt.yticks(range(len(df_cleaned.columns)), df_cleaned.columns)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

# Portfolio Optimization Based on Predicted Returns

# Calculate expected returns based on predictions
predicted_returns = np.mean(predictions_unscaled, axis=0)

# Define objective function to minimize (negative Sharpe ratio)
def objective(weights):
    portfolio_return = np.sum(predicted_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(correlation_matrix, weights)))
    return -portfolio_return / portfolio_volatility

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(len(df_cleaned.columns)))

# Initial guess
init_guess = [1./len(df_cleaned.columns) for _ in range(len(df_cleaned.columns))]

# Optimization
opt_results = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_results.x

# Display Optimal Weights
print("Optimal Weights:", optimal_weights)

# Plotting the optimal asset allocation
plt.figure(figsize=(10, 6))
plt.bar(df_cleaned.columns, optimal_weights, color=colors)
plt.xlabel('ETF')
plt.ylabel('Weight')
plt.title('Optimal Asset Allocation')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

#=====================================================
# Portfolio Optimization Based on Predicted Returns
#=====================================================

# Calculate expected returns based on predictions
predicted_returns = np.mean(predictions_unscaled, axis=0)

# Define objective function to minimize (negative Sharpe ratio)
def objective(weights):
    portfolio_return = np.sum(predicted_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(correlation_matrix, weights)))
    return -portfolio_return / portfolio_volatility

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(len(df_cleaned.columns)))

# Initial guess
init_guess = [1./len(df_cleaned.columns) for _ in range(len(df_cleaned.columns))]

# Optimization
opt_results = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_results.x

# Display Optimal Weights
print("Optimal Weights:", optimal_weights)

# Allocate $100,000 based on optimal weights
portfolio_value = 100000
allocation = portfolio_value * optimal_weights

# Display Allocation
allocation_df = pd.DataFrame({'ETF': df_cleaned.columns, 'Optimal Weight': optimal_weights, 'Allocation ($)': allocation})
print(allocation_df)

# Plotting the optimal asset allocation
plt.figure(figsize=(10, 6))
plt.bar(df_cleaned.columns, allocation, color=colors)
plt.xlabel('ETF')
plt.ylabel('Allocation ($)')
plt.title('Optimal Asset Allocation for $100,000')
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

