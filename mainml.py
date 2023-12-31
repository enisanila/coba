import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
import tensorflow as tf

# Load data
file_path = './Dataset/inflasi_v3.csv'
df = pd.read_csv(file_path)

# Select relevant columns
df = df[['City', 'Month', 'Year', 'Inflation', 'CPI']]

# Remove 'KOTA' from the 'City' column
df['City'] = df['City'].str.replace('KOTA ', '', regex=False)

# Check if 'Inflation' and 'CPI' columns already contain numeric values
for col in ['Inflation', 'CPI']:
    if not pd.api.types.is_numeric_dtype(df[col]):
        # Replace commas with dots and convert to float
        df[col] = df[col].str.replace(',', '.').astype(float)

# Sort the data
df = df.sort_values(by=['City', 'Year', 'Month'])

# Scale inflation data and other numerical features
scaler = MinMaxScaler()
df[['Inflation', 'CPI']] = scaler.fit_transform(df[['Inflation', 'CPI']].values)

# Separate data into time series for each city
city_data = {}
for city in df['City'].unique():
    city_data[city] = df[df['City'] == city][['Inflation', 'CPI']].values

# Combine data for all cities into a single time series
all_cities_data = np.concatenate(list(city_data.values()))

# Function to create time series sequences
def create_time_series(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps + 1):
        a = data[i:(i + time_steps), :]
        X.append(a)
        y.append(data[i + time_steps - 1, 0])  # Assuming 'Inflation' is in the first column
    return np.array(X), np.array(y)

# Hyperparameters
time_steps = 12
n_features = 2

# Create time series for all cities
X_all, y_all = create_time_series(all_cities_data, time_steps)
X_all = X_all.reshape((X_all.shape[0], time_steps, n_features))

# Create and train a single LSTM model for all cities
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(time_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_all, y_all, epochs=50, verbose=0)

# Save the Keras model in the native format
model.save('./Final_Model/all_cities_lstm_model_v3.h5')

# Example prediction for a city
sample_city = 'YOGYAKARTA'
input_data = scaler.transform(city_data[sample_city][-time_steps:])
input_data = input_data.reshape((1, time_steps, n_features))
predicted_inflation = model.predict(input_data)

# Inverse transform the prediction to get the actual value
predicted_inflation_actual = scaler.inverse_transform(np.concatenate([predicted_inflation, np.zeros_like(predicted_inflation)], axis=1))
predicted_inflation_actual = predicted_inflation_actual[:, 0]
print(f'Predicted Inflation for {sample_city}: {predicted_inflation_actual[0]}')

# Compare with the most recent actual inflation value
historical_inflation = city_data[sample_city]
last_actual_inflation = historical_inflation[-1, 0]  # Assuming 'Inflation' is in the first column

# Determine the trend
if predicted_inflation_actual[0] > last_actual_inflation:
    trend = 'up'
elif predicted_inflation_actual[0] < last_actual_inflation:
    trend = 'down'
else:
    trend = 'unchanged'

print(f'Trend for {sample_city} inflation: {trend}')