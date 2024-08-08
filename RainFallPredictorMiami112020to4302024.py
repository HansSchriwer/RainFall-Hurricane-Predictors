#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# In[9]:


# Load historical rainfall data (replace with your data)
# Example data loading (uncomment and replace with actual data loading code)
# rainfall_data = pd.read_csv('rainfall_data.csv')

# For demonstration purposes, let's create a dummy DataFrame
# Uncomment the above line and comment this block when using actual data


# In[45]:


dates = pd.date_range(start='2020-01-01', periods=100)
rainfall = np.random.lognormal(mean=0, sigma=0.02, size=len(dates)) * 100
rainfall_data = pd.DataFrame(data={'date': dates, 'rainfall': rainfall})
rainfall_data.set_index('date', inplace=True)


# In[46]:


# Check if the 'rainfall' column exists
if 'rainfall' not in rainfall_data.columns:
    raise KeyError("The 'rainfall' column is not present in the data.")


# In[47]:


# Normalize data
scaler = MinMaxScaler()
try:
    scaled_data = scaler.fit_transform(rainfall_data["rainfall"].values.reshape(-1, 1))
except KeyError as e:
    print(f"Error: {e}")
    scaled_data = None


# In[48]:


# Ensure scaled_data is defined
if scaled_data is None:
    raise ValueError("Scaled data is not defined. Please check the input data.")


# In[49]:


# Create sequences for LSTM
sequence_length = 10
sequences = []
for i in range(len(scaled_data) - sequence_length):
    sequences.append(scaled_data[i : i + sequence_length])

X = np.array(sequences)
y = scaled_data[sequence_length:]


# In[50]:


# Ensure the LSTM input shape is correct
X = X.reshape((X.shape[0], X.shape[1], 1))


# In[51]:


# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation="relu", input_shape=(sequence_length, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=10, batch_size=32)


# In[52]:


# Predict future rainfall
future_data = rainfall_data["rainfall"].values[-sequence_length:]
future_data = scaler.transform(future_data.reshape(-1, 1))
future_data = future_data.reshape(1, sequence_length, 1)
predicted_rainfall = model.predict(future_data)
predicted_rainfall = scaler.inverse_transform(predicted_rainfall)

print("Predicted Rainfall:", predicted_rainfall[0][0])


# In[53]:


# Visualization
plt.figure(figsize=(14, 7))
plt.plot(rainfall_data.index, rainfall_data["rainfall"], label='Actual Rainfall')
plt.axvline(x=rainfall_data.index[-1], color='r', linestyle='--', label='Prediction Point')
plt.scatter(rainfall_data.index[-1], predicted_rainfall[0][0], color='r', label='Predicted Rainfall', zorder=5)
plt.title('Rainfall Prediction')
plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()


# In[54]:


# Decompose the time series to understand trend, seasonality, and residual components
decomposition = seasonal_decompose(rainfall_data['rainfall'], model='additive', period=30)  # Assuming a monthly seasonality
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


# In[55]:


# Plot the decomposition
plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(rainfall_data.index, rainfall_data['rainfall'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(rainfall_data.index, trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(rainfall_data.index, seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(rainfall_data.index, residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[56]:


# Plot the historical data to visualize the trend
plt.figure(figsize=(12, 6))
plt.plot(dates, rainfall, label='Daily Rainfall')
plt.xlabel('Date')
plt.ylabel('Rainfall (inches)')
plt.title('Historical Daily Rainfall in Miami, Florida')
plt.legend()
plt.show()


# In[57]:


# Plot the decomposed components
plt.figure(figsize=(12, 6))
plt.subplot(411)
plt.plot(dates, rainfall, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(dates, trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(dates, seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(dates, residual, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[58]:


# Determine the order of the ARIMA model using ACF and PACF plots
plot_acf(rainfall, lags=30)
plt.show()


# In[59]:


plot_pacf(rainfall, lags=30)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




