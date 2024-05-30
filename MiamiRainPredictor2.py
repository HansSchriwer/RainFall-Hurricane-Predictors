#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


# In[ ]:


# Load the historical daily rainfall data for Miami, Florida
data = pd.read_csv("daily_rainfall_miami.csv")  # Assuming you have a CSV file with the historical data
dates = pd.to_datetime(data['date'])
rainfall = data['rainfall']


# In[ ]:


# Plot the historical data to visualize the trend
plt.figure(figsize=(12, 6))
plt.plot(dates, rainfall, label='Daily Rainfall')
plt.xlabel('Date')
plt.ylabel('Rainfall (inches)')
plt.title('Historical Daily Rainfall in Miami, Florida')
plt.legend()
plt.show()


# In[ ]:


# Check for stationarity in the data
# You may need to apply transformations, such as taking the first difference, to achieve stationarity


# In[ ]:


# Decompose the time series to understand trend, seasonality, and residual components
decomposition = seasonal_decompose(rainfall, period=365)  # Assuming a yearly seasonality
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


# In[ ]:


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


# In[ ]:


# Determine the order of the ARIMA model using ACF and PACF plots
plot_acf(rainfall, lags=30)
plt.show()


# In[ ]:


plot_pacf(rainfall, lags=30)
plt.show()


# In[ ]:


# Fit an ARIMA model to the data
model = ARIMA(rainfall, order=(p, d, q))  # Specify appropriate values for p, d, and q based on the ACF and PACF plots
model_fit = model.fit()


# In[ ]:


# Forecast future rainfall values
forecast_steps = 365  # Forecasting for the next year
forecast = model_fit.forecast(steps=forecast_steps)


# In[ ]:


# Print the forecasted rainfall values
print(forecast)

