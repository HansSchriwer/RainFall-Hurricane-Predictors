# Please note that real-world hurricane prediction is extremely complex and typically requires supercomputers and vast amounts of data. This code is a simplified version for educational purposes and should not be used for actual hurricane prediction or emergency planning.

# Code: Model 3 – Hurricane predictor

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Function to load and preprocess hurricane data
def load_hurricane_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Function to load and preprocess climate data (assuming NetCDF format)
def load_climate_data(file_path):
    nc = Dataset(file_path, 'r')
    lats = nc.variables['latitude'][:]
    lons = nc.variables['longitude'][:]
    time = nc.variables['time'][:]
    sst = nc.variables['sst'][:, :, :]  # Sea Surface Temperature
    wind_u = nc.variables['u10'][:, :, :]  # U component of wind
    wind_v = nc.variables['v10'][:, :, :]  # V component of wind
    humidity = nc.variables['r'][:, :, :]  # Relative humidity
    return lats, lons, time, sst, wind_u, wind_v, humidity

# Function to extract climate features for a given hurricane location and time
def extract_climate_features(lat, lon, date, lats, lons, time, sst, wind_u, wind_v, humidity):
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    time_idx = np.argmin(np.abs(time - date.timestamp()))
    
    sst_value = sst[time_idx, lat_idx, lon_idx]
    wind_speed = np.sqrt(wind_u[time_idx, lat_idx, lon_idx]**2 + wind_v[time_idx, lat_idx, lon_idx]**2)
    humidity_value = humidity[time_idx, lat_idx, lon_idx]
    
    return sst_value, wind_speed, humidity_value

# Load and preprocess data
hurricane_data = load_hurricane_data('hurricane_data.csv')
lats, lons, time, sst, wind_u, wind_v, humidity = load_climate_data('climate_data.nc')

# Extract features for each hurricane observation
features = []
for _, row in hurricane_data.iterrows():
    sst_value, wind_speed, humidity_value = extract_climate_features(
        row['Latitude'], row['Longitude'], row['Date'],
        lats, lons, time, sst, wind_u, wind_v, humidity
    )
    features.append([
        row['Latitude'], row['Longitude'],
        row['Max Wind Speed'], row['Minimum Pressure'],
        sst_value, wind_speed, humidity_value
    ])

features_df = pd.DataFrame(features, columns=[
    'Latitude', 'Longitude', 'Max Wind Speed', 'Minimum Pressure',
    'Sea Surface Temperature', 'Environmental Wind Speed', 'Relative Humidity'
])

# Prepare data for modeling
X = features_df.drop(['Max Wind Speed', 'Minimum Pressure'], axis=1)
y_wind = features_df['Max Wind Speed']
y_pressure = features_df['Minimum Pressure']

# Split data into training and testing sets
X_train, X_test, y_wind_train, y_wind_test, y_pressure_train, y_pressure_test = train_test_split(
    X, y_wind, y_pressure, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost models for wind speed and pressure prediction
wind_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
wind_model.fit(X_train_scaled, y_wind_train)

pressure_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
pressure_model.fit(X_train_scaled, y_pressure_train)

# Make predictions
wind_pred = wind_model.predict(X_test_scaled)
pressure_pred = pressure_model.predict(X_test_scaled)

# Evaluate models
wind_mse = mean_squared_error(y_wind_test, wind_pred)
wind_mae = mean_absolute_error(y_wind_test, wind_pred)
pressure_mse = mean_squared_error(y_pressure_test, pressure_pred)
pressure_mae = mean_absolute_error(y_pressure_test, pressure_pred)

print(f"Wind Speed Prediction - MSE: {wind_mse:.2f}, MAE: {wind_mae:.2f}")
print(f"Pressure Prediction - MSE: {pressure_mse:.2f}, MAE: {pressure_mae:.2f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(wind_model)
plt.title('Feature Importance for Wind Speed Prediction')
plt.show()

plt.figure(figsize=(10, 6))
xgb.plot_importance(pressure_model)
plt.title('Feature Importance for Pressure Prediction')
plt.show()

# Function to predict hurricane intensity and trajectory
def predict_hurricane(start_lat, start_lon, start_date, num_days):
    predictions = []
    current_lat, current_lon = start_lat, start_lon
    current_date = pd.to_datetime(start_date)
    
    for _ in range(num_days):
        sst_value, wind_speed, humidity_value = extract_climate_features(
            current_lat, current_lon, current_date,
            lats, lons, time, sst, wind_u, wind_v, humidity
        )
        
        features = np.array([[
            current_lat, current_lon, sst_value, wind_speed, humidity_value
        ]])
        features_scaled = scaler.transform(features)
        
        predicted_wind = wind_model.predict(features_scaled)[0]
        predicted_pressure = pressure_model.predict(features_scaled)[0]
        
        # Simple trajectory prediction (this should be much more complex in reality)
        current_lat += np.random.normal(0, 0.5)  # Random movement
        current_lon += np.random.normal(0, 0.5)  # Random movement
        current_date += pd.Timedelta(days=1)
        
        predictions.append({
            'Date': current_date,
            'Latitude': current_lat,
            'Longitude': current_lon,
            'Predicted Wind Speed': predicted_wind,
            'Predicted Pressure': predicted_pressure
        })
    
    return pd.DataFrame(predictions)

# Example usage
start_lat, start_lon = 25.0, -75.0  # Example starting position
start_date = '2024-08-01'  # Example starting date
num_days = 7  # Predict for 7 days

predicted_path = predict_hurricane(start_lat, start_lon, start_date, num_days)
print(predicted_path)

# Visualize predicted path
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)

scatter = ax.scatter(
    predicted_path['Longitude'], predicted_path['Latitude'],
    c=predicted_path['Predicted Wind Speed'], cmap='viridis',
    transform=ccrs.PlateCarree()
)
plt.colorbar(scatter, label='Predicted Wind Speed (knots)')

ax.set_extent([-100, -60, 20, 50])
plt.title('Predicted Hurricane Path and Intensity')
plt.show()

# This code does the following:
1.	Imports necessary libraries for data manipulation, machine learning, and visualization.
2.	Defines functions to load and preprocess hurricane and climate data.
3.	Extracts relevant features from the climate data for each hurricane observation.
4.	Prepares the data for modeling, including splitting into training and testing sets and scaling features.
5.	Trains XGBoost models to predict wind speed and pressure.
6.	Evaluates the models using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
7.	Plots feature importance for both wind speed and pressure prediction models.
8.	Defines a function to predict hurricane intensity and trajectory over multiple days.
9.	Visualizes the predicted hurricane path and intensity on a map.
Key aspects and potential improvements:
•	This model uses XGBoost, which can capture non-linear relationships in the data.
•	It incorporates various environmental factors like sea surface temperature, wind speed, and humidity.
•	The trajectory prediction is overly simplified and should be replaced with a more sophisticated model in a real application.
•	Additional features could be incorporated, such as upper-level winds, vertical wind shear, and atmospheric stability indices.
•	The model could be improved by using ensemble methods, incorporating data from multiple climate models, and using more advanced time series techniques.
•	Real-world hurricane prediction would require much more data, including satellite imagery and data from hurricane hunter aircraft.
  
# Remember, hurricane prediction is an extremely complex task that requires extensive data, powerful computing resources, and expert knowledge. This code is a simplified example and should not be used for actual hurricane forecasting or emergency planning.
