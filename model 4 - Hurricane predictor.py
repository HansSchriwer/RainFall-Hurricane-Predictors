# Model 4 – Hurricane Predictor
Real hurricane prediction is an extremely complex process that requires vast amounts of data, sophisticated models, and significant computational resources. It's typically performed by national weather services and research institutions using supercomputers and teams of expert meteorologists.
Having said that, I am providing a more advanced example that incorporates some additional techniques used in hurricane forecasting. This code is still a significant simplification and should not be used for actual forecasting or emergency planning, but it demonstrates some key concepts:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.interpolate import griddata

# Function to load historical hurricane data
def load_hurricane_data(file_path):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y%m%d %H%M')
    return df

# Function to load climate reanalysis data (e.g., from NCEP/NCAR Reanalysis)
def load_climate_data(file_path):
    nc = Dataset(file_path, 'r')
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]
    time = nc.variables['time'][:]
    sst = nc.variables['sst'][:, :, :]
    slp = nc.variables['slp'][:, :, :]
    uwnd = nc.variables['uwnd'][:, :, :]
    vwnd = nc.variables['vwnd'][:, :, :]
    rhum = nc.variables['rhum'][:, :, :]
    return lats, lons, time, sst, slp, uwnd, vwnd, rhum

# Function to calculate derived variables
def calculate_derived_variables(uwnd, vwnd):
    wind_speed = np.sqrt(uwnd**2 + vwnd**2)
    vorticity = np.gradient(vwnd, axis=1) - np.gradient(uwnd, axis=2)
    return wind_speed, vorticity

# Function to extract climate features for a given location and time
def extract_climate_features(lat, lon, date, lats, lons, time, sst, slp, uwnd, vwnd, rhum):
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    time_idx = np.argmin(np.abs(time - date.timestamp()))
    
    sst_value = sst[time_idx, lat_idx, lon_idx]
    slp_value = slp[time_idx, lat_idx, lon_idx]
    uwnd_value = uwnd[time_idx, lat_idx, lon_idx]
    vwnd_value = vwnd[time_idx, lat_idx, lon_idx]
    rhum_value = rhum[time_idx, lat_idx, lon_idx]
    
    wind_speed, vorticity = calculate_derived_variables(uwnd[time_idx], vwnd[time_idx])
    wind_speed_value = wind_speed[lat_idx, lon_idx]
    vorticity_value = vorticity[lat_idx, lon_idx]
    
    return sst_value, slp_value, uwnd_value, vwnd_value, rhum_value, wind_speed_value, vorticity_value

# Load data
hurricane_data = load_hurricane_data('historical_hurricanes.csv')
lats, lons, time, sst, slp, uwnd, vwnd, rhum = load_climate_data('climate_reanalysis.nc')

# Extract features for each hurricane observation
features = []
for _, row in hurricane_data.iterrows():
    climate_features = extract_climate_features(
        row['latitude'], row['longitude'], row['datetime'],
        lats, lons, time, sst, slp, uwnd, vwnd, rhum
    )
    features.append([row['latitude'], row['longitude']] + list(climate_features) + [row['max_wind'], row['min_pressure']])

columns = ['latitude', 'longitude', 'sst', 'slp', 'uwnd', 'vwnd', 'rhum', 'wind_speed', 'vorticity', 'max_wind', 'min_pressure']
features_df = pd.DataFrame(features, columns=columns)

# Prepare data for modeling
X = features_df.drop(['max_wind', 'min_pressure'], axis=1)
y_wind = features_df['max_wind']
y_pressure = features_df['min_pressure']

# Split data and scale features
X_train, X_test, y_wind_train, y_wind_test, y_pressure_train, y_pressure_test = train_test_split(
    X, y_wind, y_pressure, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
wind_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
wind_model.fit(X_train_scaled, y_wind_train)

pressure_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
pressure_model.fit(X_train_scaled, y_pressure_train)

# Evaluate models
wind_pred = wind_model.predict(X_test_scaled)
pressure_pred = pressure_model.predict(X_test_scaled)

wind_mse = mean_squared_error(y_wind_test, wind_pred)
wind_mae = mean_absolute_error(y_wind_test, wind_pred)
pressure_mse = mean_squared_error(y_pressure_test, pressure_pred)
pressure_mae = mean_absolute_error(y_pressure_test, pressure_pred)

print(f"Wind Speed Prediction - MSE: {wind_mse:.2f}, MAE: {wind_mae:.2f}")
print(f"Pressure Prediction - MSE: {pressure_mse:.2f}, MAE: {pressure_mae:.2f}")

# Function to predict hurricane trajectory using a simple steering flow approach
def predict_trajectory(lat, lon, uwnd, vwnd, time_steps):
    trajectory = [(lat, lon)]
    for _ in range(time_steps):
        u = griddata((lats, lons), uwnd, (lat, lon), method='linear')
        v = griddata((lats, lons), vwnd, (lat, lon), method='linear')
        lat += v * 3600 / 111000  # Approximate 1 degree latitude = 111 km
        lon += u * 3600 / (111000 * np.cos(np.radians(lat)))  # Adjust for latitude
        trajectory.append((lat, lon))
    return trajectory

# Function to forecast hurricane intensity and path
def forecast_hurricane(start_lat, start_lon, start_date, forecast_hours):
    forecast = []
    current_lat, current_lon = start_lat, start_lon
    current_date = pd.to_datetime(start_date)
    
    for hour in range(forecast_hours):
        climate_features = extract_climate_features(
            current_lat, current_lon, current_date,
            lats, lons, time, sst, slp, uwnd, vwnd, rhum
        )
        
        features = np.array([[current_lat, current_lon] + list(climate_features)])
        features_scaled = scaler.transform(features)
        
        predicted_wind = wind_model.predict(features_scaled)[0]
        predicted_pressure = pressure_model.predict(features_scaled)[0]
        
        # Predict next position
        trajectory = predict_trajectory(current_lat, current_lon, uwnd[-1], vwnd[-1], 1)
        current_lat, current_lon = trajectory[-1]
        
        current_date += timedelta(hours=1)
        
        forecast.append({
            'datetime': current_date,
            'latitude': current_lat,
            'longitude': current_lon,
            'predicted_wind': predicted_wind,
            'predicted_pressure': predicted_pressure
        })
    
    return pd.DataFrame(forecast)

# Example usage
start_lat, start_lon = 25.0, -75.0
start_date = '2024-08-01 00:00:00'
forecast_hours = 120  # 5-day forecast

hurricane_forecast = forecast_hurricane(start_lat, start_lon, start_date, forecast_hours)

# Visualize forecast
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)

scatter = ax.scatter(
    hurricane_forecast['longitude'], hurricane_forecast['latitude'],
    c=hurricane_forecast['predicted_wind'], cmap='viridis',
    transform=ccrs.PlateCarree()
)
plt.colorbar(scatter, label='Predicted Wind Speed (knots)')

ax.set_extent([-100, -60, 20, 50])
plt.title('Predicted Hurricane Path and Intensity')
plt.show()

print(hurricane_forecast)

# This code incorporates several additional concepts used in hurricane forecasting:
1.	It uses more climate variables, including sea level pressure and relative humidity.
2.	It calculates derived variables like wind speed and vorticity.
3.	It implements a simple steering flow approach for trajectory prediction.
4.	The forecast is done hour-by-hour, allowing for more detailed predictions.

# However, this code still has significant limitations:
•	It doesn't account for complex atmospheric dynamics that influence hurricane development and movement.
•	It doesn't incorporate data from multiple models or ensemble forecasting techniques.
•	It lacks many important data sources used in real hurricane forecasting, such as satellite imagery, radar data, and dropsondes from hurricane hunter aircraft.
•	The trajectory prediction is overly simplistic and doesn't account for factors like the beta effect or interactions with other weather systems.
Real hurricane forecasting systems used by national weather services are far more complex, incorporating multiple numerical weather prediction models, statistical post-processing, and expert human judgment. They also use much more data and vastly more computational resources than this example.
For actual hurricane forecasting, always rely on official sources like the National Hurricane Center (for Atlantic hurricanes) or other national meteorological agencies. This code is for educational purposes only and should not be used for real-world forecasting or decision-making.
