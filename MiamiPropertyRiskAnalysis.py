#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Data Collection
def collect_climate_data():
    # Implement data collection logic
    return pd.DataFrame()  # Return empty DataFrame for now

def collect_flood_zone_data():
    # Load flood zone data from FEMA or local sources
    return gpd.read_file('path_to_flood_zone_shapefile.shp')

def collect_elevation_data():
    # Load Digital Elevation Model (DEM) data
    return rasterio.open('path_to_dem_file.tif')

def collect_property_data():
    # Implement property data collection logic
    return gpd.GeoDataFrame()  # Return empty GeoDataFrame for now

# Data Preprocessing
def preprocess_data(climate_data, flood_zones, elevation, property_data):
    scaler = MinMaxScaler()
    climate_data_scaled = scaler.fit_transform(climate_data)
    property_elevations = extract_elevations(elevation, property_data)
    property_data_with_flood_zones = gpd.sjoin(property_data, flood_zones, how="left", predicate="within")
    combined_data = pd.concat([pd.DataFrame(climate_data_scaled), property_data_with_flood_zones, property_elevations], axis=1)
    return combined_data

def extract_elevations(dem, locations):
    elevations = []
    with rasterio.open(dem) as src:
        for idx, location in locations.iterrows():
            lon, lat = location.geometry.x, location.geometry.y
            row, col = src.index(lon, lat)
            elevation = src.read(1)[row, col]
            elevations.append(elevation)
    return pd.Series(elevations, name='elevation')

# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(input_shape, 1)),
        LSTM(100, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ARIMA Model
def build_arima_model(data):
    return ARIMA(data, order=(1, 1, 1))

# Prophet Model
def build_prophet_model(data):
    model = Prophet(seasonality_mode='multiplicative', 
                    yearly_seasonality=True, 
                    weekly_seasonality=True,
                    daily_seasonality=False)
    model.add_country_holidays(country_name='US')
    model.fit(data)
    return model

# TFT Model
def build_tft_model(data):
    max_prediction_length = 365
    max_encoder_length = 365 * 3
    
    training = TimeSeriesDataSet(
        data,
        time_idx="timestamp",
        target="target_variable",
        group_ids=["property_id"],
        static_categoricals=["flood_zone", "property_type"],
        static_reals=["elevation", "distance_from_coast"],
        time_varying_known_reals=["time_idx", "price_trend"],
        time_varying_unknown_reals=["climate_factor1", "climate_factor2"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4
    )
    return tft

# Risk Assessment
def calculate_risk_score(lstm_output, arima_output, prophet_output, tft_output, flood_zone, elevation):
    climate_risk = 0.3 * lstm_output + 0.2 * arima_output + 0.25 * prophet_output + 0.25 * tft_output
    flood_zone_factor = {'A': 1.5, 'AE': 1.3, 'X': 0.8}.get(flood_zone, 1.0)
    elevation_factor = max(0, 1 - (elevation / 10))
    final_risk_score = climate_risk * flood_zone_factor * elevation_factor
    return final_risk_score

# Visualization
def visualize_results(risk_scores, locations):
    gdf = gpd.GeoDataFrame(locations, geometry='geometry')
    gdf['risk_score'] = risk_scores

    fig, ax = plt.subplots(figsize=(15, 10))
    gdf.plot(column='risk_score', ax=ax, legend=True, cmap='RdYlGn_r', legend_kwds={'label': 'Risk Score'})
    ax.set_title('Property Investment Risk in Miami-Dade, Florida')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(gdf[['risk_score', 'elevation', 'flood_zone_factor']], annot=True, cmap='YlOrRd')
    plt.title('Heatmap of Risk Factors')
    plt.show()

# Main function
def main():
    climate_data = collect_climate_data()
    flood_zones = collect_flood_zone_data()
    elevation_data = collect_elevation_data()
    property_data = collect_property_data()
    
    processed_data = preprocess_data(climate_data, flood_zones, elevation_data, property_data)
    
    lstm_model = build_lstm_model(processed_data.shape[1])
    arima_model = build_arima_model(processed_data['target_variable'])
    prophet_model = build_prophet_model(processed_data[['ds', 'y']])
    tft_model = build_tft_model(processed_data)

    # Train and predict with each model
    lstm_output = lstm_model.predict(processed_data)
    arima_output = arima_model.fit().forecast(steps=len(processed_data))
    prophet_output = prophet_model.predict(processed_data[['ds']])['yhat']
    tft_output = tft_model.predict(processed_data)
    
    risk_scores = []
    for lstm, arima, prophet, tft, flood_zone, elevation in zip(lstm_output, arima_output, prophet_output, tft_output, processed_data['flood_zone'], processed_data['elevation']):
        risk_score = calculate_risk_score(lstm, arima, prophet, tft, flood_zone, elevation)
        risk_scores.append(risk_score)
    
    visualize_results(risk_scores, property_data)

if __name__ == "__main__":
    main()

