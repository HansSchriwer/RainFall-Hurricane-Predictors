# Flood Zone predictor.
# 1. Configure file ('config.json')
{
  "noaa_api_key": "your_noaa_api_key",
  "zillow_api_key": "your_zillow_api_key",
  "flood_zone_shapefile_path": "path/to/flood_zone.shp",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "location": "CITY:US390029"
}
# 2. Python Script implementation
import requests
import json
import pandas as pd
import geopandas as gpd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def load_config(config_path='config.json'):
    """
    Load configuration from a JSON file.
    """
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def collect_climate_data(api_key, start_date, end_date, location):
    """
    Collect climate data from NOAA API.
    """
    url = f"https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&startdate={start_date}&enddate={end_date}&locationid={location}&units=metric&limit=1000"
    
    headers = {
        'token': api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        climate_df = pd.json_normalize(data['results'])
        logging.info(f"Climate data collected successfully for location: {location}")
        return climate_df
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch climate data: {e}")
        return None

def collect_flood_zone_data(shapefile_path):
    """
    Load flood zone data from a local shapefile or FEMA API.
    """
    try:
        flood_zones = gpd.read_file(shapefile_path)
        logging.info("Flood zone data loaded successfully.")
        return flood_zones
    except Exception as e:
        logging.error(f"Error loading flood zone data: {e}")
        return None

def collect_elevation_data(lat, lon):
    """
    Collect elevation data from the USGS Elevation API.
    """
    url = f"https://nationalmap.gov/epqs/pqs.php?x={lon}&y={lat}&units=Meters&output=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        elevation_data = response.json()
        elevation = elevation_data['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
        logging.info(f"Elevation data collected successfully for coordinates: ({lat}, {lon})")
        return elevation
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch elevation data: {e}")
        return None

def collect_property_data(api_key, location):
    """
    Collect property data from Zillow or another property data API.
    """
    url = f"https://api.zillow.com/v1/property?location={location}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        property_data = response.json()
        property_df = pd.json_normalize(property_data)
        logging.info(f"Property data collected successfully for location: {location}")
        return property_df
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch property data: {e}")
        return None

def preprocess_data(climate_data, flood_zones, elevation_data, property_data):
    """
    Preprocess the collected data by combining and normalizing it.
    """
    # Placeholder for actual data preprocessing steps
    logging.info("Preprocessing data...")
    # Combine datasets, handle missing values, normalize, etc.
    # You can implement specific preprocessing based on your requirements
    pass

def main():
    config = load_config()

    # Collect data
    climate_data = collect_climate_data(config['noaa_api_key'], config['start_date'], config['end_date'], config['location'])
    flood_zones = collect_flood_zone_data(config['flood_zone_shapefile_path'])
    elevation = collect_elevation_data(25.7617, -80.1918)  # Example coordinates for Miami, FL
    property_data = collect_property_data(config['zillow_api_key'], config['location'])
    
    if climate_data is not None and flood_zones is not None and elevation is not None and property_data is not None:
        # Preprocess the data
        preprocess_data(climate_data, flood_zones, elevation, property_data)
        # Further processing and model training can be done here
        logging.info("Data collection and preprocessing completed successfully.")
    else:
        logging.error("Data collection failed for one or more datasets.")

if __name__ == "__main__":
    main()

### Key Features of the Final Code:
# API Integration: The code integrates the NOAA Climate Data API, FEMA shapefiles, USGS Elevation API, and Zillow Property Data API. Each function retrieves data and converts it into a usable format.
Error Handling and Logging: The code uses try-except blocks to handle errors and logs both successful operations and errors.
Configuration File: The code reads API keys, file paths, and parameters from a config.json file, making it more flexible.
Preprocessing Placeholder: A placeholder function preprocess_data is included, which you can fill in with specific data processing steps, such as merging datasets and normalizing values.
Logging: The code logs activities to app.log, which helps with debugging and tracking execution.###

# Then next steps are:
# API Keys: Obtain and input your actual API keys in the config.json file.
# Testing: Run the code and test it with real data.
# Extend the Preprocessing: Add specific preprocessing steps based on your data requirements.
# Model Training: Implement your model training steps after preprocessing the data.

### Steps to Obtain and Use API Keys:
# Visit the API Provider's Website:
NOAA: NOAA Climate Data API
Zillow: Zillow API
USGS: No API key required for the Elevation API.
FEMA: No API key required for downloading flood zone shapefiles.
Sign Up:
Register for an account with each provider if required.
Request API Key:
Once signed up, request an API key, which will be provided in your account dashboard.
Add Keys to config.json:
Copy the API keys into your config.json file.###

# Then Preprocess Data and do the model training
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import logging

def preprocess_data(climate_data, flood_zones, elevation_data, property_data):
    """
    Preprocess the collected data by combining and normalizing it.
    """
    logging.info("Starting data preprocessing...")

    # Example: Merging climate data with property data based on location and date
    merged_data = pd.merge(property_data, climate_data, left_on='location', right_on='location')

    # Example: Adding elevation data to the merged dataset
    merged_data['elevation'] = elevation_data

    # Example: Merging flood zone data (this assumes flood_zones is a GeoDataFrame)
    # We need to spatially join flood zones with property locations
    property_gdf = gpd.GeoDataFrame(property_data, geometry=gpd.points_from_xy(property_data.lon, property_data.lat))
    merged_data = gpd.sjoin(property_gdf, flood_zones, how="left", op="within")

    # Handle missing values
    merged_data.fillna(method='ffill', inplace=True)

    # Scale features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_data.drop(columns=['target_variable']))  # Exclude target variable
    target = merged_data['target_variable'].values  # Assuming target variable exists in your data

    logging.info("Data preprocessing completed.")
    return scaled_data, target, scaler

def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))  # Output layer with a single neuron for regression

    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("LSTM model built successfully.")
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the LSTM model on the preprocessed data.
    """
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
    logging.info("Model training completed.")
    return history

def prepare_data_for_lstm(scaled_data, target, time_steps=60):
    """
    Prepare data for LSTM model by creating sequences of the time_steps length.
    """
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(target[i])

    X, y = np.array(X), np.array(y)
    return X, y

def main():
    config = load_config()

    # Collect data
    climate_data = collect_climate_data(config['noaa_api_key'], config['start_date'], config['end_date'], config['location'])
    flood_zones = collect_flood_zone_data(config['flood_zone_shapefile_path'])
    elevation = collect_elevation_data(25.7617, -80.1918)  # Example coordinates for Miami, FL
    property_data = collect_property_data(config['zillow_api_key'], config['location'])
    
    if climate_data is not None and flood_zones is not None and elevation is not None and property_data is not None:
        # Preprocess the data
        scaled_data, target, scaler = preprocess_data(climate_data, flood_zones, elevation, property_data)

        # Prepare data for LSTM
        X, y = prepare_data_for_lstm(scaled_data, target)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the LSTM model
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Train the model
        history = train_model(model, X_train, y_train, X_val, y_val)

        # Save the trained model
        model.save("lstm_model.h5")
        logging.info("Trained LSTM model saved as 'lstm_model.h5'.")
    else:
        logging.error("Data collection failed for one or more datasets.")

if __name__ == "__main__":
    main()

### Explanation of the Code:
# Preprocessing Data:
Merging Datasets: The example merges property data with climate data and adds elevation data as a new feature. It also performs a spatial join between property locations and flood zones.
Handling Missing Values: Missing data is handled using forward fill.
Feature Scaling: The features are scaled using MinMaxScaler to normalize the data, which is important for LSTM models.
Building and Training the LSTM Model:###

###LSTM Model: The LSTM model has two LSTM layers followed by a Dense layer for regression output. You can adjust the number of layers and neurons based on your data and use case.
Training: The model is trained using the training data and validated with a validation set. The training history is stored for analysis.
Model Saving: After training, the model is saved as lstm_model.h5.
Preparing Data for LSTM:###

### The function prepare_data_for_lstm creates sequences of the specified length (time_steps) to be used as input for the LSTM model. This step is crucial for time-series forecasting.
Splitting Data:
#The data is split into training and validation sets using train_test_split from scikit-learn.###
# Next Steps:
# Hyperparameter Tuning: Experiment with different LSTM architectures, time steps, batch sizes, and learning rates to find the optimal configuration for your data.
# Model Evaluation: Evaluate the trained model using metrics like Mean Squared Error (MSE) on the validation set.
# Visualization: Visualize the training history and predictions against actual values to assess model performance.
