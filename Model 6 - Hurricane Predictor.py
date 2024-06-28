# Let's integrate LightGBM alongside XGBoost and LSTM, and combine these models using a stacking approach. 
We'll also perform hyperparameter tuning for both XGBoost and LightGBM to optimize their performance.

# Step 1: Data Preparation
First, let's ensure we have the necessary data preprocessing steps.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load your dataset
data = pd.read_csv('hurricane_data.csv')

# Feature engineering: create interaction features
data['temp_wind'] = data['weather_temp'] * data['wind_speed']
data['temp_humidity'] = data['weather_temp'] * data['humidity']
data['wind_humidity'] = data['wind_speed'] * data['humidity']

# Create lag features for time-series data
for lag in range(1, 4):
    data[f'weather_temp_lag_{lag}'] = data['weather_temp'].shift(lag)
    data[f'wind_speed_lag_{lag}'] = data['wind_speed'].shift(lag)
    data[f'water_temp_lag_{lag}'] = data['water_temp'].shift(lag)
    data[f'humidity_lag_{lag}'] = data['humidity'].shift(lag)
    data[f'pressure_lag_{lag}'] = data['pressure'].shift(lag)

# Drop rows with NaN values created by lag features
data = data.dropna()

# Feature and target variables
features = ['weather_temp', 'wind_speed', 'water_temp', 'humidity', 'pressure', 'storm_duration',
            'temp_wind', 'temp_humidity', 'wind_humidity',
            'weather_temp_lag_1', 'wind_speed_lag_1', 'water_temp_lag_1', 'humidity_lag_1', 'pressure_lag_1',
            'weather_temp_lag_2', 'wind_speed_lag_2', 'water_temp_lag_2', 'humidity_lag_2', 'pressure_lag_2',
            'weather_temp_lag_3', 'wind_speed_lag_3', 'water_temp_lag_3', 'humidity_lag_3', 'pressure_lag_3',
            'sea_level_pressure', 'wind_gust', 'storm_size']
target = 'hurricane_category'

X = data[features]
y = data[target]

# Encode categorical variables if necessary (Assuming target is categorical)
y = pd.get_dummies(y).values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Hyperparameter Tuning for XGBoost and LightGBM
Perform hyperparameter tuning for both models.

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_model = XGBClassifier(random_state=42)
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)
best_params_xgb = grid_search_xgb.best_params_
print(f"Best parameters for XGBoost: {best_params_xgb}")

# Train XGBoost with best parameters
xgb_model_best = XGBClassifier(**best_params_xgb)
xgb_model_best.fit(X_train, y_train)

# Hyperparameter tuning for LightGBM
param_grid_lgbm = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

lgbm_model = LGBMClassifier(random_state=42)
grid_search_lgbm = GridSearchCV(estimator=lgbm_model, param_grid=param_grid_lgbm, cv=3, n_jobs=-1, verbose=2)
grid_search_lgbm.fit(X_train, y_train)
best_params_lgbm = grid_search_lgbm.best_params_
print(f"Best parameters for LightGBM: {best_params_lgbm}")

# Train LightGBM with best parameters
lgbm_model_best = LGBMClassifier(**best_params_lgbm)
lgbm_model_best.fit(X_train, y_train)

# Predictions for both models
y_pred_xgb = xgb_model_best.predict(X_test)
y_pred_lgbm = lgbm_model_best.predict(X_test)

# Evaluate both models
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
print(f"XGBoost Classification Report:\n{classification_report(y_test, y_pred_xgb)}")
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgbm)}")
print(f"LightGBM Classification Report:\n{classification_report(y_test, y_pred_lgbm)}")

# Step 3: Advanced LSTM Model
Build a more advanced LSTM model with additional layers, dropout, and batch normalization.

# Reshape data for LSTM [samples, time steps, features]
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the advanced LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[1])))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))
lstm_model.add(LSTM(50, activation='relu', return_sequences=True))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))
lstm_model.add(LSTM(50, activation='relu'))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(100, activation='relu'))
lstm_model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
lstm_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for learning rate reduction and early stopping
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, restore_best_weights=True)

# Train the model
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=2, callbacks=[lr_reduction, early_stopping])

# Predictions
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
y_test_lstm = np.argmax(y_test, axis=1)

# Evaluate the model
print(f"LSTM Accuracy: {accuracy_score(y_test_lstm, y_pred_lstm)}")
print(f"LSTM Classification Report:\n{classification_report(y_test_lstm, y_pred_lstm)}")

# Step 4: Ensemble Model
Combine the predictions from XGBoost, LightGBM, and LSTM models using a stacking approach.

from sklearn.ensemble import StackingClassifier

# Custom classifier to combine LSTM predictions
class LSTMWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    def predict(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        return np.argmax(self.model.predict(X), axis=1)

# Instantiate the LSTM wrapper
lstm_wrapper = LSTMWrapper(lstm_model)

# Combine the models using Stacking
estimators = [
    ('xgb', xgb_model_best),
    ('lgbm', lgbm_model_best),
    ('lstm', lstm_wrapper)
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=XGBClassifier(random_state=42), cv=3, n_jobs=-1)
stacking_model.fit(X_train, y_train)

# Predictions
y_pred_stacking = stacking_model.predict(X_test)

# Evaluate the model
print(f"Stacking Model Accuracy: {accuracy_score(y_test, y_pred_stacking)}")
print(f"Stacking Model Classification Report:\n{classification_report(y_test, y_pred_stacking)}")

# Final Thoughts
This final version of the hurricane prediction model integrates XGBoost, LightGBM, and an advanced LSTM model, combining their strengths using a stacking approach. 
Hyperparameter tuning is performed for both XGBoost and LightGBM to optimize their performance. This ensemble model provides a robust solution for predicting hurricanes, leveraging multiple advanced techniques and models.


