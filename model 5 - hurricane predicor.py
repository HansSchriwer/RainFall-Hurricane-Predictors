import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load your dataset
data = pd.read_csv('hurricane_data.csv')

# Display the first few rows of the dataset
print(data.head())

#Step 2: Data Preprocessing
Next, we'll handle missing values, encode categorical variables if any, and normalize the features.

# Handle missing values
data = data.dropna()

# Feature and target variables
features = ['weather_temp', 'wind_speed', 'water_temp', 'humidity', 'pressure', 'storm_duration']
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

# Step 3: Random Forest Classifier
We will first build a Random Forest classifier for hurricane prediction.

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
Step 4: LSTM Model
For the LSTM model, we need to reshape the data to be in the format expected by LSTM layers.

# Reshape data for LSTM [samples, time steps, features]
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(100, activation='relu', input_shape=(1, X_train.shape[1])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=2)

# Predictions
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
y_test_lstm = np.argmax(y_test, axis=1)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test_lstm, y_pred_lstm)}")
print(f"Classification Report:\n{classification_report(y_test_lstm, y_pred_lstm)}")
Step 5: Combining Models (Optional)
You can also combine the predictions from both models to improve accuracy using an ensemble method.
python
Copy code
from sklearn.ensemble import VotingClassifier

# Combine the models
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('lstm', lstm_model)], voting='soft')

# Train the combined model
voting_model.fit(X_train, y_train)

# Predictions
y_pred_voting = voting_model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred_voting)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_voting)}")
# Final Thought
# This code provides a comprehensive approach to building a hurricane prediction model. Ensure that you have a good dataset with the necessary features, and feel free to tweak the hyperparameters for better performance. 
# This model can be further improved by incorporating more sophisticated techniques and additional relevant features.
