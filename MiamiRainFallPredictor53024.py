#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[ ]:


# Load the data
data = pd.read_csv('miami_rainfall_data.csv')


# In[ ]:


# Preprocess the data
data['date'] = pd.to_datetime(data['date'])
data['rain'] = (data['rainfall'] > 0).astype(int)


# In[ ]:


# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.125, random_state=42)


# In[ ]:


# Train the logistic regression model
X_train = train_data[['temperature', 'humidity', 'wind_speed', 'pressure']]
y_train = train_data['rain']
model = LogisticRegression()
model.fit(X_train, y_train)


# In[ ]:


# Evaluate the model on the testing set
X_test = test_data[['temperature', 'humidity', 'wind_speed', 'pressure']]
y_test = test_data['rain']
y_pred = model.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[ ]:


print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')


# In[ ]:


# Use the model to make predictions
future_data = pd.DataFrame({
    'date': pd.date_range(start=data['date'].max() + pd.Timedelta(days=1), periods=7, freq='D'),
    'temperature': [85, 86, 87, 88, 89, 90, 91],
    'humidity': [70, 72, 74, 76, 78, 80, 82],
    'wind_speed': [10, 12, 14, 16, 18, 20, 22],
    'pressure': [1020, 1018, 1016, 1014, 1012, 1010, 1008]
})


# In[ ]:


future_rain_prob = model.predict_proba(future_data[['temperature', 'humidity', 'wind_speed', 'pressure']])[:, 1]
future_data['rain_probability'] = future_rain_prob


# In[ ]:


print(future_data)

