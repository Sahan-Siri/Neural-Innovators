
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Load the Data
data = pd.read_csv('daily_data.csv')

# Step 2: Handle Missing Values
# Identify numerical and categorical columns
numerical_cols = ['temperature_celsius', 'wind_kph', 'wind_degree', 'pressure_mb', 'precip_mm', 'humidity', 'cloud',
                  'feels_like_celsius', 'visibility_km', 'uv_index', 'gust_kph', 'air_quality_us-epa-index']
categorical_cols = ['condition_text', 'sunrise', 'sunset']

# Fill missing numerical values with mean
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# Assuming your time is in the format 'HH:MM AM/PM'
data['sunrise'] = pd.to_datetime(data['sunrise'], format='%I:%M %p').dt.hour + pd.to_datetime(data['sunrise'], format='%I:%M %p').dt.minute / 60
data['sunset'] = pd.to_datetime(data['sunset'], format='%I:%M %p').dt.hour + pd.to_datetime(data['sunset'], format='%I:%M %p').dt.minute / 60

# Encode condition_text column
label_encoder = LabelEncoder()
data['condition_text_encoded'] = label_encoder.fit_transform(data['condition_text'].astype(str))

# Step 3: Split the Data
# Select relevant features
features = numerical_cols + ['sunrise', 'sunset']
X = data[data['condition_text'].notna()][features]
y = data[data['condition_text'].notna()]['condition_text_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection and Training
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Prediction and Evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Step 6: Predict Missing Values
missing_data = data[data['condition_text'].isna()]
X_missing = missing_data[features]
missing_data['condition_text_pred'] = label_encoder.inverse_transform(rf_model.predict(X_missing))

# Step 7: Prepare Submission File
submission = pd.read_csv('submission.csv')
submission = submission.merge(missing_data[['day_id', 'condition_text_pred']], on='day_id', how='left')
submission['condition_text'].fillna(submission['condition_text_pred'], inplace=True)
submission.drop(columns=['condition_text_pred'], inplace=True)
submission.to_csv('submission.csv', index=False)
