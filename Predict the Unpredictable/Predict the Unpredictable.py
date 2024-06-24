import pandas as pd
from prophet import Prophet
from tqdm import tqdm

# Load data
historical_weather = pd.read_csv('historical_weather.csv')
submission_key = pd.read_csv('submission_key.csv')

# Preprocessing
historical_weather['date'] = pd.to_datetime(historical_weather['date'])
submission_key['date'] = pd.to_datetime(submission_key['date'])

# Initialize the final submission dataframe
final_submission = pd.DataFrame()

# Unique city IDs
city_ids = historical_weather['city_id'].unique()

# Iterate over each city
for city_id in tqdm(city_ids):
    # Filter data for the current city
    city_data = historical_weather[historical_weather['city_id'] == city_id]
    
    # Prepare data for Prophet
    prophet_data = city_data[['date', 'avg_temp_c']].rename(columns={'date': 'ds', 'avg_temp_c': 'y'})
    prophet_data = prophet_data.dropna(subset=['y'])
    
    # Initialize the Prophet model
    model = Prophet()
    
    # Fit the model
    model.fit(prophet_data)
    
    # Prepare the future dataframe
    future_dates = pd.DataFrame({'ds': pd.date_range(start='2019-01-01', end='2019-01-07')})
    
    # Predict future values
    forecast = model.predict(future_dates)
    
    # Extract relevant columns from the forecast
    predictions = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'avg_temp_c'})
    predictions['city_id'] = city_id
    
    # Merge with submission_key to get submission_IDs
    city_submission_key = submission_key[submission_key['city_id'] == city_id]
    city_submission = city_submission_key.merge(predictions, on=['date', 'city_id'], how='left')
    
    # Append to final submission
    final_submission = pd.concat([final_submission, city_submission], ignore_index=True)

# Create the final submission file
final_submission = final_submission[['submission_ID', 'avg_temp_c']]
final_submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
