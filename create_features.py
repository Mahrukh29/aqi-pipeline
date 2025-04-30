import pandas as pd
import numpy as np
import hopsworks
import logging
import requests
import os
from datetime import datetime, timedelta

# Set up logging to track progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API keys from GitHub Actions secrets
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Constants
FEATURE_GROUP_NAME = "pm10_features"
DATA_PATH = 'AQI Prediction Dataset.csv'

def fetch_new_data(existing_data):
    try:
        logging.info("Fetching new data from OpenWeather API...")
        lat, lon = 24.8607, 67.0011  # Karachi

        # Air pollution history (past 24 hours)
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=1)).timestamp())
        url_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={OPENWEATHER_API_KEY}"

        response_pollution = requests.get(url_pollution)
        response_pollution.raise_for_status()
        data_pollution = response_pollution.json()

        # Current weather data
        url_weather = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        response_weather = requests.get(url_weather)
        response_weather.raise_for_status()
        data_weather = response_weather.json()

        temperature = data_weather['main']['temp'] - 273.15
        humidity = data_weather['main']['humidity']
        wind_speed = data_weather['wind']['speed']
        precipitation = data_weather.get('rain', {}).get('1h', 0)

        records = []
        for entry in data_pollution['list']:
            dt = datetime.fromtimestamp(entry['dt'])
            components = entry['components']
            is_weekend = 1 if dt.weekday() >= 5 else 0
            had_rain = 1 if precipitation > 0 else 0
            heavy_rain = 1 if precipitation > 2.5 else 0
            wind_category = 'low' if wind_speed < 3 else 'medium' if wind_speed < 6 else 'high'
            wind_pm_impact = wind_speed * components.get('pm10', 0)
            wind_rain_interaction = wind_speed * precipitation

            records.append({
                'datetime': dt,
                'record_id': int(dt.strftime('%Y%m%d%H%M%S')),
                'pm10': components.get('pm10', None),
                'pm25': components.get('pm2_5', None),
                'no2': components.get('no2', None),
                'o3': components.get('o3', None),
                'aqi': entry['main']['aqi'],
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'precipitation': precipitation,
                'is_weekend': is_weekend,
                'had_rain': had_rain,
                'heavy_rain': heavy_rain,
                'wind_category': wind_category,
                'wind_pm_impact': wind_pm_impact,
                'wind_rain_interaction': wind_rain_interaction,
            })

        new_data = pd.DataFrame(records)
        logging.info(f"Fetched {len(new_data)} rows from OpenWeather API.")

        # Filter for new entries
        latest_existing_time = existing_data['datetime'].max()
        new_data = new_data[new_data['datetime'] > latest_existing_time]
        logging.info(f"{len(new_data)} new rows after removing overlaps.")

        # Append to CSV
        if not new_data.empty:
            existing_columns = existing_data.columns.tolist()
            new_data = new_data.reindex(columns=existing_columns, fill_value=None)
            new_data['datetime'] = new_data['datetime'].dt.strftime('%d/%m/%Y %H:%M')
            new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
            logging.info(f"Appended {len(new_data)} rows to {DATA_PATH}.")

        return new_data

    except Exception as e:
        logging.error(f"Error fetching OpenWeather data: {str(e)}")
        return pd.DataFrame()

def create_features():
    try:
        data = pd.read_csv(DATA_PATH)
        logging.info(f"Loaded {len(data)} rows from dataset.")

        # Parse datetime
        try:
            data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M', dayfirst=True, errors='coerce')
            mask = data['datetime'].isna()
            if mask.any():
                data.loc[mask, 'datetime'] = pd.to_datetime(data.loc[mask, 'datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        except:
            data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True, errors='coerce')

        data = data.dropna(subset=['datetime'])

        if data['record_id'].dtype == 'object':
            data['record_id'] = data['record_id'].str.replace('rec_', '', regex=False).astype('int64')
        else:
            data['record_id'] = data['record_id'].astype('int64')

        # Fetch new data
        new_data = fetch_new_data(data)
        if not new_data.empty:
            new_data['datetime'] = pd.to_datetime(new_data['datetime'], errors='coerce')
            data = pd.concat([data, new_data], ignore_index=True)

        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data = data.sort_values('datetime').reset_index(drop=True)

        # Drop unnecessary features
        data.drop(columns=[ 
            'is_weekend', 'had_rain', 'heavy_rain', 'wind_category', 
            'wind_pm_impact', 'wind_rain_interaction' 
        ], inplace=True, errors='ignore')

        # Clean values
        data = data[data['pm10'] != -9999]
        data = data[data['pm10'] <= 500]

        # Time-based features
        data['hour'] = data['datetime'].dt.hour.astype('int32')
        data['day'] = data['datetime'].dt.day.astype('int32')
        data['month'] = data['datetime'].dt.month.astype('int32')
        data['day_of_week'] = data['datetime'].dt.dayofweek.astype('int32')

        # Cyclic encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

        # Lag features
        data['lag_1_pm10'] = data['pm10'].shift(1).ffill()
        data['lag_1_pm25'] = data['pm25'].shift(1).ffill()
        data['lag_1_aqi'] = data['aqi'].shift(1).ffill()

        # Rolling stats
        data['rolling_pm10_mean'] = data['pm10'].rolling(3).mean().fillna(data['pm10'].mean())
        data['rolling_pm10_std'] = data['pm10'].rolling(3).std().fillna(data['pm10'].std())
        data['rolling_pm25_mean'] = data['pm25'].rolling(3).mean().fillna(data['pm25'].mean())
        data['rolling_pm25_std'] = data['pm25'].rolling(3).std().fillna(data['pm25'].std())

        # Change + ratio
        data['pm10_change'] = data['pm10'].diff().fillna(0)
        data['pm25_change'] = data['pm25'].diff().fillna(0)
        data['pm25_pm10_ratio'] = np.where(data['pm10'] != 0, data['pm25'] / data['pm10'], 0)

        # Environmental interaction features
        data['temperature'] = data['temperature'].fillna(data['temperature'].mean())
        data['humidity'] = data['humidity'].fillna(data['humidity'].mean()).round().astype('int64')  # Cast humidity to int64
        data['wind_speed'] = data['wind_speed'].fillna(data['wind_speed'].mean())
        data['precipitation'] = data['precipitation'].fillna(data['precipitation'].mean())
        data['dew_point'] = data['temperature'] - ((100 - data['humidity']) / 5)
        data['heat_index'] = data['temperature'] + 0.5 * data['humidity']
        data['temp_humidity'] = data['temperature'] * data['humidity']

        # Ensure correct dtypes
        data = data.dropna()
        data['aqi'] = data['aqi'].astype('int64')
        data['pm10'] = data['pm10'].astype('float64')
        data['pm25'] = data['pm25'].astype('float64')
        data['no2'] = data['no2'].astype('float64')
        data['o3'] = data['o3'].astype('float64')

        # Save to Hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        feature_store = project.get_feature_store()

        feature_group = feature_store.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=1,
            primary_key=['record_id'],
            description="Features for PM10 prediction",
            event_time='datetime'
        )

        feature_group.insert(data, write_options={"wait_for_job": True})
        logging.info(f"Inserted {len(data)} rows into feature group.")

        hopsworks.logout()

    except Exception as e:
        logging.error(f"Feature creation error: {str(e)}")
        hopsworks.logout()
        raise

if __name__ == "__main__":
    logging.info("Feature engineering started...")
    create_features()
    logging.info("Feature engineering completed.")
