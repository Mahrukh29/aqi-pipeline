import pandas as pd
import numpy as np
import hopsworks
import json
import logging
import requests
from datetime import datetime, timedelta

# Set up logging to track progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths and settings
HOPSWORKS_API_KEY_PATH = r"C:\Users\MAHRUKH BAIG\OneDrive\Desktop\AQI Prediction\secrets.json"
DATA_PATH = r"C:\Users\MAHRUKH BAIG\OneDrive\Desktop\AQI Prediction\AQI Prediction Dataset.csv"
FEATURE_GROUP_NAME = "pm10_features"

# Load API keys from secrets.json
with open(HOPSWORKS_API_KEY_PATH, 'r') as f:
    secrets = json.load(f)
HOPSWORKS_API_KEY = secrets["api_key"]
OPENWEATHER_API_KEY = secrets["openweather_api_key"]

def fetch_new_data(existing_data):
    """
    Fetch new air pollution and weather data from OpenWeather API for Karachi.
    """
    try:
        logging.info("Fetching new data from OpenWeather API...")
        # Coordinates for Karachi
        lat, lon = 24.8607, 67.0011
        # Fetch air pollution data
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=1)).timestamp())
        url_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={OPENWEATHER_API_KEY}"

        response_pollution = requests.get(url_pollution)
        response_pollution.raise_for_status()
        data_pollution = response_pollution.json()

        # Fetch current weather data (approximation for historical data)
        url_weather = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        response_weather = requests.get(url_weather)
        response_weather.raise_for_status()
        data_weather = response_weather.json()

        # Extract weather data
        temperature = data_weather['main']['temp'] - 273.15  # Convert Kelvin to Celsius
        humidity = data_weather['main']['humidity']
        wind_speed = data_weather['wind']['speed']
        precipitation = data_weather.get('rain', {}).get('1h', 0)  # Rainfall in the last hour (mm)

        # Process the API response into a DataFrame
        records = []
        for entry in data_pollution['list']:
            dt = datetime.fromtimestamp(entry['dt'])
            components = entry['components']
            # Compute derived features
            is_weekend = 1 if dt.weekday() >= 5 else 0
            had_rain = 1 if precipitation > 0 else 0
            heavy_rain = 1 if precipitation > 2.5 else 0
            wind_category = 'low' if wind_speed < 3 else 'medium' if wind_speed < 6 else 'high'
            wind_pm_impact = wind_speed * components.get('pm10', 0)
            wind_rain_interaction = wind_speed * precipitation

            record = {
                'datetime': dt,
                'record_id': int(dt.strftime('%Y%m%d%H%M%S')),  # Integer format
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
            }
            records.append(record)
        new_data = pd.DataFrame(records)
        logging.info(f"Fetched {len(new_data)} rows of new data from OpenWeather API.")

        # Ensure new data doesn't overlap with existing data
        latest_existing_time = existing_data['datetime'].max()
        new_data = new_data[new_data['datetime'] > latest_existing_time]
        logging.info(f"After filtering overlap, {len(new_data)} rows remain.")

        # Append the new data to the CSV file
        if not new_data.empty:
            # Ensure the new data has the same columns as the existing CSV
            existing_columns = existing_data.columns.tolist()
            new_data = new_data.reindex(columns=existing_columns, fill_value=None)
            # Format datetime as DD/MM/YYYY HH:MM for consistency
            new_data['datetime'] = new_data['datetime'].dt.strftime('%d/%m/%Y %H:%M')
            # Append to CSV (mode='a' for append, header=False to avoid duplicating header)
            new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
            logging.info(f"Appended {len(new_data)} rows to {DATA_PATH}.")

        return new_data

    except Exception as e:
        logging.error(f"Error fetching data from OpenWeather API: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def create_features():
    try:
        # Step 1: Load the dataset
        data = pd.read_csv(DATA_PATH)
        logging.info(f"Loaded dataset with {len(data)} rows.")

        # Log sample datetime values to debug format
        logging.info(f"Sample datetime values before parsing:\n{data['datetime'].head().to_list()}")

        # Try parsing datetime with multiple formats, prioritizing DD/MM/YYYY
        try:
            # First try DD/MM/YYYY HH:MM with dayfirst=True
            data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M', dayfirst=True, errors='coerce')
            # Fill any NaT values by trying YYYY-MM-DD HH:MM:SS
            mask = data['datetime'].isna()
            if mask.any():
                data.loc[mask, 'datetime'] = pd.to_datetime(data.loc[mask, 'datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        except Exception as e:
            logging.warning(f"Datetime parsing failed: {str(e)}")
            # Fallback to automatic parsing with dayfirst=True
            data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True, errors='coerce')
            if data['datetime'].isna().all():
                logging.error("Failed to parse datetime with any known format.")
                raise ValueError("Failed to parse datetime column with any known format.")

        logging.info(f"Datetime type after parsing: {data['datetime'].dtype}")

        # Check for invalid datetime values and drop them
        invalid_datetime_rows = data['datetime'].isna().sum()
        if invalid_datetime_rows > 0:
            logging.info(f"Found {invalid_datetime_rows} rows with invalid datetime values. Dropping them.")
            data = data.dropna(subset=['datetime'])
            logging.info(f"Rows after dropping invalid datetime values: {len(data)}")
        else:
            logging.info("No invalid datetime values found.")

        # Convert record_id to integer, handling both string and integer cases
        if data['record_id'].dtype == 'object':  # If record_id is a string
            data['record_id'] = data['record_id'].str.replace('rec_', '', regex=False).astype('int64')
        else:
            data['record_id'] = data['record_id'].astype('int64')  # Ensure it's int64 if already numeric

        # Step 2: Fetch new data and combine with existing data
        new_data = fetch_new_data(data)
        if not new_data.empty:
            new_data['datetime'] = pd.to_datetime(new_data['datetime'], errors='coerce')
            data = pd.concat([data, new_data], ignore_index=True)
            logging.info(f"Combined dataset now has {len(data)} rows.")
        else:
            logging.info("No new data fetched. Proceeding with existing dataset.")

        # Ensure datetime remains correct after combining
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data = data.sort_values('datetime').reset_index(drop=True)
        logging.info(f"Datetime type after combining: {data['datetime'].dtype}")

        # Step 3: Drop low-importance columns early to match the feature group schema
        columns_to_drop = [
            'is_weekend', 'had_rain', 'heavy_rain', 'wind_category',
            'wind_pm_impact', 'wind_rain_interaction'
        ]
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        logging.info(f"After dropping low-importance columns: {list(data.columns)}")

        # Step 4: Clean the data
        # Remove invalid PM10 values (-9999)
        data = data[data['pm10'] != -9999]
        logging.info(f"After removing invalid PM10 values: {len(data)} rows.")

        # Cap PM10 values at 500 to handle outliers
        data = data[data['pm10'] <= 500]
        logging.info(f"After capping PM10 at 500: {len(data)} rows.")
        logging.info(f"PM10 stats after cleaning:\n{data['pm10'].describe()}")

        # Step 5: Create new features
        # Time-based features
        data['hour'] = data['datetime'].dt.hour.fillna(0).astype('int32')  # Use int32 to match Hopsworks schema
        data['day'] = data['datetime'].dt.day.fillna(0).astype('int32')
        data['month'] = data['datetime'].dt.month.fillna(0).astype('int32')
        data['day_of_week'] = data['datetime'].dt.dayofweek.fillna(0).astype('int32')

        # Cyclic encoding for hour (to capture daily patterns)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24).astype('float64')
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24).astype('float64')

        # Lag features (previous hour's values)
        data['lag_1_pm10'] = data['pm10'].shift(1).astype('float64')
        data['lag_1_pm25'] = data['pm25'].shift(1).astype('float64')
        data['lag_1_aqi'] = data['aqi'].shift(1).astype('float64')

        # Fill missing lag values with the next available value
        lag_columns = ['lag_1_pm10', 'lag_1_pm25', 'lag_1_aqi']
        for col in lag_columns:
            data[col] = data[col].ffill()

        # Rolling statistics (3-hour window)
        data['rolling_pm10_mean'] = data['pm10'].rolling(window=3).mean().astype('float64')
        data['rolling_pm10_std'] = data['pm10'].rolling(window=3).std().astype('float64')
        data['rolling_pm25_mean'] = data['pm25'].rolling(window=3).mean().astype('float64')
        data['rolling_pm25_std'] = data['pm25'].rolling(window=3).std().astype('float64')

        # the column mean
        rolling_columns = ['rolling_pm10_mean', 'rolling_pm10_std', 'rolling_pm25_mean', 'rolling_pm25_std']
        for col in rolling_columns:
            data[col] = data[col].fillna(data[col].mean())

        # Change features (difference from previous hour)
        data['pm10_change'] = data['pm10'].diff().fillna(0).astype('float64')
        data['pm25_change'] = data['pm25'].diff().fillna(0).astype('float64')

        # Ratio feature (avoid division by zero)
        data['pm25_pm10_ratio'] = np.where(data['pm10'] != 0, data['pm25'] / data['pm10'], 0).astype('float64')

        # Environmental interaction features
        # Handle missing temperature and humidity for new data
        data['temperature'] = data['temperature'].fillna(data['temperature'].mean()).astype('float64')
        data['humidity'] = data['humidity'].fillna(data['humidity'].mean()).round().astype('int64')
        data['wind_speed'] = data['wind_speed'].fillna(data['wind_speed'].mean()).astype('float64')
        data['precipitation'] = data['precipitation'].fillna(data['precipitation'].mean()).astype('float64')
        data['dew_point'] = (data['temperature'] - ((100 - data['humidity']) / 5)).astype('float64')
        data['heat_index'] = (data['temperature'] + 0.5 * data['humidity']).astype('float64')
        data['temp_humidity'] = (data['temperature'] * data['humidity']).astype('float64')

        # Ensure other columns have correct types
        data['aqi'] = data['aqi'].astype('int64')
        data['pm10'] = data['pm10'].astype('float64')
        data['pm25'] = data['pm25'].astype('float64')
        data['no2'] = data['no2'].astype('float64')
        data['o3'] = data['o3'].astype('float64')

        # Step 6: Final cleanup
        # Debug: Check for NaN values before dropping
        nan_counts = data.isna().sum()
        logging.info(f"NaN counts before dropping:\n{nan_counts[nan_counts > 0]}")

        # Remove any rows with missing values
        data = data.dropna()
        logging.info(f"Final row count: {len(data)}")

        # Step 7: Save to Hopsworks
        # Connect to Hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        feature_store = project.get_feature_store()

        # Create or update the feature group
        feature_group = feature_store.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=1,
            primary_key=['record_id'],
            description="Features for PM10 prediction",
            event_time='datetime'
        )

        # Insert the data into the feature group
        feature_group.insert(data, write_options={"wait_for_job": True})
        logging.info(f"Feature group created/updated with {len(data)} rows.")

        # Close the Hopsworks connection
        hopsworks.logout()
        logging.info("Hopsworks connection closed.")

    except Exception as e:
        logging.error(f"Error during feature creation: {str(e)}")
        hopsworks.logout()
        raise

if __name__ == "__main__":
    logging.info("Feature engineering pipeline started...")
    create_features()
    logging.info("Feature engineering pipeline finished successfully!")