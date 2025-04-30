import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import hopsworks
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
LAT = 24.8607  # Karachi latitude
LON = 67.0011  # Karachi longitude
DATASET_PATH = Path("AQI Prediction Dataset.csv")

# Load API keys from environment variables
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def fetch_openweather_data(start_date, end_date):
    """Fetch historical air quality and weather data from OpenWeather API."""
    logger.info("Fetching new data from OpenWeather API...")
    if not OPENWEATHER_API_KEY:
        logger.error("OPENWEATHER_API_KEY environment variable not set.")
        return pd.DataFrame()
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_date}&end={end_date}&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        records = []
        for entry in data['list']:
            record = {
                'datetime': datetime.fromtimestamp(entry['dt']),
                'pm25': entry['components']['pm2_5'],
                'pm10': entry['components']['pm10'],
                'no2': entry['components']['no2'],
                'o3': entry['components']['o3'],
                'aqi': entry['main']['aqi']
            }
            records.append(record)
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        logger.info(f"Fetched {len(df)} rows from OpenWeather API.")
        return df
    except Exception as e:
        logger.error(f"Error fetching OpenWeather data: {str(e)}")
        return pd.DataFrame()

def load_existing_data():
    """Load existing dataset from CSV."""
    try:
        if DATASET_PATH.exists():
            df = pd.read_csv(DATASET_PATH, parse_dates=['datetime'], date_format='%d/%m/%Y %H:%M', dayfirst=True)
            logger.info(f"Loaded {len(df)} rows from dataset.")
            # Validate datetime column
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                logger.error("Datetime column contains non-datetime values. Attempting to fix...")
                df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M', errors='coerce', dayfirst=True)
                if df['datetime'].isna().any():
                    logger.error(f"Found {df['datetime'].isna().sum()} rows with invalid datetime values. Dropping these rows.")
                    df = df.dropna(subset=['datetime'])
            return df
        else:
            logger.info("No existing dataset found. Creating new dataset.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

def save_to_hopsworks(df):
    """Save the dataset to Hopsworks Feature Group."""
    try:
        if not HOPSWORKS_API_KEY:
            raise ValueError("HOPSWORKS_API_KEY environment variable not set.")
        
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name="pm10_features",
            version=1,
            description="PM10 and related features for Karachi",
            primary_key=['datetime']
        )
        fg.insert(df, overwrite=False)
        logger.info("Successfully saved data to Hopsworks Feature Group.")
    except Exception as e:
        logger.error(f"Error saving to Hopsworks: {str(e)}")
        raise

def create_features():
    """Main function to create and update features."""
    logger.info("Feature engineering started...")
    
    # Load existing data
    existing_df = load_existing_data()
    
    # Determine the date range for fetching new data
    if not existing_df.empty:
        last_date = existing_df['datetime'].max()
        logger.info(f"Last date in dataset: {last_date}, type: {type(last_date)}")
        if pd.isna(last_date):
            logger.error("Last date is NaT. Check dataset for invalid datetime values.")
            return
        start_date = int((last_date + timedelta(hours=1)).timestamp())
        logger.info(f"Fetching new data from {last_date + timedelta(hours=1)} (timestamp: {start_date})")
    else:
        # If no data exists, fetch the last 30 days
        start_date = int((datetime.now() - timedelta(days=30)).timestamp())
        logger.info(f"No existing data. Fetching last 30 days from timestamp: {start_date}")
    
    end_date = int(datetime.now().timestamp())
    logger.info(f"Fetching data up to timestamp: {end_date}")
    
    # Fetch new data from OpenWeather API
    new_data = fetch_openweather_data(start_date, end_date)
    
    if not new_data.empty:
        # Remove overlaps based on datetime
        if not existing_df.empty:
            new_data = new_data[~new_data['datetime'].isin(existing_df['datetime'])]
            logger.info(f"{len(new_data)} new rows after removing overlaps.")
        else:
            logger.info(f"{len(new_data)} new rows to be added.")
        
        # Append new data to existing dataset
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        updated_df = updated_df.sort_values('datetime').reset_index(drop=True)
        
        # Save updated dataset to CSV
        updated_df.to_csv(DATASET_PATH, index=False)
        logger.info(f"Appended {len(new_data)} rows to {DATASET_PATH}.")
        
        # Save to Hopsworks
        save_to_hopsworks(updated_df)
    else:
        logger.warning("No new data fetched from OpenWeather API.")

if __name__ == "__main__":
    try:
        create_features()
    except Exception as e:
        logger.error(f"Feature creation error: {str(e)}")
        exit(1)