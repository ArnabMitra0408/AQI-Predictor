import json
import requests
import os
from dotenv import load_dotenv
from coordinates import states, latitudes, longitudes
import openmeteo_requests
import requests_cache
import pandas as pd
import time
from retry_requests import retry
load_dotenv()

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
state_weather_data_path=os.getenv("state_weather_data_path")
weather_base_url=os.getenv("Base_URL_Weather")
start_date=os.getenv("Start_date_UTC")
end_date=os.getenv("End_date_UTC")

if __name__=='__main__':
    print('****************Data Download Started*************************')
    for index in range(len(latitudes)):
        file_name=state_weather_data_path+states[index]+'.csv'
        if os.path.exists(file_name):
            print(f"Data for {states[index]} already exists. Skipping Download")
            continue
        else:
            print(f"Downloading Data for {states[index]}")
            params = {
                "latitude": float(latitudes[index]),
                "longitude": float(longitudes[index]),
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "wind_direction_10m", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm"]
            }
            responses = openmeteo.weather_api(weather_base_url, params=params)
            response = responses[0]
            hourly = response.Hourly()
            hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
            hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
            hourly_rain = hourly.Variables(2).ValuesAsNumpy()
            hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
            hourly_wind_direction_10m = hourly.Variables(4).ValuesAsNumpy()
            hourly_soil_temperature_0_to_7cm = hourly.Variables(5).ValuesAsNumpy()
            hourly_soil_moisture_0_to_7cm = hourly.Variables(6).ValuesAsNumpy()

            hourly_data = {"date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )}
            hourly_data["temperature_2m"] = hourly_temperature_2m
            hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
            hourly_data["rain"] = hourly_rain
            hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
            hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
            hourly_data["soil_temperature_0_to_7cm"] = hourly_soil_temperature_0_to_7cm
            hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
            hourly_data['State']=states[index]
            hourly_dataframe = pd.DataFrame(data = hourly_data)
            hourly_dataframe.to_csv(file_name,index=False)
            time.sleep(15)                        