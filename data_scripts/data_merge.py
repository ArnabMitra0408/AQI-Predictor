import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
load_dotenv()

final_data_path=os.getenv("final_data_path")
aqi_data_path=os.getenv("csv_aqi_data_path")
weather_data_path=os.getenv("csv_weather_data_path")



if __name__=='__main__':
    print("Beginning Data Merge")
    weather=pd.read_csv(weather_data_path)
    weather['date'] = pd.to_datetime(weather['date']).dt.tz_convert(None)

    aqi=pd.read_csv(aqi_data_path)
    aqi['timestamp']=pd.to_datetime(aqi['timestamp'])

    min_dt=max(weather['date'].min(),aqi['timestamp'].min())
    max_dt=min(weather['date'].max(),aqi['timestamp'].max())

    weather=weather[(weather['date'] >= min_dt) & (weather['date'] <= max_dt)]
    aqi=aqi[(aqi['timestamp'] >= min_dt) & (aqi['timestamp'] <= max_dt)]

    Final_Data=pd.merge(aqi,weather,how='left',left_on=['timestamp','state'],right_on=['date','State'])

    Final_Data=Final_Data[['timestamp','aqi','state','co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3','temperature_2m',
       'relative_humidity_2m', 'rain', 'wind_speed_10m', 'wind_direction_10m','soil_temperature_0_to_7cm', 'soil_moisture_0_to_7cm']].reset_index(drop=True)
    
    Final_Data.to_csv(final_data_path,index=False,header=True)
    print(f"Data Merged Suceesfully at Location {final_data_path}")