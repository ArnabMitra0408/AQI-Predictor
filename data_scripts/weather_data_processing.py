import json
import os
from dotenv import load_dotenv
import pandas as pd
from coordinates import states, latitudes, longitudes
load_dotenv()


state_weather_data_path=os.getenv("state_weather_data_path")
csv_folder_path=os.getenv('csv_weather_folder_path')
csv_data_path=os.getenv('csv_weather_data_path')
dataframes=[]

if __name__=='__main__':

    print("Creating Unified Weather Dataset")
    if os.path.exists(csv_folder_path):
        print(f"{csv_folder_path} Folder Path Exists. Starting Data Parsing Now")
    else:
        os.makedirs(csv_folder_path)
        print(f"{csv_folder_path} Folder Created: Starting Data Parsing")
        
    for state in states:
        print(f"parsing {state} data")
        read_file_path=state_weather_data_path+state+'.csv'
        df=pd.read_csv(read_file_path)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(csv_data_path,index=False)
    print("Final Weather Data Saved")