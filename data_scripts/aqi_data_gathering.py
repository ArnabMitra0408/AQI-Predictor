import numpy as np
import pandas as pd
import requests
import json
from dotenv import load_dotenv
from coordinates import states,latitudes,longitudes
import os
import time
load_dotenv()


API_KEY=os.getenv('api_key')
aqi_base_url=os.getenv('aqi_base_url')
start_date=os.getenv('Start_date_Unix')
end_date=os.getenv('End_date_Unix')
json_aqi_data_path=os.getenv('json_aqi_data_path')



if __name__=='__main__':
    print('****************Data Download Started*************************')
    for index in range(len(latitudes)):
        file_name=json_aqi_data_path+states[index]+'.json'
        if os.path.exists(file_name):
            print(f"Data for {states[index]} already exists. Skipping Download")
            continue
        else:
            print(f"Downloading Data for {states[index]}")
            complete_url = aqi_base_url + "lat=" + latitudes[index] + "&lon=" + longitudes[index]+"&start="+start_date+'&end='+end_date+'&appid='+API_KEY
            response = requests.get(complete_url)
            data = response.json()
            with open(file_name, 'w') as f:
                json.dump(data, f)
            print(file_name,' created!!!')
            time.sleep(60)

    print('JSON DATA DOWNLOADED!!!!')
