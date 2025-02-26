import numpy as np
import pandas as pd
import json
from datetime import datetime
from coordinates import states
import os
from dotenv import load_dotenv
load_dotenv()


json_data_path=os.getenv('json_aqi_data_path')
csv_data_path=os.getenv('csv_aqi_data_path')
csv_folder_path=os.getenv('csv_aqi_folder_path')

if __name__=='__main__':

    print('******************Converting To DataFrame Now*********************')
    co=[]
    no=[]
    no2=[]
    o3=[]
    so2=[]
    pm2_5=[]
    pm10=[]
    nh3=[]
    aqi=[]
    state_name=[]
    timestamp=[]
    if os.path.exists(csv_folder_path):
        print(f"{csv_folder_path} Folder Path Exists. Starting Data Parsing Now")
    else:
        os.makedirs(csv_folder_path)
        print(f"{csv_folder_path} Folder Created: Starting Data Parsing")
        
    for state in states:
        print(f"parsing {state} data")
        read_file_path=json_data_path+state+'.json'
        with open(read_file_path) as f:
            data = json.load(f)
            for i in range(len(data['list'])):
                aqi.append(data['list'][i]['main']['aqi'])
                co.append(data['list'][i]['components']['co'])
                no.append(data['list'][i]['components']['no'])
                no2.append(data['list'][i]['components']['no2'])
                o3.append(data['list'][i]['components']['o3'])
                so2.append(data['list'][i]['components']['so2'])
                pm2_5.append(data['list'][i]['components']['pm2_5'])
                pm10.append(data['list'][i]['components']['pm10'])
                nh3.append(data['list'][i]['components']['nh3'])
                temp_date=int(data['list'][i]['dt'])
                date=datetime.utcfromtimestamp(temp_date).strftime('%Y-%m-%d %H:%M:%S')
                timestamp.append(date)
                state_name.append(state)
            del data


    aqi_data=pd.DataFrame(co,columns=['co'])
    aqi_data['no']=no
    aqi_data['no2']=no2
    aqi_data['o3']=o3
    aqi_data['so2']=so2
    aqi_data['pm2_5']=pm2_5
    aqi_data['pm10']=pm10
    aqi_data['nh3']=nh3
    aqi_data['aqi']=aqi_data.mean(axis=1).round(2)
    aqi_data['state']=state_name
    aqi_data['timestamp']=timestamp
    aqi_data.to_csv(csv_data_path)