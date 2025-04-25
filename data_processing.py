import pandas as pd
import numpy as np
from collections import defaultdict

# load data
df_air_quality = pd.read_csv('/Users/victorli/Documents/GitHub/Air-Quality-Sensing/Data/AirQualityComplete/airquality.csv')
df_station_locations = pd.read_csv('/Users/victorli/Documents/GitHub/Air-Quality-Sensing/Data/AirQualityComplete/station.csv')[["station_id", "latitude", "longitude", 'district_id']]
df_meteorology = pd.read_csv('/Users/victorli/Documents/GitHub/Air-Quality-Sensing/Data/AirQualityComplete/meteorology.csv')

# combine air quality with station locations
df = df_air_quality.merge(df_station_locations, left_on="station_id", right_on="station_id")

# fill in unavailable district data with city data
city_ids = pd.read_csv('/Users/victorli/Documents/GitHub/Air-Quality-Sensing/Data/AirQualityComplete/city.csv')[['city_id']].to_numpy().flatten()
district_ids = set(df['district_id'].unique())
meteorology_ids = set(df_meteorology['id'].unique())
missing_district_ids = district_ids - meteorology_ids

for district_id in missing_district_ids:
    longest_prefix = ''
    for city_id in city_ids:
        city_id = str(city_id)
        district_id = str(district_id)
        if district_id.startswith(city_id):
            if len(longest_prefix) < len(city_id):
                longest_prefix = city_id
    
    df.loc[df['district_id'] == int(district_id), 'district_id'] = int(longest_prefix)

# add in meteorology data
df_merged = pd.merge(df, df_meteorology, left_on=['district_id', 'time'], right_on=['id', 'time'], how='left')

# drop unnecessary columns
df_merged = df_merged.drop(columns=['id', 'district_id'])

# save complete csv
df_merged.to_csv('complete_data.csv', index=False)

# 80/20 train/test split
df_merged = df_merged.sort_values(by='time')

split_index = int(len(df_merged) * 0.8) - 4
df_train = df_merged.iloc[:split_index]
df_test = df_merged.iloc[split_index:]

df_train = df_train.sort_values(by='station_id', kind='mergesort')
df_test = df_test.sort_values(by='station_id', kind='mergesort')

df_train.to_csv('train_data.csv', index=False)
df_test.to_csv('test_data.csv', index=False)