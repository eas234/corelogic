import os
import numpy as np
import pandas as pd

## Change working directory to this script's location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

## Read data and remove bad rows
df = pd.read_csv('/oak/stanford/groups/deho/proptax/clean/corelogic_census_2018_2023.csv', usecols=['fips', 'SALE_YEAR', 'MULTI_OR_SPLIT_PARCEL_CODE', 'CALCULATED_TOTAL_VALUE', 'CALCULATED_TOTAL_VALUE_SOURCE_CODE', 'SALE_AMOUNT'])
print('Original Shape:', df.shape)
df = df[df['MULTI_OR_SPLIT_PARCEL_CODE'].isnull()]
print('Drop Multi-Parcel Shape:', df.shape)
df = df.dropna(subset=['fips', 'SALE_YEAR', 'CALCULATED_TOTAL_VALUE', 'SALE_AMOUNT'])
print('Drop Null Shape:', df.shape)

df = df[['fips', 'APPRAISED_TOTAL_VALUE', 'ASSESSED_TOTAL_VALUE', 'MARKET_TOTAL_VALUE', 'SALE_AMOUNT', 'SALE_YEAR']]
df['APPRAISED_SALE_RATIO'] = df['APPRAISED_TOTAL_VALUE'] / df['SALE_AMOUNT']
df['ASSESSED_SALE_RATIO'] = df['ASSESSED_TOTAL_VALUE'] / df['SALE_AMOUNT']
df['MARKET_SALE_RATIO'] = df['MARKET_TOTAL_VALUE'] / df['SALE_AMOUNT']
df['APPRAISED_MARKET_RATIO'] = df['APPRAISED_TOTAL_VALUE'] / df['MARKET_TOTAL_VALUE']
df['ASSESSED_MARKET_RATIO'] = df['ASSESSED_TOTAL_VALUE'] / df['MARKET_TOTAL_VALUE']
df['ASSESSED_APPRAISED_RATIO'] = df['ASSESSED_TOTAL_VALUE'] / df['APPRAISED_TOTAL_VALUE']

df = df[['fips', 'SALE_YEAR', 'APPRAISED_SALE_RATIO', 'ASSESSED_SALE_RATIO', 'MARKET_SALE_RATIO', 'APPRAISED_MARKET_RATIO', 'ASSESSED_MARKET_RATIO', 'ASSESSED_APPRAISED_RATIO']].groupby(['fips', 'SALE_YEAR']).agg(['mean', 'median', 'std', 'count']).reset_index()
df.columns = ["_".join(col_name).rstrip('_') for col_name in df.columns] 

df.to_csv('/oak/stanford/groups/deho/proptax/clean/value_ratios_all_fips_2018_2023.csv', index=False)