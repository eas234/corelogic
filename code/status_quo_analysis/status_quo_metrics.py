import sys
import os
import numpy as np
import pandas as pd

## Change working directory to this script's location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

sys.path.insert(0, '..')
from evaluation import create_cross_county_comparison

## Read data and remove bad rows
df = pd.read_csv('/oak/stanford/groups/deho/proptax/clean/corelogic_census_2018_2023.csv', usecols=['fips', 'SALE_YEAR', 'MULTI_OR_SPLIT_PARCEL_CODE', 'CALCULATED_TOTAL_VALUE', 'CALCULATED_TOTAL_VALUE_SOURCE_CODE', 'SALE_AMOUNT'])
print('Original Shape:', df.shape)
df = df[df['MULTI_OR_SPLIT_PARCEL_CODE'].isnull()]
print('Drop Multi-Parcel Shape:', df.shape)
df = df.dropna(subset=['fips', 'SALE_YEAR', 'CALCULATED_TOTAL_VALUE', 'SALE_AMOUNT'])
print('Drop Null Shape:', df.shape)

## Create county-year-level sales metrics
out_count = df[['fips', 'SALE_YEAR', 'SALE_AMOUNT']].groupby(['fips', 'SALE_YEAR']).count().reset_index()
out_count.columns = ['fips', 'year', 'num_sales']
out_avg = df[['fips', 'SALE_YEAR', 'SALE_AMOUNT']].groupby(['fips', 'SALE_YEAR']).mean().reset_index()
out_avg.columns = ['fips', 'year', 'avg_sale_price']
out_med = df[['fips', 'SALE_YEAR', 'SALE_AMOUNT']].groupby(['fips', 'SALE_YEAR']).median().reset_index()
out_med.columns = ['fips', 'year', 'med_sale_price']
out_std = df[['fips', 'SALE_YEAR', 'SALE_AMOUNT']].groupby(['fips', 'SALE_YEAR']).std().reset_index()
out_std.columns = ['fips', 'year', 'std_sale_price']
out_type = df[['fips', 'SALE_YEAR', 'CALCULATED_TOTAL_VALUE_SOURCE_CODE']].value_counts().reset_index().drop_duplicates(subset=['fips', 'SALE_YEAR'], keep='first').drop(columns=['count'])
out_type.columns = ['fips', 'year', 'assessment_type']

out = out_count.merge(out_avg, on=['fips', 'year']).merge(out_med, on=['fips', 'year']).merge(out_std, on=['fips', 'year']).merge(out_type, on=['fips', 'year'])

out.to_csv('/oak/stanford/groups/deho/proptax/clean/sale_summary_statistics_all_fips_2018_2023.csv', index=False)

## Cross-county comparison
df = df[['fips', 'SALE_YEAR', 'CALCULATED_TOTAL_VALUE', 'SALE_AMOUNT']]
df.columns = ['fips', 'year', 'assessed', 'sale']

out = create_cross_county_comparison(df)
out = out.dropna()

out.to_csv('/oak/stanford/groups/deho/proptax/clean/accuracy_regressivity_metrics_all_fips_2018_2023.csv', index=False)