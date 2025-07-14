import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from typing import Union
from sklearn.metrics import confusion_matrix
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# load data
## UPDATE FILE LOCATION
corelogic = pd.read_csv('../data/cook_county/cook_sales_assessments_2018_2023.csv')
## UPDATE FILE LOCATION
ccao = pd.read_parquet('../data/cook_county/training_data.parquet')

# clean corelogic
corelogic.dropna(subset=['latitude', 'longitude', 'address', 'SALE_AMOUNT', 'sale_date'], inplace=True)
corelogic['sale_yr_month'] = [math.floor(x/100) for x in corelogic['sale_date']]
corelogic=corelogic[corelogic['MULTI_OR_SPLIT_PARCEL_CODE'].isnull()]
corelogic['latitude_trunc'] = corelogic['latitude'].round(4)
corelogic['longitude_trunc'] = corelogic['longitude'].round(4)
corelogic['street_num'] = corelogic['address'].str.extract(r'^(\d+)').astype(int)
corelogic.drop_duplicates(subset=['CLIP', 'sale_yr_month'],inplace=True)

# clean CCAO
ccao.dropna(subset=['loc_latitude', 'loc_longitude', 'loc_property_address', 'meta_sale_date'], how='any', inplace=True)
ccao['street_num'] = ccao['loc_property_address'].str.extract(r'^(\d+)').astype(int)
ccao[['loc_latitude', 'loc_longitude']] = ccao[['loc_latitude', 'loc_longitude']].round(6)
ccao[['loc_latitude_trunc', 'loc_longitude_trunc']] = ccao[['loc_latitude', 'loc_longitude']].round(4)
ccao['date_stripped'] = ccao.meta_sale_date.astype(str).str.replace('-', '', regex=False)
ccao['date_stripped'] = ccao.date_stripped.astype(int)
ccao['date_yr_month'] = [math.floor(x/100) for x in ccao.date_stripped]

# merge corelogic and ccao
merged = corelogic.merge(ccao[['street_num', 
                               'loc_latitude_trunc', 
                               'loc_longitude_trunc', 
                               'loc_latitude', 
                               'loc_longitude', 
                               'meta_sale_document_num', 
                               'meta_sale_price', 
                               'loc_property_address', 
                               'date_stripped', 
                               'date_yr_month']], 
                         left_on=['street_num', 'latitude_trunc', 'longitude_trunc', 'sale_yr_month'], 
                         right_on=['street_num', 'loc_latitude_trunc', 'loc_longitude_trunc', 'date_yr_month'], 
                         how='inner')

# drop duplicates and unmatched addresses
merged['address_similarity'] = [similar(str(x),str(y)) for x, y in zip(merged.address, merged.loc_property_address)]
merged = merged.loc[merged.address_similarity >= 0.6]

merged.sort_values("address_similarity", ascending = False).drop_duplicates(subset=['CLIP', 'meta_sale_document_num', 'meta_sale_price', 'date_stripped'], 
                                                                            keep="first", 
                                                                            inplace=True)

merged.sort_values('address_similarity', ascending = False).drop_duplicates(subset=['CLIP', 'sale_date', 'SALE_AMOUNT'], 
                                                                            keep='first', 
                                                                            inplace=True)

# ensure no multi-parcel sales remain in data
merged = merged[merged.MULTI_OR_SPLIT_PARCEL_CODE.isnull()]

# generate variables for analysis
merged['pct_error'] = [abs((x-y)/x) for x, y in zip(merged.SALE_AMOUNT, merged.meta_sale_price)]
merged['error_ratio'] = [x/y for x,y in zip(merged.SALE_AMOUNT, merged.meta_sale_price)]
merged['error_amount'] = [abs(x-y)  if z> 0.05 else np.nan for x, y, z in zip(merged.SALE_AMOUNT, merged.meta_sale_price, merged.pct_error)]
merged['is_error'] = [1 if x>0.05 else 0 for x in merged.pct_error]
merged['ratio'] = merged.MARKET_TOTAL_VALUE / merged.SALE_AMOUNT
merged['log_ratio'] = [math.log(x) for x in merged.ratio]
merged['log_ccao_sale_price'] = [math.log(x) for x in merged.meta_sale_price]
merged['log_cotality_sale_price'] = [math.log(x) for x in merged.SALE_AMOUNT]

# write data
## ADD FILE LOCATION
merged.to_csv('INSERT FILE LOCATION')
