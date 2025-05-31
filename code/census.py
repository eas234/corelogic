import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import pandas as pd
import requests
import time

def preproc_census_df(df: pd.DataFrame, 
                      geo: str='tract',
                      col_list: list=None):
    
    """
    
    inputs:
    - df: pd.DataFrame, census dataframe to process
    - geo: geography level, either 'tract' or 'bg' (block group)
    """
    
    copy = df.copy()
    
    # drop first row (census column headers)
    copy.drop(index=copy.index[0], axis=0, inplace=True)
    
    # gen tract or block group id
    if not isinstance(geo, str):
        raise TypeError("geo must be a string, equal to either 'tract' or 'bg'")

    elif geo == 'tract':
        copy['tract_id'] = pd.to_numeric([x[9:] for x in copy['GEO_ID']], errors='coerce')
        
    elif geo == 'bg':
        copy['block_group_id'] = pd.to_numeric([x[9:] for x in copy['GEO_ID']], errors='coerce')
        
    else:
        raise TypeError("Invalid geography specified. geo must equal either 'tract' or 'bg'.")
        
    # subset to relevant columns
    if col_list and geo == 'tract':
        copy = copy[col_list + ['tract_id']]
        
    elif col_list and geo == 'bg':
        copy = copy[col_list + ['block_group_id']]
        
    for col in col_list:
        copy[col] = pd.to_numeric(copy[col], errors='coerce')
        
    return copy

def gen_varbs(df: pd.DataFrame, 
              geo: str='tract'):
    
    """
    Generate named variables from census inputs.
    
    inputs:
    - df: dataframe with unprocessed columns of data from census
    - geo: string indicating level of geography. Valid values are 'tract' and 'bg'.
    """
    
    # check geo level
    if not isinstance(geo, str):
        raise TypeError("geo must be a string, equal to either 'tract' or 'bg'")
    elif geo != 'tract' and geo != 'bg':
        raise TypeError("Invalid geography specified. geo must equal either 'tract' or 'bg'.")
        
    copy = df.copy()
        
    copy['census_pct_children_' + geo] = (copy.B01001_003E + copy.B01001_004E + copy.B01001_005E + copy.B01001_006E + copy.B01001_027E + copy.B01001_028E + copy.B01001_029E + copy.B01001_030E + copy.B01001_007E + copy.B01001_031E) / copy.B01001_001E
    copy['census_pct_senior_' + geo] = (copy.B01001_020E + copy.B01001_021E + copy.B01001_022E + copy.B01001_023E + copy.B01001_024E + copy.B01001_025E + copy.B01001_044E + copy.B01001_045E + copy.B01001_046E + copy.B01001_047E + copy.B01001_048E + copy.B01001_049E) / copy.B01001_001E
    copy['census_med_age_' + geo] = copy.B01002_001E
    copy['census_pct_married_hh_' + geo] = copy.B11001_002E/copy.B11001_001E
    copy['census_pct_single_hh_' + geo] = copy.B11001_008E/copy.B11001_001E
    copy['census_pct_high_school_' + geo] = (copy.B15003_025E + copy.B15003_023E + copy.B15003_022E + copy.B15003_017E)/copy.B15003_001E
    copy['census_pct_college_' + geo] = (copy.B15003_025E + copy.B15003_023E + copy.B15003_022E)/copy.B15003_001E
    copy['census_pct_graduate_' + geo] = (copy.B15003_025E + copy.B15003_023E)/copy.B15003_001E
    copy['census_pct_poverty_' + geo] = copy.B17017_002E / copy.B17017_001E
    copy['census_med_hh_inc_' + geo] = copy.B19013_001E
    copy['census_med_per_cap_inc_' + geo] = copy.B19301_001E
    copy['census_pct_snap_' + geo] = copy.B22010_002E / copy.B22010_001E
    copy['census_unemp_rate_' + geo] = copy.B23025_005E/copy.B23025_002E
    copy['census_med_yr_built_' + geo] = copy.B25035_001E
    copy['census_pct_renter_occ_' + geo] = copy.B25003_002E/copy.B25003_001E
    copy['census_med_rent_' + geo] = copy.B25064_001E
    
    subset = [x for x in copy.columns if 'census' in x]
    
    if geo == 'tract':
        copy = copy[subset + ['tract_id']]
    else:
        copy = copy[subset + ['block_group_id']]
    
    return copy

def merge_corelogic_census(corelogic, 
                           tract,
                           block_group):
    """
    inputs:
    - corelogic: dataframe with corelogic data
    - tract: dataframe with tract-level census varbs
    - block_group: dataframe with block-group level census varbs

    returns:
    - merged: dataframe with merged corelogic and census varbs
    """

    # clean a bit
    corelogic['CENSUS_ID'] = corelogic.CENSUS_ID.astype(str).str.zfill(10)
    corelogic['fips'] = corelogic.fips.astype(str).str.zfill(5)

    # gen merge varbs
    corelogic['tract'] = corelogic.CENSUS_ID.str[:6]
    corelogic['block_group'] = corelogic.CENSUS_ID.str[:7]

    tract.tract_id = tract.tract_id.astype(str).str.zfill(11)
    tract['fips'] = tract.tract_id.str[:5]
    tract['tract'] = tract.tract_id.str[-6:]

    block_group.block_group_id = block_group.block_group_id.astype(str).str.zfill(12)
    block_group['fips'] = block_group.block_group_id.str[:5]
    block_group['block_group'] = block_group.block_group_id.str[-7:]

    # merge tract-level varbs
    merged = corelogic.merge(tract, how='left', on=['fips', 'tract'])

    # merge bg-level varbs
    merged = merged.merge(block_group, how='left', on=['fips', 'block_group'])

    return merged

def split_nulls(data):
    
    copy = data.copy()
    
    nulls = copy[copy['tract_id'].isnull() | copy['block_group_id'].isnull()]
    non_nulls = copy[~copy['tract_id'].isnull() & ~copy['block_group_id'].isnull()]
    
    return nulls, non_nulls

def drop_census_cols(data):
    
    copy = data.copy()
    
    copy = copy[[x for x in copy.columns if 'census' not in x]]
    copy.drop(columns=['fips', 'tract', 'block_group', 'tract_id', 'block_group_id'], inplace=True)
    
    return copy

def split_null_lat_long(data):
    
    copy = data.copy()
    non_nulls = copy[~copy['latitude'].isnull() & ~copy['longitude'].isnull()]
    nulls = copy[copy['latitude'].isnull() | copy['longitude'].isnull()]
    
    return nulls, non_nulls

def concat(df1, df2):
    
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    concat = pd.concat([df1_copy, df2_copy], ignore_index=True)
    
    return concat

def get_distinct_lat_long(df):
    
    copy = df.copy()
    
    copy[['latitude', 'longitude']].drop_duplicates(inplace=True)
    
    return copy

def filter_lat_long_overlap(df, mapping_path):
    
    copy = df.copy()
    
    if os.path.exists(mapping_path): 
        
        # load existing mapper
        mapper = pd.read_csv(mapping_path)

        # drop any nulls in mapper
        mapper = mapper[~mapper.fips.isnull()]
        
        # id lat-long combos where map already exists
        filtered = copy.merge(mapper, how='left', on=['latitude', 'longitude'], indicator=True)
        
        # drop combos with existing map
        filtered = filtered[filtered['_merge'] == 'left_only']
        
        # subset to lat-long
        filtered = filtered[['latitude', 'longitude']]
        
        return filtered
        
    else:
        
        # if no mapper exists, return full lat-long list
        return copy

def query_chunk(chunk, mapping_path, max_attempts=5):
    
    """
    Currently only set up to reverse geocode 2020 census
    """
    
    # specify census URL to query and set up empty results list
    base_url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    results = []
    
    for index, row in chunk.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        url = f"{base_url}?x={lon}&y={lat}&benchmark=2020&vintage=2020&format=json"

        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP request errors
            data = response.json()
            # Extract the relevant geocode information (FIPS, Tract, Block Group)
            if 'result' in data and 'geographies' in data['result']:
                geographies = data['result']['geographies']
                for geog in geographies.get('Census Blocks', []):
                    state = geog.get('STATE', None)
                    county = geog.get('COUNTY', None)
                    tract = geog.get('TRACT', None)
                    block = geog.get('BLOCK', None)
                    results.append({
                        'fips': state + county,
                        'tract': tract,
                        'block_group': tract + block[:1],
                        'latitude': lat,
                        'longitude': lon
                    })
            else:
                results.append({
                    'fips': None,
                    'tract': None,
                    'block_group': None,
                    'latitude': lat,
                    'longitude': lon
                })
        except requests.exceptions.RequestException as e:
            print(f"Error processing {lat}, {lon}: {e}")
            results.append({
                'fips': None,
                'tract': None,
                'block_group': None,
                'latitude': lat,
                'longitude': lon
            })

        # Pause to avoid hitting API rate limits
        time.sleep(0.5)

    # cast maps to dataframe
    results_df = pd.DataFrame(results)
    
    if os.path.exists(mapping_path): 
        
        # load existing mapper
        mapper = pd.read_csv(mapping_path)
        
        # add new maps to mapper
        updated_mapper = concat(mapper, results_df)
        updated_mapper = updated_mapper[~updated_mapper.fips.isnull()]
        
        # save updated mapper
        updated_mapper.to_csv(mapping_path, index=False)
        
        return None
        
    else:
        
        results_df.to_csv(mapping_path, index=False)
        
        return None
    
def clean_val(val, n_chars=5):
    try:
        if pd.isnull(val):
            return None
        # Remove decimal if it's float-like
        return str(int(float(val))).zfill(n_chars)
    except (ValueError, TypeError):
        return None  # fallback for things like '00nan' or non-numeric strings

        
def merge_maps(df, mapping_path):
    
    
    copy = df.copy()
    
    if os.path.exists(mapping_path):
        mapper = pd.read_csv(mapping_path)
        
        # ensure col types are correct format
        mapper.fips = mapper.fips.astype(str).str.zfill(5)
        mapper.tract = mapper.tract.astype(str).str.zfill(6)
        mapper.block_group = mapper.block_group.astype(str).str.zfill(7)
        
        merged = copy.merge(mapper, how='left', on=['latitude', 'longitude'])
        
        return merged
        
    else:
        raise ValueError('No mapper exists')
    
def merge_corelogic_census(corelogic, 
                           tract,
                           block_group):
    
    """
    inputs:
    - corelogic: dataframe with corelogic data
    - tract: dataframe with tract-level census varbs
    - block_group: dataframe with block-group level census varbs

    returns:
    - merged: dataframe with merged corelogic and census varbs
    """

    # clean a bit
    corelogic['CENSUS_ID'] = corelogic.CENSUS_ID.apply(lambda x: clean_val(x, n_chars=10))
    corelogic['fips'] = corelogic.fips.apply(lambda x: clean_val(x, n_chars=5))

    # gen merge varbs
    corelogic['tract'] = corelogic.CENSUS_ID.str[:6]
    corelogic['block_group'] = corelogic.CENSUS_ID.str[:7]

    tract.tract_id = tract.tract_id.apply(lambda x: clean_val(x, n_chars=11))
    tract['fips'] = tract.tract_id.str[:5]
    tract['tract'] = tract.tract_id.str[-6:]

    block_group.block_group_id = block_group.block_group_id.apply(lambda x: clean_val(x, n_chars=12))
    block_group['fips'] = block_group.block_group_id.str[:5]
    block_group['block_group'] = block_group.block_group_id.str[-7:]

    # merge tract-level varbs
    merged = corelogic.merge(tract, how='left', on=['fips', 'tract'])

    # merge bg-level varbs
    merged = merged.merge(block_group, how='left', on=['fips', 'block_group'])

    return merged

def merge_map_to_census(df, 
                           tract,
                           block_group):
    
    """
    inputs:
    - df: dataframe with reverse-geocoded, or mapped, corelogic data
    - tract: dataframe with tract-level census varbs
    - block_group: dataframe with block-group level census varbs

    returns:
    - merged: dataframe with merged corelogic and census varbs
    """

    # clean a bit
    df['fips'] = df.fips.apply(lambda x: clean_val(x, n_chars=5))
    df['tract'] = df.tract.apply(lambda x: clean_val(x, n_chars=6))
    df['block_group'] = df.block_group.apply(lambda x: clean_val(x, n_chars=7))

    tract.tract_id = tract.tract_id.apply(lambda x: clean_val(x, n_chars=11))
    tract['fips'] = tract.tract_id.str[:5]
    tract['tract'] = tract.tract_id.str[-6:]

    block_group.block_group_id = block_group.block_group_id.apply(lambda x: clean_val(x, n_chars=12))
    block_group['fips'] = block_group.block_group_id.str[:5]
    block_group['block_group'] = block_group.block_group_id.str[-7:]

    # merge tract-level varbs
    merged = df.merge(tract, how='left', on=['fips', 'tract'])

    # merge bg-level varbs
    merged = merged.merge(block_group, how='left', on=['fips', 'block_group'])

    return merged

def wrapper(df, tract, block_group, mapping_path='mapper.csv', chunk_size=1000):
    
    # split out unsuccessful matches from successful matches in df
    nulls, non_nulls = split_nulls(df)
    
    # among unsuccessful matches, identify those with null latitudes and longitudes
    null_lat_long, remainder = split_null_lat_long(nulls)
    
    # append null_lat_long to successful matches - to help keep tabs on match coverage down the line
    output = concat(non_nulls, null_lat_long)
    
    # drop census cols from remainder
    remainder = drop_census_cols(remainder)
    
    # subset to unique lat long in remainder
    lat_long_list = get_distinct_lat_long(remainder)
    
    # filter out lat_longs that already have a mapping
    lat_long_list = filter_lat_long_overlap(lat_long_list, mapping_path)
    
    # establish chunk size
    num_chunks = math.ceil(len(lat_long_list)/chunk_size)
    lat_long_list.reset_index(inplace=True)
    
    # chunk lat long list and develop mapper for each chunk
    for i in range(num_chunks):
        print(f"Processing chunk {i+1} of {num_chunks}...")
        chunk = lat_long_list.iloc[i * chunk_size:(i + 1) * chunk_size]
        query_chunk(chunk, mapping_path)
        
    # load completed mapper and merge in geographies
    remainder = merge_maps(remainder, mapping_path)
    
    # merge in census data to remainder
    remainder = merge_map_to_census(remainder, tract, block_group)
    
    # concat remainder to output
    output = concat(output, remainder)
    
    return output
