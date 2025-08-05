import os
import pandas as pd
import redivis
import sys
import yaml

# change working directory to this script's location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# append working directory to path
sys.path.append(os.getcwd())

sys.path.append("..")
from census import preproc_census_df, gen_varbs, merge_corelogic_census, wrapper

# load config
with open('../../config/clean_data/census_config.yaml', 'r') as stream:
    out = yaml.safe_load(stream)

# check if tract file already exists. if not, create it.
if os.path.exists(os.path.join(out['outdir'],'census_tract_level.csv')):
    tract = pd.read_csv(os.path.join(out['outdir'],'census_tract_level.csv'))

else: 
    # merge tract data
    tract = None

    for key in out['tract'].keys():
        data = pd.read_csv(os.path.join(out['datapath'],out['tract'][key]['filename']))
        data = preproc_census_df(data, geo='tract', col_list=out['tract'][key]['varlist'])
        if tract is None:
            tract = data
        else:
            tract = tract.merge(data, how='outer', on='tract_id', validate='1:1')

        # generate varbs
        tract = gen_varbs(tract, geo='tract')

        # write clean file
        tract.to_csv(os.path.join(out['outdir'], 'census_tract_level.csv'), index=False)

# check if block group file already exists. if not, create it.

if os.path.exists(os.path.join(out['outdir'],'census_bg_level.csv')):
    block_group = pd.read_csv(os.path.join(out['outdir'],'census_bg_level.csv'))

else:
    block_group = None

    for key in out['block_group'].keys():
        data = pd.read_csv(os.path.join(out['datapath'],out['block_group'][key]['filename']))
        data = preproc_census_df(data, geo='bg', col_list=out['block_group'][key]['varlist'])
        if block_group is None:
            block_group = data
        else:
            block_group = block_group.merge(data, how='outer', on='block_group_id', validate='1:1')

        # generate varbs
        block_group = gen_varbs(block_group, geo='bg')

        # write clean file
        block_group.to_csv(os.path.join(out['outdir'], 'census_bg_level.csv'), index=False)

# merge to corelogic
corelogic = pd.read_csv(out['corelogic']['filename'])
#user = redivis.user(out['corelogic']['redivis_user'])
#workflow = user.workflow(out['corelogic']['workflow'])
#table = workflow.table(out['corelogic']['workflow_table'])
#corelogic = table.to_pandas_dataframe()
merged = merge_corelogic_census(corelogic, tract, block_group)

# query census to reverse geocode observations where census merge failed
output = wrapper(merged, tract, block_group, mapping_path=out['mapping_path'], query=False, chunk_size=None)
#output = wrapper(merged, tract, block_group, mapping_path=out['mapping_path'], query=True, chunk_size=1000)

print('writing data')
output.to_csv(os.path.join(out['outdir'], 'corelogic_census_2018_2023.csv'), index=False)
