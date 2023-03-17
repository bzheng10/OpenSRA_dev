import os

# get local app data directory path
local_app_data_dir = os.environ['LOCALAPPDATA']

# add env variable
cache_dir = os.path.join(local_app_data_dir, 'numba_cache')
os.environ["NUMBA_CACHE_DIR"] = cache_dir