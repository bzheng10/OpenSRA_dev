import os

# get local app data directory path
curr_path = os.path.abspath(__file__)
src_dir = os.path.dirname(curr_path)
opensra_dir = os.path.dirname(src_dir)
# local_app_data_dir = os.environ['LOCALAPPDATA']
# set cache dir
cache_dir = os.path.join(opensra_dir, '__nbcache__')
os.environ["NUMBA_CACHE_DIR"] = cache_dir