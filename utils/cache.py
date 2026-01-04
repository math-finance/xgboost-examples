import os
from base.utils import auto_path_join, hzlog
import functools
import pandas as pd
import datetime
import time
from pathlib import Path
from config import DATA_ROOT

__CACHE_ROOT = auto_path_join(DATA_ROOT, 'caches')
def load_cached_table_if_any(key, check_existence_only=False):
    assert key[0] == '/', 'key need to start with /'
    cached_f = auto_path_join(__CACHE_ROOT, key[1:])

    expired_caches = []
    if os.path.exists(cached_f) == False:
        hzlog(f'Cannot find untimed cached file: [ROOT]/caches/{key[1:]}', level=0)
        cached_f_path = Path(cached_f)
        cached_f_folder = str(cached_f_path.parent)
        cached_f_file = str(cached_f_path.name)
        cached_files = [f for f in os.listdir(cached_f_folder) if f.startswith(f'{cached_f_file}.exp.')]
        if len(cached_files) == 0:
            return None
    
        cached_files_sorted = sorted(cached_files)
        hzlog(f'Found timed cache files: {cached_files_sorted}', level=0)
        now = f'{int(time.time())}'.zfill(10)
        cache_expiring_time = cached_files_sorted[-1].split('.exp.')[-1]

        if now > cache_expiring_time:
            return None
        else:
            cached_f = auto_path_join(cached_f_folder, cached_files_sorted[-1])
            expired_caches = [auto_path_join(cached_f_folder, f) for f in cached_files_sorted if f != cached_files_sorted[-1]]

    # at this point, we know the cache file exists and has not expired
    if check_existence_only:
        return True

    cached_f_time_modified = datetime.datetime.fromtimestamp(round(os.path.getmtime(cached_f), 0), datetime.timezone.utc)
    hzlog(f'Loading cached file ({cached_f_time_modified}): {cached_f.replace(__CACHE_ROOT, "[ROOT]/caches")}', level=0)
    cached_f_df = pd.read_parquet(cached_f)
    
    try:
        for f in expired_caches:
            hzlog(f'Deleting expired cached file: {f}', level=0)
            os.remove(f)
    except:
        pass

    return cached_f_df

def query_tables_from_cache(prefix):
    assert prefix[0] == '/', 'key need to start with /'
    return os.listdir(auto_path_join(__CACHE_ROOT, prefix[1:]))

@functools.lru_cache(maxsize=5)
def ensure_cache_dir(cache_dir):
    assert cache_dir[0] == '/', 'key need to start with /'
    cache_dir = auto_path_join(__CACHE_ROOT, cache_dir[1:])
    if os.path.exists(cache_dir) == False:
        os.makedirs(cache_dir)

def save_table_to_cache(key, df, auto_expire_in_sec=None):
    assert key[0] == '/', 'key need to start with /'
    cache_dir = os.path.dirname(key)
    ensure_cache_dir(cache_dir)
    cache_f = auto_path_join(__CACHE_ROOT, key[1:])

    if auto_expire_in_sec is not None:
        t_expiring = f'{int(time.time()) + auto_expire_in_sec}'.zfill(10)
        cache_f = f'{cache_f}.exp.{t_expiring}'
    df.to_parquet(cache_f, compression='zstd')
    hzlog(f'Saved table to cache: {cache_f}', level=0)

def delete_table_from_cache(key):
    assert key[0] == '/', 'key need to start with /'
    cache_f = auto_path_join(__CACHE_ROOT, key[1:])
    os.remove(cache_f)

def run_app_http(port=8080):
    from twisted.web.server import Site
    from twisted.web.static import File
    from twisted.internet import reactor
    static_files_directory = f'{DATA_ROOT}'
    resource = File(static_files_directory)
    site = Site(resource)
    reactor.listenTCP(8080, site)
    reactor.run()
