import os
import pandas as pd
import functools
import pickle
import shutil
import uuid
import io
from config import DATA_ROOT
from base.utils import auto_path_join, hzlog


__DIFC_WRITE_TEMP_PREFIX = '/tmp/difc/temp_folder_dont_touch'
@functools.lru_cache()
def cached_makedirs(folder):
    os.makedirs(folder, exist_ok=True)
cached_makedirs(__DIFC_WRITE_TEMP_PREFIX)


def atomic_rename(src, dst):
    try:
        # Create a hard link to the source as the destination
        os.link(src, dst)
        # Remove the original file if link is successful
        os.remove(src)
    except FileExistsError:
        print(f"Rename failed: {dst} already exists.")
    except Exception as e:
        print(f"Error occurred: {e}")
        

def difc_exist(key, is_dataframe):
    assert key[0] == '/', 'key need to start with /'
    key =  auto_path_join(DATA_ROOT, key[1:])
    name = f'{key}.parquet' if is_dataframe else f'{key}.pkl'
    return os.path.exists(name)

def difc_write(obj, key, is_dataframe):
    assert key[0] == '/', 'key need to start with /'
    key =  auto_path_join(DATA_ROOT, key[1:])
    temp_file = auto_path_join(__DIFC_WRITE_TEMP_PREFIX, str(uuid.uuid4()))
    folder = os.path.dirname(key)
    cached_makedirs(folder)

    # please do not use anything else below apart from os.rename()
    # as we need to make sure copy not happening, if copying is happening you are doing thing wrong somehow
    if is_dataframe:
        assert isinstance(obj, pd.DataFrame)
        obj.to_parquet(temp_file, compression='zstd')
        name = f'{key}.parquet'
        # os.replace(src=temp_file, dst=name)
        shutil.move(src=temp_file, dst=name)
    elif isinstance(obj, dict):
        name = f'{key}.pkl'
        if any(isinstance(v, pd.DataFrame) for v in obj.values()):
            new_obj = dict()
            for k, v in obj.items():
                if isinstance(v, pd.DataFrame):
                    value_bio = io.BytesIO()
                    if v.empty:
                        v = pd.DataFrame()
                    v.to_parquet(value_bio, compression='zstd')
                    value_bio.seek(0)
                    new_obj[k] = {
                        'Type': 'Parquet',
                        'BytesIO': value_bio.read()
                    }
                else:
                    new_obj[k] = v
            obj = new_obj
        else:
            pass
        with open(temp_file, 'wb') as f:
            pickle.dump(obj, f)
        os.replace(src=temp_file, dst=name)
    else:
        raise Exception("Uknown type")
        # please discuss with Haoxian before you implement/change this


def difc_list(prefix):
    assert prefix[0] == '/', 'prefix need to start with /'
    files = sorted(os.listdir(auto_path_join(DATA_ROOT, prefix[1:])))
    return files

def difc_latest_scenario(scenario):
    prefix = auto_path_join('/model_data/scenarios/', scenario)
    latest_month = auto_path_join(prefix, max(difc_list(prefix)))
    latest = auto_path_join(latest_month, max(difc_list(latest_month)))
    return latest


def difc_load(name):
    assert name[0] == '/', 'name need to start with /'
    name =  auto_path_join(DATA_ROOT, name[1:])
    with open(name, 'rb') as f:
        bio = f.read()
    return bio


def difc_read(key, is_dataframe=True, cet_index=False, not_exist_raise=True):
    #it is not worth the risk of having stale data for the benefits of having timed cache, just dont do it for difc_read 
    assert key[0] == '/', 'key need to start with /'
    suffix = 'pkl' if not is_dataframe else 'parquet'

    f = f'{key}.{suffix}'
    f = f.replace('.pkl.pkl', '.pkl')

    f_path = auto_path_join(DATA_ROOT, f[1:])
    if not is_dataframe:
        with open(f_path, 'rb') as fh:
            obj = pickle.load(fh)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict) and 'Type' in v and 'BytesIO' in v:
                        if v['Type'].lower() == 'parquet':
                            df = pd.read_parquet(io.BytesIO(v['BytesIO']) )
                            if cet_index is True:
                                assert 'timestamp[UTC]' in df.columns, f'{key}: {df.columns}'
                                df = df.sort_values('timestamp[UTC]')
                                df = df.set_index('timestamp[UTC]').tz_localize('UTC').tz_convert('CET')
                                df.index.name = 'timestamp[CET]'
                            obj[k] = df
                        else:
                            pass
                    else:
                        pass
            else:
                pass

            return obj
    else:
        try:
            df = pd.read_parquet(f_path)
        except :
            if not_exist_raise is False:
                if os.path.exists(f_path) is False:
                    return None
            raise
        
        if cet_index:
            assert 'timestamp[UTC]' in df.columns, f'{key}: {df.columns}'
            df = df.sort_values('timestamp[UTC]')
            df = df.set_index('timestamp[UTC]').tz_localize('UTC').tz_convert('CET')
            df.index.name = 'timestamp[CET]'
        return df
