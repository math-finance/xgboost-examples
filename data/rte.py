from base.utils import hzlog, assert_period_day_format
from base.utils.cache import load_cached_table_if_any, save_table_to_cache

import pandas as pd
import requests
import time
import functools

# Client ID and secret code, you can get those at https://data.rte-france.com/
__CLIENT_ID       = '70ffdb32-c067-4b57-b15e-1366d70edf42'
__CLIENT_SECRET   = 'd2564da7-7642-440b-9478-c2e4d5574a16'

@functools.lru_cache(maxsize=20, typed=False)
def __get_token(tick):
    r = requests.post('https://digital.iservices.rte-france.com/token/oauth/',
        auth=(__CLIENT_ID, __CLIENT_SECRET))
    if r.ok:
        access_token = r.json()['access_token']
        token_type = r.json()['token_type']
    else:
        Warning("Authentication failed")
        access_token = None
        token_type = None

    return token_type, access_token

def get_token():
    tick = time.time() // 60
    return __get_token(tick=tick)

# unused 
def __manual_fill_data(df, period_date_str):
    raise Exception('This function is unused')
    manual_fill_dates = [
        '2024-06-17',  '2024-09-15', '2024-10-27', '2024-12-05', '2024-12-30', 
        '2025-01-09','2025-07-17', '2025-07-18', '2025-07-19', '2025-07-21',
    ]
    if period_date_str in manual_fill_dates:
        return df.ffill().bfill()
    else:
        return df


def _download_actual_generation_by_day(period_day, use_cache=True, cet_index=True):
    assert_period_day_format(period_day)
    cache_key = f'/rte/actual_gen_by_day/{period_day}'
    period_date_str = period_day
    
    if use_cache:
        cached_df = load_cached_table_if_any(key=cache_key)
        if cached_df is not None:
            assert 'timestamp[CET]' == cached_df.index.name, Exception(f'index name "{cached_df.index.name}" != "timestamp[CET]"')
            if cet_index == False:
                cached_df = cached_df.tz_convert('UTC')
                cached_df.index.name = 'timestamp[UTC]'

            return cached_df

    # Base URL for the RTE API
    BASE_URL = 'https://digital.iservices.rte-france.com/open_api/actual_generation/v1/actual_generations_per_unit'

    token_type, access_token = get_token()
    # Headers for the request
    headers = {
        'Authorization': f'{token_type} {access_token}',
        'Content-Type': 'application/json'
    }

    period_day = pd.Timestamp(period_day).normalize()
    period_day_index = pd.date_range(start=period_day, end=((period_day + pd.offsets.Day(1)).normalize() - pd.offsets.Hour(1)), freq='h', tz='CET')

    # Parameters for the request
    params = {
        'start_date': (pd.Timestamp(period_day) + pd.offsets.Day(-1)).strftime('%Y-%m-%dT00:00:00Z'),
        'end_date'  : (pd.Timestamp(period_day) + pd.offsets.Day(1)).strftime('%Y-%m-%dT00:00:00Z')
    }

    # Make the API request
    response = requests.get(BASE_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        unit_names = []
        unit_types = []
        unit_series = []
        for unit in data['actual_generations_per_unit']:
            unit_names.append(unit['unit']['name'])
            unit_types.append(unit['unit']['production_type'])
            series = pd.DataFrame(unit['values']).set_index('start_date')
            series.index = pd.DatetimeIndex(series.index, tz='UTC').tz_convert('CET')
            unit_series.append(series['value'].reindex(period_day_index))

        df = pd.concat(unit_series, axis=1).resample('h').mean()
        df.columns = pd.MultiIndex.from_tuples(zip(unit_types, unit_names), names=['type', 'name'])
        df.index = pd.DatetimeIndex(df.index).tz_convert('CET')
        df.index.name = 'timestamp[CET]'

        if df.isna().sum(axis=1).max() < len(df.columns):
            save_table_to_cache(key=cache_key, df=df)
    else:
        hzlog(f'Failed to actual generation by day for {period_day} data: {response.status_code} - {response.text}')
        return None

    return df

def download_actual_generation(month, use_cache=True, cet_index=True):
    period_start = f'{month}-01'
    period_end = (pd.Timestamp(period_start) + pd.offsets.MonthBegin(1) - pd.offsets.Day(1)).strftime('%Y-%m-%d')
    today = pd.Timestamp('today').strftime('%Y-%m-%d')
    period_end = today if period_end > today else period_end

    hzlog(f'Downloading actual production per unit for [{period_start}] => [{period_end}]')
    dfs = []
    for d in pd.date_range(period_start, end=period_end, freq='D').strftime('%Y-%m-%d'):
        try:
            df = _download_actual_generation_by_day(period_day=d, use_cache=use_cache, cet_index=True)
            if df is not None:
                dfs.append(df)
        except Exception as e:
            hzlog(f'Failed to download actual generation for {d}: {e}')

    if len(dfs) == 0:
        hzlog(f"No data found for actual generation for [{period_start}] -> [{period_end}]")
        return None

    df = pd.concat(dfs)

    if df.empty:
        return None

    if cet_index == False:
        df.index = pd.DatetimeIndex(df.index).tz_convert('UTC')
        df.index.name = 'timestamp[UTC]'

    return df


def download_demand_by_month(month, use_cache=True, cet_index=True):
    period_start = f'{month}-01'
    period_end = (pd.Timestamp(period_start) + pd.offsets.MonthBegin(1) + pd.offsets.Day(-1)).strftime('%Y-%m-%d')
    day_ahead = (pd.Timestamp('today', tz='CET') + pd.offsets.Day(1)).strftime('%Y-%m-%d')
    period_end = day_ahead if period_end > day_ahead else period_end

    cache_key = f'/rte/demand/{month}'

    if use_cache:
        cached_df = load_cached_table_if_any(key=cache_key)
        if cached_df is not None and cached_df is not True:
            assert 'timestamp[CET]' == cached_df.index.name, Exception(f'index name "{cached_df.index.name}" != "timestamp[CET]"')
            if cet_index == False:
                cached_df = cached_df.tz_convert('UTC')
                cached_df.index.name = 'timestamp[UTC]'

            return cached_df

    hzlog(f'Downloading RTE demands for [{period_start}] => [{period_end}]')

    token_type, access_token = get_token()
    # Headers for the request
    headers = {
        'Authorization': f'{token_type} {access_token}',
        'Content-Type': 'application/json'
    }

    BASE_URL = 'https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term?'

    # Parameters for the request
    params = {
        'start_date': (pd.Timestamp(period_start) + pd.offsets.Day(-2)).strftime('%Y-%m-%dT00:00:00Z'),
        'end_date'  : pd.Timestamp(period_end).strftime('%Y-%m-%dT23:59:59Z'),
        'type': 'REALISED,ID,D-1,D-2',
    }

    # Make the API request
    response = requests.get(BASE_URL, headers=headers, params=params)

    if response.status_code == 200:
        dfs = []
        types = []
        for data in response.json()['short_term']:
            data_type = data['type']
            df = pd.DataFrame(data['values']).set_index('start_date')['value']
            df.index = pd.to_datetime(df.index, utc=True)
            dfs.append(df.resample('h').mean())
            types.append(data_type)
        df = pd.concat(dfs, axis=1, keys=types)

        expected_period_end = (pd.Timestamp(period_start) + pd.offsets.MonthEnd(1) + pd.offsets.Day(1)).normalize() - pd.offsets.Hour(1)
        expected_period_end = expected_period_end.tz_localize('CET')

        master_index = pd.date_range(start=period_start, end=expected_period_end.strftime('%Y-%m-%d %H:%M:00'), freq='h', tz='CET')
        df = df.reindex(master_index).dropna(how='all')
        df.index.name = 'timestamp[CET]'

        if df.index[-1] == expected_period_end:
            save_table_to_cache(key=cache_key, df=df)
        if cet_index == False:
            df.index = pd.DatetimeIndex(df.index).tz_convert('UTC')
            df.index.name = 'timestamp[UTC]'
        return df

    else:
        hzlog(f'Failed to download demand for [{period_start}] => [{period_end}] data: {response.status_code} - {response.text}')
        return pd.DataFrame()

def download_demand(period, use_cache=True, cet_index=True):
    assert len(period) == 7 or len(period) == 4, Exception(f'Invalid period format: {period}')
    if len(period) == 7:
        return download_demand_by_month(period, use_cache=use_cache, cet_index=cet_index)
    else:
        dfs = []
        this_month = pd.Timestamp('today').strftime('%Y-%m')
        for d in pd.date_range(f'{period}-01-01', f'{period}-12-31', freq='MS').strftime('%Y-%m'):
            if d > this_month:
                continue
            df = download_demand_by_month(d, use_cache=use_cache, cet_index=cet_index)
            if df.empty == False:
                dfs.append(df)
        return pd.concat(dfs)

def download_capacities_per_production_unit(month, use_cache=True, cet_index=True):

    assert len(month) == 7, Exception(f'Invalid period format: {month}')

    cache_key = f'/rte/capacities_per_production_unit/{month}'

    if use_cache:
        cached_df = load_cached_table_if_any(key=cache_key)
        if cached_df is not None:
            assert 'timestamp[CET]' == cached_df.index.name, Exception(f'index name "{cached_df.index.name}" != "timestamp[CET]"')
            if cet_index == False:
                cached_df = cached_df.tz_convert('UTC')
                cached_df.index.name = 'timestamp[UTC]'

            return cached_df

    period_start = pd.Timestamp(f'{month}-01').normalize()
    period_end   = period_start + pd.offsets.MonthBegin(1) - pd.offsets.Hour(1)
    period_index = pd.date_range(start=period_start, end=period_end, freq='h', tz='CET')
    hzlog(f'Downloading capacity per unit for [{period_start}] => [{period_end}]')

    # Base URL for the RTE API
    BASE_URL = 'https://digital.iservices.rte-france.com/open_api/generation_installed_capacities/v1/capacities_per_production_unit'

    token_type, access_token = get_token()
    # Headers for the request
    headers = {
        'Authorization': f'{token_type} {access_token}',
        'Content-Type': 'application/json'
    }

    # Parameters for the request
    params = {
        'start_date': (period_start + pd.offsets.Day(-1)).strftime('%Y-%m-%dT00:00:00Z'),
        'end_date'  : (period_end   + pd.offsets.Day( 1)).strftime('%Y-%m-%dT23:59:59Z'),
    }

    # Make the API request
    response = requests.get(BASE_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()

        unit_names = []
        unit_types = []
        unit_series = []
        for unit in data['capacities_per_production_unit']:
            unit_names.append(unit['production_unit']['name'])
            unit_types.append(unit['values'][0]['type'])
            series = pd.Series(index=period_index)
            unit_values = [(value['start_date'], value['installed_capacity']) for value in unit['values']]
            for series_start_date, installed_capacity in sorted(unit_values):
                series_start_date = pd.Timestamp(series_start_date).tz_convert('CET')
                series.loc[series.index >= series_start_date] = installed_capacity
            unit_series.append(series)
        df = pd.concat(unit_series, axis=1)
        df.columns = pd.MultiIndex.from_tuples(zip(unit_types, unit_names), names=['type', 'name'])
        df = df.loc[:,~df.columns.duplicated()].reindex(period_index)
        df.index = pd.DatetimeIndex(df.index, tz='UTC').tz_convert('CET')
        df.index.name = 'timestamp[CET]'

        if df.isna().sum(axis=1).max() < len(df.columns):
            save_table_to_cache(key=cache_key, df=df)

    else:
        hzlog(f'Failed to download installed capacity by day for {period_start} - {period_end} data: {response.status_code} - {response.text}')
        return None

    if cet_index == False:
        df.index = pd.DatetimeIndex(df.index).tz_convert('UTC')
        df.index.name = 'timestamp[UTC]'

    return df


def daily_backfill_run():
    pass
