from typing_extensions import List
import pandas as pd
import paramiko
from io import StringIO
import os
import tempfile
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from base.utils.cache import load_cached_table_if_any, save_table_to_cache, query_tables_from_cache, delete_table_from_cache
from base.utils import auto_path_join, hzlog
from base.utils import runner
import functools

# SFTP server details
__hostname = "sftp-transparency.entsoe.eu"
__username = "haoxian@zhao.uk"
__password = "Hurricane8760!!"

def download_csv_from_sftp(remote_dir, cache_prefix):
    #local_dir = auto_path_join(__CACHE_ROOT, 'ActualGenerationOutputPerGenerationUnit_16.1.A_r2.1')
    #os.makedirs(local_dir, exist_ok=True)

    # Connect to the SFTP server
    sftp = transport = None
    try:
        transport = paramiko.Transport((__hostname, 22))
        transport.connect(username=__username, password=__password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # List files in the remote directory
        files = sftp.listdir_attr(remote_dir)
        files = sorted(files, key=lambda x: x.filename, reverse=True)

        # Process files
        for file_attr in files:
            file_timestamp = file_attr.st_mtime
            remote_file_path = auto_path_join(remote_dir, file_attr.filename)
            cache_name_prefix = f"{file_attr.filename.replace('.csv', '')}"
            cache_name = f'{cache_name_prefix}.{file_timestamp}'
            cache_key = auto_path_join(cache_prefix, cache_name)
                
            if 'filepart' in cache_key:
                continue

            exists = load_cached_table_if_any(cache_key, check_existence_only=True)
            if exists == True:
                continue

            temp_file_name = auto_path_join(tempfile.gettempdir(), os.urandom(24).hex())

            hzlog(f'Downloading {remote_file_path} to \n{temp_file_name}', level=0)
            sftp.get(remote_file_path, temp_file_name)
            df = pd.read_csv(temp_file_name, delimiter='\t')
            save_table_to_cache(cache_key, df)
            os.remove(temp_file_name)

            for stale_f in query_tables_from_cache(cache_prefix):
                if stale_f.startswith(cache_name_prefix) and stale_f != cache_name:
                    stale_f = auto_path_join(cache_prefix, stale_f)
                    hzlog(f'Removing stale cache file: {stale_f}')
                    delete_table_from_cache(stale_f)


    finally:
        if sftp is not None:
            sftp.close()

        if transport is not None:
            transport.close()


def download_ActualGenerationOutputPerGenerationUnit():
    download_csv_from_sftp("/TP_export/ActualGenerationOutputPerGenerationUnit_16.1.A_r2.1/", "/entsoe/sftp_actual_gen_by_units")

def download_InstalledCapacityProductionUnit():
    download_csv_from_sftp("/TP_export/InstalledCapacityProductionUnit_14.1.B/", "/entsoe/sftp_installed_capacity_by_units")

def download_UnavailabilityOfGenerationUnits():
    download_csv_from_sftp("/TP_export/UnavailabilityOfGenerationUnits_15.1.A_B/", '/entsoe/sftp_unavail_gen_by_units_ab')
    download_csv_from_sftp("/TP_export/UnavailabilityOfProductionUnits_15.1.C_D/", '/entsoe/sftp_unavail_gen_by_units_cd')

def download_InstalledGenerationCapacityAggregated():
    download_csv_from_sftp("/TP_export/InstalledGenerationCapacityAggregated_14.1.A/", '/entsoe/sftp_installed_capacity_aggregated')

def download_AggregatedGenerationPerType():
    download_csv_from_sftp("/TP_export/AggregatedGenerationPerType_16.1.B_C/", '/entsoe/sftp_aggregated_gen_per_type')

def download_ActualTotalLoad():
    download_csv_from_sftp("/TP_export/ActualTotalLoad_6.1.A/", '/entsoe/sftp_actual_total_load')

def download_MonthAheadTotalLoadForecast():
    download_csv_from_sftp("/TP_export/MonthAheadTotalLoadForecast_6.1.D/", '/entsoe/sftp_month_ahead_total_load_forecast')

def download_PhysicalFlows():
    download_csv_from_sftp("/TP_export/PhysicalFlows_12.1.G/", '/entsoe/sftp_physical_flows_ab')

def download_CommercialSchedules():
    download_csv_from_sftp("/TP_export/CommercialSchedules_12.1.F_r3/", '/entsoe/sftp_commercial_schedules')

def query_ActualGenerationOutputPerGenerationUnit(period, cet_index=True, consumption=False):
    caches = query_tables_from_cache('/entsoe/sftp_actual_gen_by_units')
    prefix = period.replace('-', '_')
    caches_in_scope = pd.DataFrame([(c[:-10], int(c[-10:])) for c in caches if c.startswith(prefix)], columns=pd.Index(['key', 'timestamp']))
    caches_in_scope = caches_in_scope.groupby('key').max().reset_index()
    df_list = []
    for key, timestamp in caches_in_scope[['key', 'timestamp']].values:
        print(key, timestamp)
        cache_key = auto_path_join('/entsoe/sftp_actual_gen_by_units', f'{key}{timestamp}')
        df = load_cached_table_if_any(cache_key)
        if df is not None:
            try:
                df = df.rename(columns={
                    'ProductionType'            : 'GenerationUnitType',
                    'PowerSystemResourceName'   : 'GenerationUnitName',
                    'ActualGenerationOutput'    : 'ActualGenerationOutput(MW)'
                })

                df['GenerationUnitType'] = df['GenerationUnitType'].apply(lambda x: x.strip() if isinstance(x, str) else x)
                df['GenerationUnitName'] = df['GenerationUnitName'].apply(lambda x: x.strip() if isinstance(x, str) else x)
                df['MapCode'] = df['MapCode'].apply(lambda x: x.strip() if isinstance(x, str) else x)
            except:
                print(df)
                raise

            df_list.append(df)
    if len(df_list) == 0:
        return None
    #df = pd.concat(df_list).pivot_table(index='DateTime', columns=['MapCode', 'ProductionType', 'PowerSystemResourceName'], values='ActualGenerationOutput')
    df = pd.concat(df_list).pivot_table(index='DateTime (UTC)', columns=['MapCode', 'GenerationUnitType', 'GenerationUnitName'], values='ActualGenerationOutput(MW)')
    
    if cet_index:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('CET')
        df.index.name = 'DateTime[CET]'
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'DateTime[UTC]'
    return df.resample('h').mean()

def normalise_index(df, cet_index):
    if cet_index:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('CET')
        df.index.name = 'DateTime[CET]'
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'DateTime[UTC]'
    return df


def query_ActualGenerationOutputPerGenerationType(period, cet_index=True, consumption=False):
    caches = query_tables_from_cache('/entsoe/sftp_aggregated_gen_per_type')
    prefix = period.replace('-', '_')
    caches_in_scope = pd.DataFrame([(c[:-10], int(c[-10:])) for c in caches if c.startswith(prefix)], columns=pd.Index(['key', 'timestamp']))
    caches_in_scope = caches_in_scope.groupby('key').max().reset_index()
    df_list_60m = []
    df_list_15m = []
    df_list_30m = []

    df_list_15m_sum = []
    df_list_30m_sum = []



    for key, timestamp in caches_in_scope[['key', 'timestamp']].values:
        print(key, timestamp)
        cache_key = auto_path_join('/entsoe/sftp_aggregated_gen_per_type', f'{key}{timestamp}')
        df = load_cached_table_if_any(cache_key)
        if df is not None:
            try:
                df = df.rename(columns={
                    'ProductionType'            : 'GenerationType',
                    'ActualGenerationOutput'    : 'ActualGenerationOutput(MW)'
                })

                df['GenerationType'] = df['GenerationType'].apply(lambda x: x.strip() if isinstance(x, str) else x)
                df['MapCode'] = df['MapCode'].apply(lambda x: x.strip() if isinstance(x, str) else x)

                df_60m = df.loc[df['ResolutionCode'] == 'PT60M']
                df_30m = df.loc[df['ResolutionCode'] == 'PT30M']
                df_15m = df.loc[df['ResolutionCode'] == 'PT15M']
            
            except:
                print(df)
                raise


            df_list_60m.append(df_60m)
            df_list_30m.append(df_30m.loc[df_30m['MapCode'] != 'IT'])      
            df_list_15m.append(df_15m.loc[df_15m['MapCode'] != 'IT'])

            df_list_15m_sum.append(df_30m.loc[df_30m['MapCode'] == 'IT'])      
            df_list_30m_sum.append(df_15m.loc[df_15m['MapCode'] == 'IT'])

    df_list = []

    if len(df_list_60m) >0:
        df_60m = pd.concat(df_list_60m).pivot_table(index='DateTime', columns=['MapCode', 'GenerationType'], values='ActualGenerationOutput(MW)')
        df_list.append(normalise_index(df_60m, cet_index=cet_index).resample('h').mean())

    if len(df_list_30m) >0:
        df_30m = pd.concat(df_list_30m).pivot_table(index='DateTime', columns=['MapCode', 'GenerationType'], values='ActualGenerationOutput(MW)')
        df_list.append(normalise_index(df_30m, cet_index=cet_index).resample('h').mean())

    if len(df_list_15m) >0:
        df_15m = pd.concat(df_list_15m).pivot_table(index='DateTime', columns=['MapCode', 'GenerationType'], values='ActualGenerationOutput(MW)')
        df_list.append(normalise_index(df_15m, cet_index=cet_index).resample('h').mean())

    if len(df_list_30m_sum) >0:
        df_30m_sum = pd.concat(df_list_30m_sum).pivot_table(index='DateTime', columns=['MapCode', 'GenerationType'], values='ActualGenerationOutput(MW)')
        df_list.append(normalise_index(df_30m_sum, cet_index=cet_index).resample('h').sum())

    if len(df_list_15m_sum) >0:
        df_15m_sum = pd.concat(df_list_15m_sum).pivot_table(index='DateTime', columns=['MapCode', 'GenerationType'], values='ActualGenerationOutput(MW)')
        df_list.append(normalise_index(df_15m_sum, cet_index=cet_index).resample('h').sum())

    if len(df_list) == 0:
        return None

    df = pd.concat(df_list)


    return df



def query_ActualTotalLoad(period, cet_index=True):
    period = period.replace('-', '_')
    df = load_latest_entsoe_tables(f'/entsoe/sftp_actual_total_load/{period}')
    if df is None:
        return pd.DataFrame()

    df = df.sort_values('UpdateTime').groupby(['DateTime', 'MapCode']).last().reset_index()
    df = df.pivot_table(index='DateTime', columns='MapCode', values='TotalLoadValue')
    df.columns.names = [None]
    if cet_index:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('CET')
        df.index.name = 'DateTime[CET]'
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'DateTime[UTC]'
    return df.resample('h').mean()



def query_commercial_schedules_da_cap(period, cet_index=True):
    period = period.replace('-', '_')
    df = load_latest_entsoe_tables(f'/entsoe/sftp_commercial_schedules/{period}')
    if df is None:
        return pd.DataFrame()

    df['MapCode'] = [f'{code_out}->{code_in}' for code_out, code_in in df[['OutMapCode', 'InMapCode']].values]
    df = df.sort_values('UpdateTime(UTC)').groupby(['DateTime(UTC)', 'MapCode']).last().reset_index()
    df = df.pivot_table(index='DateTime(UTC)', columns='MapCode', values='DayAheadCapacity[MW]')
    df.columns.names = [None]
    if cet_index:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('CET')
        df.index.name = 'DateTime[CET]'
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'DateTime[UTC]'
    return df.resample('h').mean()


def query_commercial_schedules_total_cap(period, cet_index=True):
    period = period.replace('-', '_')
    df = load_latest_entsoe_tables(f'/entsoe/sftp_commercial_schedules/{period}')
    if df is None:
        return pd.DataFrame()

    df['MapCode'] = [f'{code_out}->{code_in}' for code_out, code_in in df[['OutMapCode', 'InMapCode']].values]
    df = df.sort_values('UpdateTime(UTC)').groupby(['DateTime(UTC)', 'MapCode']).last().reset_index()
    df = df.pivot_table(index='DateTime(UTC)', columns='MapCode', values='TotalCapacity[MW]')
    df.columns.names = [None]
    if cet_index:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('CET')
        df.index.name = 'DateTime[CET]'
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'DateTime[UTC]'
    return df.resample('h').mean()



def compress_ActualTotalLoad(period_month):
    assert len(period_month) == 7, f"Invalid format ({period_month}) for period_month"
    folder = '/entsoe/sftp_actual_total_load/'
    file_prefix = period_month.replace('-', '_')
    table_names = [t for t in query_tables_from_cache(folder) if t.startswith(f'{file_prefix}_ActualTotalLoad')]
    names, ts = [], []
    for t in table_names:
        parts = t.split('.')
        names.append('.'.join(parts[:-1]))
        ts.append(parts[-1])

    for group_name, group in pd.DataFrame(list(zip(names, ts)), columns=['names', 'ts']).groupby('names'):
        tables = []
        files = []
        for name, ts in group.sort_values(['ts'], ascending=False)[['names', 'ts']].values:
            f_name = auto_path_join(folder, f'{name}.{ts}')
            t = load_cached_table_if_any(f_name)
            if t is not None:
                tables.append(t)
                files.append(f_name)
            else:
                hzlog(f'Cannot log cache: {f_name}')
        table = pd.concat(tables).drop_duplicates().reset_index(drop=True)
        compressed_f_name = auto_path_join(folder, f'{group_name}.0')
        save_table_to_cache(compressed_f_name, table)
        hzlog(f'Compressed cache saved: {compressed_f_name}')

        skipped = False
        hzlog(f'Deleting uncompressed caches for {group_name}')
        for f in sorted(files, reverse=True):
            if f == compressed_f_name:
                continue

            if skipped == False:
                hzlog(f'Skipping: {f}')
                skipped = True
                continue
            else:
                delete_table_from_cache(f)
                hzlog(f'uncompressed cache deleted: {f}')

def test():
    download_ActualGenerationOutputPerGenerationUnit()

def run_downloads(state=None):
    download_ActualGenerationOutputPerGenerationUnit()
    download_InstalledCapacityProductionUnit()
    download_InstalledGenerationCapacityAggregated()
    download_AggregatedGenerationPerType()
    download_ActualTotalLoad()
    download_MonthAheadTotalLoadForecast()
    download_PhysicalFlows()
    download_CommercialSchedules()
    return True



def load_latest_entsoe_tables(prefix):
    folder = os.path.dirname(prefix)
    file_prefix = os.path.basename(prefix)
    tables = [t for t in query_tables_from_cache(folder) if t.startswith(file_prefix)]
    assert all(len(t.split('.')[-1]) == 10 or t.split('.')[-1] == '0' for t in tables), Exception(f"Invalid tables found in {tables}")
    tables = [f'{folder}/{key}{time}' for key, time in (pd.DataFrame([(t[:-10], t[-10:]) for t in tables], columns=['key', 'time']).groupby('key').max().reset_index().values)]
    print(f'Loading tables: {tables}')
    tables = [load_cached_table_if_any(t) for t in tables]
    tables = [t for t in tables if t is not None]
    if len(tables) == 0:
        return None
    return pd.concat(tables)


run = functools.partial(runner, func=run_downloads, interval=60*10)
