import pandas as pd
import functools

@functools.lru_cache(maxsize=2, typed=True)
def get_tenors(start='today', end='2027-12-31'):

    start = pd.Timestamp(start).strftime('%Y-%m-%d 00:00:00')
    end   = pd.Timestamp(end).strftime('%Y-%m-%d 23:00:00')


    tenors = {
        'yearly'   : {},
        'quarterly': {},
        'monthly'  : {},
        'weekly'   : {}
    }
    for y in range(int(start[:4]), int(end[:4])):
        tenors['yearly'][f'{y}'] = pd.date_range(f'{y}-01-01', f'{y}-12-31 23:00:00', freq='h', tz='CET')

        tenors['quarterly'][f'Q1{y % 2000}'] = pd.date_range(f'{y}-01-01', f'{y}-03-31 23:00:00', freq='h', tz='CET')
        tenors['quarterly'][f'Q2{y % 2000}'] = pd.date_range(f'{y}-04-01', f'{y}-06-30 23:00:00', freq='h', tz='CET')
        tenors['quarterly'][f'Q3{y % 2000}'] = pd.date_range(f'{y}-07-01', f'{y}-09-30 23:00:00', freq='h', tz='CET')
        tenors['quarterly'][f'Q4{y % 2000}'] = pd.date_range(f'{y}-10-01', f'{y}-12-31 23:00:00', freq='h', tz='CET')

        tenors['monthly'][f'Jan-{y % 2000}'] = pd.date_range(f'{y}-01-01', f'{y}-01-31 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Feb-{y % 2000}'] = pd.date_range(f'{y}-02-01', (pd.Timestamp(f'{y}-03-01') - pd.offsets.Day(1)).strftime('%Y-%m-%d 23:00:00'), freq='h', tz='CET')
        tenors['monthly'][f'Mar-{y % 2000}'] = pd.date_range(f'{y}-03-01', f'{y}-03-31 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Apr-{y % 2000}'] = pd.date_range(f'{y}-04-01', f'{y}-04-30 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'May-{y % 2000}'] = pd.date_range(f'{y}-05-01', f'{y}-05-31 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Jun-{y % 2000}'] = pd.date_range(f'{y}-06-01', f'{y}-06-30 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Jul-{y % 2000}'] = pd.date_range(f'{y}-07-01', f'{y}-07-31 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Aug-{y % 2000}'] = pd.date_range(f'{y}-08-01', f'{y}-08-31 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Sep-{y % 2000}'] = pd.date_range(f'{y}-09-01', f'{y}-09-30 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Oct-{y % 2000}'] = pd.date_range(f'{y}-10-01', f'{y}-10-31 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Nov-{y % 2000}'] = pd.date_range(f'{y}-11-01', f'{y}-11-30 23:00:00', freq='h', tz='CET')
        tenors['monthly'][f'Dec-{y % 2000}'] = pd.date_range(f'{y}-12-01', f'{y}-12-31 23:00:00', freq='h', tz='CET')

    df_index = pd.DataFrame(index=pd.date_range(start, end, freq='h', tz='CET'))
    df_index['week'] = df_index.index.isocalendar().week
    df_index_temp = df_index.resample('W-Sun').mean()
    df_index_temp['full'] = (df_index['week'].resample('W-Sun').count() >= (7*24 - 1))
    df_index_temp = df_index_temp.loc[df_index_temp['full'] == True]
    df_index_temp['weekly'] = [f'Wk{w:02}-{str(y)[-2:]}'for w, y in zip(df_index_temp['week'].values, df_index_temp.index.year)]
    df_index = df_index_temp[[c for c in df_index_temp.columns if c not in ['week', 'full']]].copy(True)

    for sun, contract in df_index['weekly'].items():
        tenors['weekly'][contract] = pd.date_range((sun - pd.offsets.Day(6)).normalize(), (sun + pd.offsets.Day(1)).normalize() - pd.offsets.Hour(1), freq='h')

    start = pd.Timestamp(start).tz_localize('CET')
    end = pd.Timestamp(end).tz_localize('CET')

    out_of_scope_tenors = []
    for freq in tenors.keys():
        for tenor, idx in tenors[freq].items():
            if idx.min() < start or idx.max() > end:
                out_of_scope_tenors.append((freq, tenor))

    for freq, tenor in out_of_scope_tenors:
        del tenors[freq][tenor]

    return tenors
