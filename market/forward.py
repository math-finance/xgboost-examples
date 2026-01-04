from base.fpc import baseload_list, baseload_download
import pandas as pd
import functools

def get_full_month_px(px, head=None):
    months = px.index.strftime('%Y-%m').unique()
    full_months = []
    for m in months:
        expected_start = pd.Timestamp(f'{m}-01', tz=px.index.tz)
        expected_end = (expected_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(hours=1)
        
        month_data = px[expected_start:expected_end]
        if not month_data.empty and month_data.index[0] == expected_start and month_data.index[-1] == expected_end:
            full_months.append(m)

    px_months = px.resample('MS').mean()
    px_months.index = px_months.index.strftime('%Y-%m')
    px_months = px_months.loc[full_months].copy(True)
    px_months.sort_index(inplace=True)
    px_months.index.name = None
    #px_months.index = [f'M{idx:02}' for idx in range(1, len(px_months) + 1)]
    return px_months.head(head) if head else px_months


def get_full_week_px(px, head=None):
    px_w = pd.DataFrame(px)
    px_w['w'] = [f'{y}-Wk{w:02}' for w, y in zip(px_w.index.isocalendar().week, px_w.index.isocalendar().year)]

    full_weeks = []
    for week_label, week_data in px_w.groupby('w'):
        week_start = week_data.index[0].normalize()
        expected_week_start = week_start - pd.Timedelta(days=week_start.dayofweek)
        expected_week_end = (expected_week_start + pd.Timedelta(days=7)).normalize() - pd.Timedelta(hours=1)
        expected_idx = pd.date_range(expected_week_start, expected_week_end, freq='h', tz=week_data.index.tz)
        week_data = week_data.sort_index()
        if (len(week_data) == len(expected_idx) and week_data.index[0] == expected_idx[0] and week_data.index[-1] == expected_idx[-1]):
            full_weeks.append(week_label)

    px_weeks = px_w.groupby('w').mean()
    px_weeks = px_weeks.loc[full_weeks].copy(True)
    px_weeks.sort_index(inplace=True)
    px_weeks.index.name = None

    return px_weeks.head(head) if head else px_weeks

@functools.lru_cache(maxsize=5)
def load_forward_fuel(cob):
    root = '/model_data/scenarios/shared/fuels'
    month = pd.Timestamp(cob).strftime('%Y-%m')
    date = pd.Timestamp(cob).strftime('%Y%m%d')
    folders = [f for f in baseload_list(f'{root}/{month}') if f.startswith(f'{root}/{month}/{date}')]
    if folders:
        return baseload_download(f'{max(folders)}/fuels', is_dataframe=True, cet_index=True)
    else:
        raise Exception("failed to load for {cob}, {folder}")

@functools.lru_cache(maxsize=5)
def load_forward_power(cob):
    root = '/model_data/scenarios/shared/borders'
    month = pd.Timestamp(cob).strftime('%Y-%m')
    date = pd.Timestamp(cob).strftime('%Y%m%d')
    folders = [f for f in baseload_list(f'{root}/{month}') if f.startswith(f'{root}/{month}/{date}')]
    if folders:
        return baseload_download(f'{max(folders)}/px_borders', is_dataframe=True, cet_index=True)
    else:
        raise Exception("failed to load for {cob}")
    
    
@functools.lru_cache(maxsize=20)
def load_forward_power_full(cob, region):
    px = load_forward_power(cob)
    px_months = get_full_month_px(px[region])
    px_weeks = get_full_week_px(px[region])
    return px_months, px_weeks
