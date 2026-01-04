import pandas as pd
import numpy as np
import functools
from base.fpc import baseload_list as bl, baseload_download as bd
from base.rte import download_actual_generation
from base.utils.cache import save_table_to_cache, load_cached_table_if_any

@functools.lru_cache()
def _load_rte_actual_gen_nuke(months):
    cache_f_key = f'/entsoe/tmp/actgen_rte_nuke_{min(months)}_{max(months)}'
    
    df = load_cached_table_if_any(key=cache_f_key)

    if df is not None:
        return df

    df = pd.concat([download_actual_generation(m, use_cache=True) for m in months])['NUCLEAR']
    save_table_to_cache(key=cache_f_key, df=df, auto_expire_in_sec=3600 * 12)

    return df



# ──────────────────────────────────────────────────────────────────────────────
# 1.  Stack definition – keep as (share, price_threshold)
# ──────────────────────────────────────────────────────────────────────────────
__FR_NUKE_STACK_PRICES = [
    (0.65, -300),  # 65 % runs if price > –300
    (0.12,   5),
    (0.06,   10),
    (0.05,   10),
    (0.05,   10),
    (0.03,   50),
    (0.02,  200),
    (0.02, 2000),
]
assert sum([x[0] for x in __FR_NUKE_STACK_PRICES]) == 1, sum([x[0] for x in __FR_NUKE_STACK_PRICES])

def _prepare_stack(stack):
    """Return two numpy arrays: thresholds (ascending) and their shares."""
    stack_sorted = sorted(stack, key=lambda x: x[1])      # sort by price
    shares      = np.array([s for s, _ in stack_sorted])  # fractions - already 0-1
    thresholds  = np.array([p for _, p in stack_sorted])  # € / MWh etc.
    return shares, thresholds

SHARES, THRESHOLDS = _prepare_stack(__FR_NUKE_STACK_PRICES)


def _allocate_top_down(avail_row: pd.Series, need: float) -> pd.Series:
    """
    Allocate `need` MW across units in `avail_row` by filling the biggest
    units first (block-dispatch).  Units stay either at 0 or at their
    full available capacity, except possibly the last unit, which is
    partially filled if necessary.

    Parameters
    ----------
    avail_row : pd.Series (one timestamp)
        Available capacity per unit (MW).
    need : float
        Target output for the fleet after applying the nuclear-stack share.

    Returns
    -------
    pd.Series with the same index as `avail_row` containing scheduled output.
    """
    if need <= 0 or avail_row.sum() == 0:
        return avail_row * 0        # nothing to run

    remaining = need
    schedule  = pd.Series(0.0, index=avail_row.index)

    # sort units by descending available capacity (biggest first)
    for unit, cap in avail_row.sort_values(ascending=False).items():
        if remaining <= 0:
            break
        commit = min(cap, remaining)
        schedule[unit] = commit
        remaining     -= commit

    return schedule


def apply_nuke_stack(
    power_prices: pd.Series,
    avail:        pd.DataFrame,
    target:       pd.Series,
    window: int = 3,
    centre: bool = True,
    allocation: str = "top_down",      # "top_down"  or "pro_rata"
):
    """
    Parameters
    ----------
    allocation :
        * "top_down" – fill largest available units first
        * "pro_rata" – original proportional scaling of all positive units
    Returns
    -------
    adjusted_target : pd.Series
    dispatch        : pd.DataFrame  (scheduled output per unit)
    """
    # --- 2a / 2b / 2c identical to the previous version ---------------------
    roll_price = power_prices.rolling(window, center=centre, min_periods=1).mean()
    mask       = roll_price.values.reshape(-1, 1) > THRESHOLDS.reshape(1, -1)
    share      = (mask * SHARES.reshape(1, -1)).sum(axis=1).clip(0, 1)
    adjusted_target = (target * share).rename("adjusted_target")

    # --- 2d  choose allocator ------------------------------------------------
    if allocation not in {"top_down", "pro_rata"}:
        raise ValueError("allocation must be 'top_down' or 'pro_rata'")

    dispatch_rows = []
    for ts, avail_row in avail.iterrows():
        need = adjusted_target.loc[ts]

        if allocation == "top_down":
            dispatch_rows.append(_allocate_top_down(avail_row, need))

        else:  # pro_rata: scale only the *positive* units, leave zeros untouched
            row_sum = avail_row[avail_row > 0].sum()
            if row_sum == 0:
                dispatch_rows.append(avail_row * 0)
            else:
                factor = min(need / row_sum, 1.0)   # never exceed physical max
                dispatch_rows.append(avail_row.where(avail_row == 0,
                                                     avail_row * factor))

    dispatch = pd.DataFrame(dispatch_rows)

    return adjusted_target, dispatch



def calculate_must_run_for_nuke(units, avail, entsoe_actual, region, idx_master, power_prices):
    region = region.lower()
    units_nuke = units.loc[units['Fuel'] == 'Nuclear']
    units_nuke_names_map = dict([(tso_name, asset_name) for tso_name, asset_name in units_nuke[['TSOGenerationUnitName', 'AssetName']].values])
    units_nuke_names = list(sorted(units_nuke_names_map.values()))

    avail_columns_set = set(avail.columns)
    units_nuke_names_set = set(units_nuke_names)
    
    assert units_nuke_names_set < avail_columns_set, Exception(
        f'Missing avail for {units_nuke_names_set - avail_columns_set}'
    )

    result = pd.DataFrame()

    if region == 'fr':
        months = sorted(set(idx_master.strftime('%Y-%m')))
        rte_actual = _load_rte_actual_gen_nuke(tuple(months))
        rte_actual = rte_actual#.head(-24*14)

        for u in units_nuke_names:
            try:
                u_must_run = pd.Series(np.nan, index=idx_master)
                if u in rte_actual.columns:
                    u_must_run = rte_actual[u].reindex(idx_master).combine_first(u_must_run)
                
                start, end = units_nuke.loc[units_nuke['AssetName'] == u][['Start', 'End']].values[0]

                if pd.isna(start) == False:
                    u_must_run.loc[u_must_run.index < pd.Timestamp(start).tz_localize('CET')] = np.nan
                if pd.isna(end) == False:
                    u_must_run.loc[u_must_run.index > pd.Timestamp(end).tz_localize('CET')]  = np.nan

                if u_must_run.dropna().empty == True:
                    continue

                result[u] = u_must_run
            except:
                print(region, u)
                raise
        
        idx_forward = idx_master[idx_master > result.dropna().index.max()]
        fg_avail = bd('/model_data/scenarios/shared/fergal_fr_nuke/availability_forecast', is_dataframe=True, cet_index=True)
        target = fg_avail.reindex(idx_forward).sum(axis=1)
        power_prices_fr = power_prices['FR'].reindex(idx_forward)
        avail_forward = avail[result.columns].reindex(idx_forward)

        __adjusted_target, dispatch = apply_nuke_stack(power_prices=power_prices_fr, avail=avail_forward, target=target, window=7) 
        dispatch.index = target.index
        result = dispatch.combine_first(result.ffill())
        result = result.reindex(idx_master)

    else:
        entsoe_actual = entsoe_actual#.head(-24*14)
        entsoe_actual = entsoe_actual.rename(columns=units_nuke_names_map)
        entsoe_actual_columns_set = set(entsoe_actual.columns)

        assert len(entsoe_actual_columns_set) == len(entsoe_actual.columns), Exception(
            f'Duplicated names/columns found in entsoe_actual data, plesae check'
        )


        for u in units_nuke_names:
            try:
                u_must_run = pd.Series(np.nan, index=idx_master)
                
                if u in avail.columns:
                    u_must_run = avail[u].reindex(idx_master).combine_first(u_must_run)
                
                if u in entsoe_actual.columns:
                    u_must_run = entsoe_actual[u].reindex(idx_master).combine_first(u_must_run)
                
                start, end = units_nuke.loc[units_nuke['AssetName'] == u][['Start', 'End']].values[0]
                if pd.isna(start) == False:
                    u_must_run.loc[u_must_run.index < pd.Timestamp(start).tz_localize('CET')] = np.nan
                if pd.isna(end) == False:
                    u_must_run.loc[u_must_run.index > pd.Timestamp(end).tz_localize('CET')]  = np.nan

                if u_must_run.dropna().empty == True:
                    continue

                result[u] = u_must_run.fillna(0)
            except:
                print(region, u)
                raise

    result = result.clip(lower=0.1)
    return result
