import pandas as pd
import numpy as np
import os
import xgboost as xgb
from base.utils.io import difc_read as dr, difc_write as dw, auto_path_join
from base.utils import hzlog
from base.fpc import baseload_download as bd, baseload_list as bl
from edge.models.m0.must_run.nuke import calculate_must_run_for_nuke
from edge.models.data import load_entsoe_actual_gen
from edge.models.m0.unexplained.de import calculate_unexplained_de
from edge.models.m0.unexplained.nl import calculate_unexplained_nl
from edge.models.m0.unexplained.fr import calculate_unexplained_fr
from edge.models.m0.unexplained.at import calculate_unexplained_at
from edge.models.m0.unexplained.be import calculate_unexplained_be



def __calculate_must_run_flat(units, avail, entsoe_actual):

    units_flat = units.loc[units['BOX'] == 'FLAT']
    units_flat_names_map = dict([(tso_name, asset_name) for tso_name, asset_name in units_flat[['TSOGenerationUnitName', 'AssetName']].values])
    units_flat_names = list(sorted(units_flat_names_map.values()))
    
    entsoe_actual = entsoe_actual.rename(columns=units_flat_names_map)
    entsoe_actual_columns_set = set(entsoe_actual.columns)

    assert len(entsoe_actual_columns_set) == len(entsoe_actual.columns), Exception(
        f'Duplicated names/columns found in entsoe_actual data, plesae check, {sorted(entsoe_actual.columns)}'
    )
    
    
    hzlog(f'Calculating must run flat for {units_flat_names}')
    avail_columns_set = set(avail.columns)
    units_flat_names_set = set(units_flat_names)
    
    assert units_flat_names_set < avail_columns_set, Exception(
        f'Missing avail for {units_flat_names_set - avail_columns_set}'
    )
    
    assert units_flat_names_set < entsoe_actual_columns_set, Exception(
        f'Missing entsoe actual for {units_flat_names_set - entsoe_actual_columns_set}'
    )

    result = pd.DataFrame(index=avail.index, columns=units_flat_names)

    for u in units_flat_names:
        try:
            u_actgen = entsoe_actual[u].dropna()
            u_avail  = avail[u].dropna()

            training_idx = u_avail.index.intersection(u_actgen.index)
            training_cutoff = (u_actgen.index.max() - pd.offsets.Hour(23)).strftime('%Y-%m-%d 23:00:00')
            training_idx = training_idx[training_idx <= training_cutoff] 
            # this filter is necessary to make sure the index is always end with the last hour of the day, respecting the timezone
            
            u_capacity = units.loc[units['AssetName']==u,'Capacity'].values[0]

            u_avail_for_training = u_avail.loc[u_avail.rolling(8, center=True).mean().values>=0.8 * u_capacity]
            idx_avail = u_avail_for_training.reindex(training_idx).dropna().index
            ratio_to_avail = min(round((u_actgen.loc[idx_avail] / u_avail.loc[idx_avail]).mean(), 3), 1)
            result[u] = (u_avail * ratio_to_avail).clip(lower=0, upper=u_capacity).round(0).astype(int)

        except:
            raise
    return result

def __calculate_must_run_temperature(units, avail, entsoe_actual, region, temperature):
    # must run units that are sensitive to temperature
    units_temperature = units.loc[units['BOX'] == 'TEMPERATURE']
    units_temperature_names_map = dict([(tso_name, asset_name) for tso_name, asset_name in units_temperature[['TSOGenerationUnitName', 'AssetName']].values])
    units_temperature_names = list(sorted(units_temperature_names_map.values()))
    
    entsoe_actual = entsoe_actual.rename(columns=units_temperature_names_map)
    entsoe_actual_columns_set = set(entsoe_actual.columns)

    assert len(entsoe_actual_columns_set) == len(entsoe_actual.columns), Exception(
        f'Duplicated names/columns found in entsoe_actual data, plesae check'
    )
    
    hzlog(f'Calculating must run temperature for {units_temperature_names}')
    avail_columns_set = set(avail.columns)
    units_temperature_names_set = set(units_temperature_names)
    
    assert units_temperature_names_set < avail_columns_set, Exception(
        f'Missing avail for {units_temperature_names_set - avail_columns_set}'
    )
    
    assert units_temperature_names_set < entsoe_actual_columns_set, Exception(
        f'Missing entsoe actual for {units_temperature_names_set - entsoe_actual_columns_set}'
    )


    temp_backtest = bd(auto_path_join('/model_data/scenarios/shared/wind_solar_demand_mk/ecens_avg', '0_backtest',f'fund_{region.lower()}'), is_dataframe=True,cet_index=True)[('z_non_stack','tt')]

    temperature_full = pd.DataFrame(temp_backtest.combine_first(temperature))
    temperature_full.columns = ['temperature']
    temperature_full['dayofweek'] = temperature_full.index.dayofweek
    temperature_full['hour'] = temperature_full.index.hour
    temperature_full['block'] = temperature_full.index.hour // 4
    temperature_full['month'] = temperature_full.index.month
    temperature = temperature_full.loc[temperature.index]

    result = pd.DataFrame(0, index=avail.index, columns=units_temperature_names)
    mdl_cache_suffix = f'{pd.Timestamp("today").strftime("%Y%m%d")}.ubj'
    for u in units_temperature_names:
        mdl_name = f"/tmp/xgb_mdl_{u}.{mdl_cache_suffix}"
        try:
            if not os.path.exists(mdl_name):
                u_actgen = entsoe_actual[u].dropna()
                u_avail  = avail[u].dropna()

                training_idx = u_avail.index.intersection(u_actgen.index)
                training_cutoff = (u_actgen.index.max() - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d 23:00:00')
                training_idx = training_idx[training_idx <= training_cutoff] 
                # this filter is necessary to make sure the index is always end with the last hour of the day, respecting the timezone

                u_capacity = units.loc[units['AssetName']==u,'Capacity'].values[0]
                u_avail_for_training = u_avail.loc[u_avail.rolling(8, center=True).mean().values>=0.8 * u_capacity]
                training_idx = u_avail_for_training.reindex(training_idx).dropna().index

                df_training = pd.concat([u_actgen.loc[training_idx], temperature_full.loc[training_idx]], axis=1)
                df_training.columns = ['actgen'] + list(temperature_full.columns)

                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )

                mdl = xgb_model.fit(df_training.drop(columns=['actgen']), df_training['actgen'])
                mdl.save_model(mdl_name)
                hzlog(f'Saved mdl to {mdl_name}')
            else:
                mdl = xgb.XGBRegressor()
                hzlog(f'Loading saved mdl {mdl_name}')
                mdl.load_model(mdl_name)

            result[u] = mdl.predict(temperature)

        except:
            raise

    return result

def __calculate_must_run_random(units, avail, entsoe_actual):

    units_random = units.loc[units['BOX'] == 'RANDOM']
    units_random_names_map = dict([(tso_name, asset_name) for tso_name, asset_name in units_random[['TSOGenerationUnitName', 'AssetName']].values])
    units_random_names = list(sorted(units_random_names_map.values()))
    

    entsoe_actual = entsoe_actual.rename(columns=units_random_names_map)
    entsoe_actual_columns_set = set(entsoe_actual.columns)

    assert len(entsoe_actual_columns_set) == len(entsoe_actual.columns), Exception(
        f'Duplicated names/columns found in entsoe_actual data, plesae check, {sorted(entsoe_actual.columns)}'
    )
    
    hzlog(f'Calculating must run random for {units_random_names}')
    avail_columns_set = set(avail.columns)
    units_random_names_set = set(units_random_names)
    
    assert units_random_names_set < avail_columns_set, Exception(
        f'Missing avail for {units_random_names_set - avail_columns_set}'
    )
    
    assert units_random_names_set < entsoe_actual_columns_set, Exception(
        f'Missing entsoe actual for {units_random_names_set - entsoe_actual_columns_set}'
    )

    result = pd.DataFrame(0, index=avail.index, columns=units_random_names)
    # ----------------------------------------------------------------------
    for u in units_random_names:
        try:
            # raw series
            u_actgen = entsoe_actual[u].dropna()
            u_avail  = avail[u].dropna()

            # ----------------- training window ------------------------------
            training_idx = u_actgen.index.intersection(u_avail.index)
            # make sure we end on the *last* hour of the last complete CET day
            training_cutoff = (u_actgen.index.max() - pd.offsets.Hour(23)).strftime('%Y-%m-%d 23:00:00')
            training_idx = training_idx[training_idx <= training_cutoff]

            if training_idx.empty:
                continue  # nothing we can learn from – leave zeros

            # keep only hours with plenty of availability
            capacity = units.loc[units['AssetName'] == u, 'Capacity'].values[0]
            u_avail_ok = u_avail.loc[u_avail.rolling(8, center=True).mean().values >= 0.8 * capacity]
            idx_train  = u_avail_ok.reindex(training_idx).dropna().index
            if idx_train.empty:
                continue

            # ratio (≤ 1) to learn
            ratio_series = (u_actgen.loc[idx_train] / u_avail.loc[idx_train]).clip(upper=1)
            df_ratio = ratio_series.to_frame('ratio')
            df_ratio['month'] = df_ratio.index.month
            df_ratio['day']   = df_ratio.index.dayofweek
            df_ratio['block'] = df_ratio.index.hour // 4   # 6 blocks per day

            # ------------- hierarchy of mean patterns -----------------------
            pat_lvl1 = df_ratio.groupby(['month', 'day', 'block'])['ratio'].mean()
            pat_lvl2 = df_ratio.groupby(['month', 'day'])['ratio'].mean()
            pat_lvl3 = df_ratio.groupby(['month'])['ratio'].mean()

            # ------------- project pattern onto the forecast horizon --------
            tmp = pd.DataFrame(index=avail.index)
            tmp['month'] = tmp.index.month
            tmp['day']   = tmp.index.dayofweek
            tmp['block'] = tmp.index.hour // 4

            tmp = tmp.join(pat_lvl1.rename('p1'), on=['month', 'day', 'block'])
            tmp = tmp.join(pat_lvl2.rename('p2'), on=['month', 'day'])
            tmp = tmp.join(pat_lvl3.rename('p3'), on=['month'])

            tmp['pattern'] = (
                tmp['p1']
                .combine_first(tmp['p2'])
                .combine_first(tmp['p3'])
                .fillna(0)         # if still NaN, assume 0 % must-run
            )

            # ------------- turn ratios into MW ------------------------------
            gen = (tmp['pattern'] * avail[u]).clip(lower=0, upper=capacity)
            result[u] = gen.round(0).astype(int)

        except Exception as exc:
            hzlog(f'Error while processing {u}: {exc}')
            raise

    return result

def calculate_must_run_all(scenario_path):
    
    FPC_ENTSOE_MAP = {
        'AT': ['AT'],
        'DE': ['DE_50HzT', 'DE_Amprion', 'DE_TenneT_GER', 'DE_TransnetBW'],
        'FR': ['FR'],
        'NL': ['NL'],
        'BE': ['BE'],
    }

    units = {}
    avail = {}
    
    must_run_explained= []
    idx_master = None
    for r, entsoe_r in FPC_ENTSOE_MAP.items():
        try:
            r = r.lower()
            units[r] = dr(auto_path_join(scenario_path, f'units_{r}'))
            avail[r] = dr(auto_path_join(scenario_path, f'avail_{r}'), cet_index=True)
            idx_master_r = avail[r].index

            if idx_master is None:
                idx_master = idx_master_r
            else:
                idx_master = idx_master.union(idx_master_r)

            entsoe_actual = load_entsoe_actual_gen()[entsoe_r]
            entsoe_actual = entsoe_actual[[c for c in entsoe_actual.columns if not c[1].startswith('Hydro')]]

            if r == 'nl':
                try:
                    amer_9_hardcoal = entsoe_actual['NL']['Fossil Hard coal']['Amer 9']
                    entsoe_actual['NL']['Biomass']['Amer 9'] = entsoe_actual['NL']['Biomass']['Amer 9'].combine_first(amer_9_hardcoal)
                    entsoe_actual.drop(columns=[('NL', 'Fossil Hard coal', 'Amer 9')], inplace=True)
                except:
                    raise

            entsoe_actual.columns = [c[2] for c in entsoe_actual.columns]

            must_run_flat = __calculate_must_run_flat(units=units[r], avail=avail[r], entsoe_actual=entsoe_actual)
            if must_run_flat.empty == False:
                must_run_flat.columns = pd.MultiIndex.from_tuples([(r, c) for c in must_run_flat.columns])
                must_run_explained.append(must_run_flat)
        
            must_run_random = __calculate_must_run_random(units=units[r], avail=avail[r], entsoe_actual=entsoe_actual)
            if must_run_random.empty == False:
                must_run_random.columns = pd.MultiIndex.from_tuples([(r, c) for c in must_run_random.columns])
                must_run_explained.append(must_run_random)

            temperature = dr(auto_path_join(scenario_path, f'fund_{r.lower()}'), is_dataframe=True, cet_index=True)[('z_non_stack','tt')]
            must_run_temperature = __calculate_must_run_temperature(units=units[r], avail=avail[r], entsoe_actual=entsoe_actual, region=r, temperature=temperature)
            if must_run_temperature.empty == False:
                must_run_temperature.columns = pd.MultiIndex.from_tuples([(r, c) for c in must_run_temperature.columns])
                must_run_explained.append(must_run_temperature)

            power_prices = dr(auto_path_join(scenario_path, f'px_borders'), is_dataframe=True, cet_index=True)
            must_run_nuke = calculate_must_run_for_nuke(units=units[r], avail=avail[r], entsoe_actual=entsoe_actual, region=r, idx_master=idx_master_r, power_prices=power_prices)
            if must_run_nuke.empty == False:
                must_run_nuke.columns = pd.MultiIndex.from_tuples([(r, c) for c in must_run_nuke.columns])
                must_run_explained.append(must_run_nuke)
        except:
            hzlog(f'Error while processing must run for region {r}: {entsoe_r}')
            raise 

    must_run_explained = pd.concat(must_run_explained, axis=1).reindex(idx_master)
    dw(must_run_explained, key=auto_path_join(scenario_path, 'must_run_explained'), is_dataframe=True)


    is_backtest = scenario_path.startswith('/model_data/scenarios/backtest/')
    unexplained_de = calculate_unexplained_de(scenario_path=scenario_path, training=(is_backtest==True))
    unexplained_at = calculate_unexplained_at(scenario_path=scenario_path, training=(is_backtest==True))
    unexplained_fr = calculate_unexplained_fr(scenario_path=scenario_path, training=(is_backtest==True))
    unexplained_nl = calculate_unexplained_nl(scenario_path=scenario_path, training=(is_backtest==True))
    unexplained_be = calculate_unexplained_be(scenario_path=scenario_path, training=(is_backtest==True))

    must_run = must_run_explained.copy(True)
    # Append unexplained gen/CHP to must_run
    must_run[('de', 'unexplained')] = unexplained_de.round(0)
    must_run[('fr', 'unexplained')] = unexplained_fr.round(0)
    must_run[('at', 'unexplained')] = unexplained_at.round(0)
    must_run[('be', 'unexplained')] = unexplained_be.round(0)
    must_run[('nl', 'unexplained')] = unexplained_nl.round(0)

    assert must_run.isna().sum().sum()==0, list((must_run.isna().sum() !=0 ).replace(False, np.nan).dropna().index)
    dw(must_run, key=auto_path_join(scenario_path, 'must_run_calc'), is_dataframe=True)




