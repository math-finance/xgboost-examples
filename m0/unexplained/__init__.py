import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import KBinsDiscretizer   # k-means binning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from edge.models.data import load_entsoe_actual_gen, load_entsoe_biomass_actual_gen
from base.fpc import baseload_download as bd
from base.utils import hzlog
from base.utils.io import difc_read as dr, auto_path_join
import pandas as pd
import numpy as np


FPC_ENTSOE_MAP = {
    'AT': ['AT'],
    'DE': ['DE_50HzT', 'DE_Amprion', 'DE_TenneT_GER', 'DE_TransnetBW'],
    'FR': ['FR'],
    'NL': ['NL'],
    'BE': ['BE'],
}


def shape_to_hourly(unexplained_d, bin_edges, cluster_shapes, idx_start, idx_end):
    out_vals = []
    for d, mean in unexplained_d.items():
        c        = np.digitize(mean, bin_edges[1:-1])       # which bin?
        shape24  = cluster_shapes.loc[c].values             # length-24
        out_vals.append(shape24 * mean)                     # scale up

    rslt = pd.Series(np.concatenate(out_vals), index=pd.date_range(idx_start, idx_end, freq='h'))
    rslt = rslt.tz_localize("CET", ambiguous="NaT", nonexistent="NaT").drop_duplicates().dropna()
    return rslt.resample('h').mean().ffill()

def train_unexplained_region_model(region, scenario_path, model_path, feature_cols, has_pumped_store=True):
    assert scenario_path.startswith('/model_data/scenarios/backtest/'), Exception('Training can only be done with backtest scenario')
    
    hzlog(f'Training unexplained for {region} started with data from {scenario_path}')
    entsoe_regions = FPC_ENTSOE_MAP[region.upper()]
    entsoe_act_gen_region = load_entsoe_actual_gen()[entsoe_regions]
    entsoe_act_gen_region.columns = [c[-1] for c in entsoe_act_gen_region.columns]

    training_fund = dr(auto_path_join(scenario_path, f'fund_{region.lower()}'), cet_index=True)
    training_idx = training_fund.index
    
    training_fund[('fixed', 'biomass')] = load_entsoe_biomass_actual_gen()[entsoe_regions].dropna().sum(axis=1).reindex(training_idx)
    #replacing biomass from backtest with entsoe biomass - eventually we might use entsoe biomasss in backtest
    
    ######## Consumption ########
    training_cons = training_fund[('rdl', 'cons')]
    hzlog(f'Consumption ({region}) {training_cons.mean().round(0).astype(int)}mw\n')

    ######## Renewables (Wind/Solar/Ror/Res) and Biomass ########
    
    training_fixed_gen = training_fund['fixed']
    
    hzlog(f'Fixed Gen ({region}) {int(round(training_fixed_gen.sum(axis=1).mean(), 0))}mw from: {sorted(training_fixed_gen.columns)}')
    for cf in training_fixed_gen.columns: 
        hzlog(f'Fixed Gen ({region}) {cf}: {training_fixed_gen[cf].mean().round(0).astype(int)}mw')
    
    training_fixed_gen = training_fixed_gen.sum(axis=1).reindex(training_idx)
    hzlog('\n')

    ######## Pumped Storage ######## 
    if has_pumped_store:   
        training_fund_calc = dr(auto_path_join(scenario_path, f'fund_{region.lower()}_calc'), cet_index=True).reindex(training_idx)
        training_ps_gen = training_fund_calc[[('fixed', 'ps_gen'), ('rdl', 'ps_con')]].sum(axis=1)
    else:
        training_ps_gen = 0


    ######## Thermal ########        
    units = dr(auto_path_join(scenario_path, f'units_{region.lower()}'))
    thermal = units.loc[units['BOX'] == 'THERMAL']
    thermal_names = sorted(thermal['TSOGenerationUnitName'].drop_duplicates().dropna().values)
    
    thermal_gen = entsoe_act_gen_region[[u for u in thermal_names if u in entsoe_act_gen_region]]
    hzlog(f'Thermal Gen ({region}) {thermal_gen.sum(axis=1).mean().round(0).astype(int)}mw from: \n{sorted(thermal_gen.columns)}\n')
    thermal_gen = thermal_gen.sum(axis=1).reindex(training_idx)

    ######## Must Run - portion of CHP with recognisable pattern ########        
    must_run_explained_gen = dr(auto_path_join(scenario_path, 'must_run_explained'))[region.lower()]
    hzlog(f'Must Run Gen ({region}) {must_run_explained_gen.sum(axis=1).mean().round(0).astype(int)}mw from: \n{sorted(must_run_explained_gen.columns)}\n')
    must_run_explained_gen = must_run_explained_gen.sum(axis=1).reindex(training_idx)

    ######## Net Flow ########        
    flows = bd('/actual_data/borders/mk/flow_com', is_dataframe=True, cet_index=True)
    export_region = flows[[f for f in flows if f.lower().startswith(f'{region.lower()},')]]
    export_region = export_region.sum(axis=1).reindex(training_idx)

    
    ######## Output, estimating the unexplained generation either from CHP or other source ######## 
    unexplained_gen_h = training_cons + export_region - training_fixed_gen - training_ps_gen - must_run_explained_gen - thermal_gen
    upper_h = unexplained_gen_h.quantile(0.95)
    lower_h = max(unexplained_gen_h.quantile(0.05), 0)
    unexplained_gen_h = unexplained_gen_h.clip(lower=lower_h, upper=upper_h)

    # ========== Clustering the hourly data into a group of shapes by daily means ==========:
    # 1. Collapse raw hourly readings → one value per exact clock-hour (handles dupes/gaps).
    # 2. Pivot to a 24-column “day × hour” matrix.
    # 3. Keep (or fill) only full 24-hour rows.
    # 4. Separate each row into:   daily_mean  and  relative_shape (row / mean).
    # 5. Auto-bin the daily_mean values into 10 k-means bins ⇒ data-driven level ranges.
    # 6. For each bin, average its relative_shape rows ⇒ a prototype 24-hour profile.
    # 7. At runtime: locate which bin a new daily mean falls into, scale that prototype by
    #    the mean, and you instantly have a 24-hour profile with minimal shape error.
    
    profiles = (
        unexplained_gen_h
          .to_frame('v')
          .assign(date=lambda s: s.index.floor('D'),
                  hour=lambda s: s.index.hour)
          .pivot_table(                     # ← lets you choose how to combine dups
              index='date',
              columns='hour',
              values='v',
              aggfunc='mean'    # or 'median', 'sum', 'first', ...
          )
    )

    profiles = profiles.dropna(how='any')
    daily_mean  = profiles.mean(axis=1)

    shape_only  = profiles.div(daily_mean, axis=0)
    
    est     = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
    labels  = est.fit_transform(daily_mean.to_numpy().reshape(-1, 1)).astype(int).ravel()
    
    shape_only['cluster'] = labels

    cluster_shapes = shape_only.groupby('cluster').mean()   # rows = 10, cols = 24
    bin_edges      = est.bin_edges_[0]
    
    unexplained_gen_d = unexplained_gen_h.resample('D').mean()
    unexplained_gen_d = pd.concat([
        unexplained_gen_d,
        training_cons.resample('d').mean(),
        training_fund[('z_non_stack', 'tt')].resample('d').mean(),
        training_fund[('z_non_stack', 'tt')].resample('d').min(),
        training_fund[('z_non_stack', 'tt')].resample('d').max(),
        training_fund[('fixed', 'wind')].resample('d').mean(),
        training_fund[('fixed', 'solar')].resample('d').mean(),
    ], axis=1, keys=['unexplained', 'cons', 'tt_avg', 'tt_min', 'tt_max', 'wnd', 'spv'])        
    unexplained_gen_d['day_of_week'] = unexplained_gen_d.index.dayofweek
    unexplained_gen_d['month'] = unexplained_gen_d.index.month

    target_cols  = ['unexplained']
    
    X = unexplained_gen_d[feature_cols]
    y = unexplained_gen_d[target_cols]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, shuffle=False   # keep temporal order
    )

    base = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, y_train)

    joblib.dump(
        {
            "model":          model,            # your fitted MultiOutputRegressor
            "bin_edges":      bin_edges,        # 1-D ndarray, len = 11
            "cluster_shapes": cluster_shapes    # 10 × 24 ndarray or DataFrame
        },
        model_path,          # one compact file
        compress=3                              # 0-9 → trade size vs CPU
    )

    # quick validation
    pred_val = pd.DataFrame(model.predict(X_val), index=X_val.index, columns=target_cols)
    mae = mean_absolute_error(y_val, pred_val)
    hzlog(f"Unexplained for {region} Validation MAE daily: {mae:.3f}")

    idx_start = pred_val.index.min().strftime('%Y-%m-%d 00:00:00')
    idx_end = pred_val.index.max().strftime('%Y-%m-%d 23:00:00')
    
    pred_val_h = shape_to_hourly(unexplained_d=pred_val.iloc[:, 0], 
                                 bin_edges=bin_edges, 
                                 cluster_shapes=cluster_shapes, 
                                 idx_start=idx_start, idx_end=idx_end)
    
    mae = mean_absolute_error(unexplained_gen_h.reindex(pred_val_h.index), pred_val_h)
    hzlog(f"Unexplained for {region} Validation MAE hourly: {mae:.3f}")


def calculate_unexplained_region(region, scenario_path, trained_model_path, feature_cols):

    trained_data    = joblib.load(trained_model_path)
    model           = trained_data["model"]
    bin_edges       = trained_data["bin_edges"]
    cluster_shapes  = pd.DataFrame(trained_data["cluster_shapes"],index=range(10), columns=range(24))

    fund = dr(auto_path_join(scenario_path, f'fund_{region.lower()}'), cet_index=True)
    idx = fund.index

    X_val = pd.concat([
        fund[('rdl', 'cons')].resample('d').mean(),
        fund[('z_non_stack', 'tt')].resample('d').mean(),
        fund[('z_non_stack', 'tt')].resample('d').min(),
        fund[('z_non_stack', 'tt')].resample('d').max(),
        fund[('fixed', 'wind')].resample('d').mean(),
        fund[('fixed', 'solar')].resample('d').mean(),
    ], axis=1, keys=['cons', 'tt_avg', 'tt_min', 'tt_max', 'wnd', 'spv'])        
    X_val['day_of_week'] = X_val.index.dayofweek
    X_val['month'] = X_val.index.month

    X_val[feature_cols]

    unexplained_d = pd.Series(model.predict(X_val).ravel())
    idx_start = idx.min().strftime('%Y-%m-%d 00:00:00')
    idx_end = idx.max().strftime('%Y-%m-%d 23:00:00')

    rslt = shape_to_hourly(unexplained_d=unexplained_d, 
                           bin_edges=bin_edges, 
                           cluster_shapes=cluster_shapes, 
                           idx_start=idx_start, idx_end=idx_end)
    rslt = rslt.reindex(idx)
    return rslt
