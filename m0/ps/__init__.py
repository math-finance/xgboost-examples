from base.fpc import baseload_list as bl, baseload_download as bd
from base.utils import auto_path_join, hzlog
from base.utils.io import difc_read, difc_write, difc_exist
import pandas as pd
import os

def __calc_ps_generic(scenario_path, cc, forec_recalc=False):
    hzlog(f'Calculating PS for {cc}')
    #-- Get data
    act_da = bd('/model_data/scenarios/shared/borders/0_backtest/px_borders', is_dataframe=True,  cet_index=True)[cc]
    act_cons = bd('/actual_data/consumption/mk',is_dataframe=True,cet_index=True)[cc]
    act_wind = bd('/actual_data/generation/mk/wind',is_dataframe=True,cet_index=True)[cc]
    act_solar = bd('/actual_data/generation/mk/solar',is_dataframe=True,cet_index=True)[cc]
    ps = bd('/actual_data/generation/mk/hydro', is_dataframe=True, cet_index=True)[f'{cc.lower()}_ps']

    fwd_da = difc_read(auto_path_join(scenario_path, 'px_borders'),is_dataframe=True,cet_index=True)[cc]
    fund_data_key = auto_path_join(scenario_path, f'fund_{cc.lower()}')
    fund_data_calc_key = auto_path_join(scenario_path, f'fund_{cc.lower()}_calc')

    if not forec_recalc and difc_exist(fund_data_calc_key, is_dataframe=True):
        return

    fund_data = difc_read(key=fund_data_key, is_dataframe=True, cet_index=True)

    da = act_da.combine_first(fwd_da)
    solar = (act_solar.combine_first(fund_data[('fixed','solar')]))
    wind = (act_wind.combine_first(fund_data[('fixed','wind')]))
    cons = (act_cons.combine_first(fund_data[('rdl','cons')]))

    #Combined/tranform data
    combined = pd.concat([da, ps, cons, wind, solar],axis=1)
    combined.columns=['px','ps','cons','wind','solar']
    combined['weekday'] = combined.index.dayofweek + 1
    combined.loc[(combined['weekday'] > 1) & (combined['weekday'] < 6), 'weekday'] = 3
    combined['hour'] = combined.index.hour
    combined['month'] = combined.index.month
    combined['year-week'] = combined.index.strftime('%Y-%W')

    combined.index.name='timestamp[CET]'
    combined = combined.reset_index()
    combined = combined.merge(combined.groupby(['year-week']).mean()[['px']].rename(columns={'px':'px_week'}),on='year-week',how='left')
    combined = combined.set_index('timestamp[CET]')

    combined['px_norm_1'] = combined['px'] / combined['px_week']
    combined['px_norm_2'] = combined['px'].dropna()/combined['px'].dropna().rolling(24).mean()
    combined['px_d'] = combined['px'] - combined['px'].dropna().rolling(24).mean()
    combined['const']=1

    combined['rdl'] = combined['cons'] - combined['solar'] - combined['wind']
    combined['rdl_sq'] = combined['rdl']**2

    from sklearn.linear_model import LinearRegression
    reg_save={}
    for m in range(1,13):
        reg_save[m]={}
        for d in [1,3,6,7]:
            reg_save[m][d]={}
            for h in range(24):
                data = combined.copy(True)
                data= data.loc[data.month==m]
                data = data.loc[data.weekday==d]
                data= data.loc[data.index.hour==h]

                exog = data[['px_norm_2','rdl','const','rdl_sq','px_d']].dropna()
                endog = data[['ps']].dropna()

                idx = [t for t in exog.index if t in endog.index]

                exog = exog.loc[idx]
                endog = endog.loc[idx]

                reg = LinearRegression().fit(exog,endog)

                reg_save[m][d][h]=reg.coef_

                fwd_idx = combined.loc[(combined.month==m)&(combined.weekday==d)&(combined.hour==h)].index
                fwd_idx = combined.loc[fwd_idx][['px_norm_2','rdl','const','rdl_sq','px_d']].dropna().index

                combined.loc[fwd_idx,'model']=reg.predict(combined.loc[fwd_idx][['px_norm_2','rdl','const','rdl_sq','px_d']])

    for m in combined.month.unique():
        maxp = combined.loc[combined.month==m,'ps'].quantile(0.995)
        minp = combined.loc[combined.month==m,'ps'].quantile(0.005)
        combined.loc[combined.month==m,'model'] = combined.loc[combined.month==m,'model'].clip(upper=maxp)
        combined.loc[combined.month==m,'model'] = combined.loc[combined.month==m,'model'].clip(lower=minp)

    model_rslt = combined.reindex(fund_data.index)['model']
    assert model_rslt.isna().sum()==0

    cols = pd.MultiIndex.from_tuples([('fixed', 'ps_gen'), ('rdl', 'ps_con')])

    fund_data_calc = pd.DataFrame(index=fund_data.index, columns=cols)
    fund_data_calc[('fixed','ps_gen')] = model_rslt.clip(lower=0)
    fund_data_calc[('rdl','ps_con')] = model_rslt.clip(upper=0)

    fund_data_calc.index = fund_data_calc.index.tz_convert('UTC').tz_localize(None)
    fund_data_calc.index.name = 'timestamp[UTC]'
    fund_data_calc = fund_data_calc.reset_index()
    hzlog(f'Saving Pumped Storage for {cc} to calculated fundamental data: {fund_data_calc_key}')
    difc_write(obj=fund_data_calc, key=fund_data_calc_key, is_dataframe=True)



def calc_ps_for_fund_de(scenario_path):
    __calc_ps_generic(scenario_path=scenario_path, cc='DE')

def calc_ps_for_fund_at(scenario_path):
    __calc_ps_generic(scenario_path=scenario_path, cc='AT')

def calc_ps_for_fund_fr(scenario_path):
    __calc_ps_generic(scenario_path=scenario_path, cc='FR')

def calc_ps_for_fund_be(scenario_path):
    __calc_ps_generic(scenario_path=scenario_path, cc='BE')
