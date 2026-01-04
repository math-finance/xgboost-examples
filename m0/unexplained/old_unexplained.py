from base.utils import auto_path_join
from base.utils.io import difc_read, difc_write
from base.fpc import baseload_list as bl, baseload_download as bd
from base import get_units_avail_gen
import pandas as pd
import re
import time

def get_temp(cc, scenario_path, asof=None):
    actl_temp =  bd(auto_path_join('/model_data/scenarios/shared/wind_solar_demand_mk/ecens_avg','0_backtest', f'fund_{cc.lower()}'),is_dataframe=True,cet_index=True)[('z_non_stack','tt')].to_frame(name='temp')
    fund_data = difc_read(auto_path_join(scenario_path, f'fund_{cc.lower()}'), is_dataframe=True, cet_index=True)

    if ('z_non_stack','tt') in fund_data.columns:
        fcst_temp = fund_data[('z_non_stack','tt')].to_frame(name='temp')
        return actl_temp, fcst_temp

    if asof is None:
        ts_from_path = re.search(r"\d{8}T*\d{6}", scenario_path)
        if ts_from_path is not None:
            ts = pd.to_datetime(ts_from_path[0]).tz_localize('UTC')
        else:
            ts = pd.Timestamp.utcnow()
    else:
        ts = asof

    month_prefix = ts.strftime('%Y-%m')
    folders = [f.split('/')[-1] for f in bl(auto_path_join('/model_data/scenarios/shared/wind_solar_demand_mk/ecens_avg', month_prefix))]
    invalid_instances = [f for f in folders if (f[8] != 'T' or len(f) != 15)]
    assert len(invalid_instances) ==0, Exception(f'Found invalid instances: {invalid_instances}')

    folders_in_scope = [f for f in folders if f.endswith('000000') and f <= ts.strftime('%Y%m%dT%H%M%S')]

    latest_in_scope = min(folders) if len(folders_in_scope) == 0 else max(folders_in_scope)

    prefix = pd.Timestamp(latest_in_scope).strftime('%Y-%m/%Y%m%dT%H%M%S')

    fcst_temp = bd(auto_path_join('/model_data/scenarios/shared/wind_solar_demand_mk/ecens_avg', prefix, f'fund_{cc.lower()}'),is_dataframe=True,cet_index=True)[('z_non_stack','tt')].to_frame(name='temp')

    if len(fcst_temp.index.intersection(fund_data.index)) < len(fund_data.index):
        prefix = (pd.to_datetime(latest_in_scope)-pd.DateOffset(days=1)).strftime('%Y-%m/%Y%m%dT%H%M%S')
        fcst_temp_previous = bd(auto_path_join('/model_data/scenarios/shared/wind_solar_demand_mk/ecens_avg',prefix,f'fund_{cc.lower()}'),is_dataframe=True,cet_index=True)[('z_non_stack','tt')].to_frame(name='temp')
        fcst_temp = fcst_temp.combine_first(fcst_temp_previous)

    return actl_temp, fcst_temp





def calc_unexplained_generic(cc, scenario_path):
    fund_data = bd(auto_path_join(scenario_path, f'fund_{cc.lower()}'),is_dataframe=True, cet_index=True)
    actl_temp, fcst_temp = get_temp(cc=cc, scenario_path=scenario_path)

    temperature = actl_temp.combine_first(fcst_temp)
    temperature['const'] = 1
    temperature['month'] = temperature.index.month
    temperature_d = temperature.resample('D').mean()

    fcst_cons =  fund_data[('res','cons')]
    fcst_solar = fund_data[('fixed','solar')]
    fcst_wind =  fund_data[('fixed','wind')]

    # Actuals
    units_avail, units_gen = get_units_avail_gen()

    units_avail = units_avail.loc[units_avail['Country']==cc.upper()]

    actgen_entsoe = get_entsoe_gen(units,start_utc='2019',end_utc =None).fillna(method='ffill',limit=24)
    for u in actgen_entsoe.columns:
        actgen_entsoe[u] = actgen_entsoe[u].clip(upper=actgen_entsoe[u].quantile(0.9999))
        actgen_entsoe[u] = actgen_entsoe[u].clip(lower=0)

    act_cons  = bd('/actual_data/consumption/mk',True,cet_index=True)
    act_bio   = bd('/actual_data/generation/mk/biomass',True,cet_index=True)
    act_hydro = bd('/actual_data/generation/mk/hydro',True,cet_index=True)
    act_solar = bd('/actual_data/generation/mk/solar',True,cet_index=True)[cc]
    act_wind  = bd('/actual_data/generation/mk/wind',True,cet_index=True)[cc]

    act_solar[c] = bd('/actual_data/generation/mk/solar',True,cet_index=True).dropna()[cc]
    act_wind[c]  = bd('/actual_data/generation/mk/wind',True,cet_index=True).dropna()[cc]

    act_flows = bd('/actual_data/borders/mk/flow_com',True,cet_index=True)

    act_netpos = pd.DataFrame(index = act_flows.index)
    act_netpos[c] = 0
    for l in act_flows.columns:
        c1,c2 = l.split(',')
        if c1==c:
            act_netpos[c]-= act_flows[l]

    unexplained_gen = act_cons[c]- actgen_entsoe.sum(axis=1) - \
                    act_solar[c] - act_wind[c] - act_netpos[c]
    unexplained_gen = pd.DataFrame(unexplained_gen)
    unexplained_gen.columns = ['unexplained']
    unexplained_gen_d = unexplained_gen.resample('D').mean()
    qmax = unexplained_gen_d['unexplained'].quantile(0.999)
    qlow = unexplained_gen_d['unexplained'].quantile(0.001)
    unexplained_gen_d = unexplained_gen_d.loc[unexplained_gen_d['unexplained']>=qlow]
    unexplained_gen_d = unexplained_gen_d.loc[unexplained_gen_d['unexplained']<=qmax]

    unexplained_gen['date'] = unexplained_gen.index.floor('D')
    unexplained_gen = unexplained_gen.loc[unexplained_gen.date.isin(unexplained_gen_d.index)]


    solar = (act_solar[c].combine_first(fcst_solar))#.combine_first(norm_solar[country])
    wind = (act_wind[c].combine_first(fcst_wind))#.combine_first(norm_wind[country])
    cons = (act_cons[c].combine_first(fcst_cons))#.combine_first(norm_cons[country])
    # hydro_ror = (act_hydro[f'{c.lower()}_ror'].combine_first(fcst_ror))
    # hydro_res = (act_hydro[f'{c.lower()}_res'].combine_first(fcst_res))
    # hydro_ps = (act_hydro[f'{c.lower()}_ps'].combine_first(fcst_ps))

    #-- Data preparation
    cutoff = (pd.to_datetime('today').floor('D')-pd.DateOffset(days=7)).tz_localize('CET')
    today = pd.Timestamp('today').tz_localize('CET').normalize()
    results = pd.DataFrame(index=cons.index)


    unexplained_gen_d['day'] =   unexplained_gen_d.index.dayofweek
    unexplained_gen_d['month'] = unexplained_gen_d.index.month

    temperature_d['day'] =   temperature_d.index.dayofweek
    temperature_d['month'] = temperature_d.index.month

    exog = temperature_d[['temp','const','month']]
    exog.columns = ['temperature','const','month']
    exog = exog.copy(True)
    exog['cons'] = cons.resample('D').mean()
    exog['wind'] = wind.resample('D').mean()
    exog['solar'] = solar.resample('D').mean()
    exog['weekend'] = exog.apply(lambda x: 1 if x.name.dayofweek>=5 else 0,axis=1)
    #exog['hydro_total'] = hydro['fr_total'].resample('D').mean()

    # exog['hydro_ror'] = hydro_ror.resample('D').mean()
    # exog['hydro_res'] = hydro_res.resample('D').mean()
    # exog['hydro_ps'] = hydro_ps.resample('D').mean()

    exog['winter'] = exog['month'].apply(lambda x: 1 if (x>=11 or x<=3) else 0)
    exog['winter_temperature'] = exog.apply(lambda x: x.temperature if (x.month>=11 or x.month<=3) else 0,axis=1)

    #-- Regression
    from sklearn.linear_model import LinearRegression
    cols = ['const','temperature','solar','wind','cons','weekend']

    for m in range (1,13):
        exog= exog[cols].dropna()
        endog = unexplained_gen_d.dropna()

        endog_m =endog.dropna()
        exog_m = exog.loc[exog.index.month==m]

        idx_reg= [t for t in exog_m.index if t in endog_m.index]
        exog_m = exog_m.loc[idx_reg]
        endog_m = endog_m.loc[idx_reg]

        reg = LinearRegression().fit(exog_m[cols],endog_m['unexplained'])

        idx_predict = exog.index.intersection(results.index)
        idx_predict = idx_predict[idx_predict.month==m]
        exog_m = exog.loc[idx_predict]

        results.loc[idx_predict,'reg']= reg.predict(exog_m.loc[idx_predict,cols])
        #results['lin_const']= res.predict(temperature_de_d[['const','DE']].rolling(1,center=True).mean())
    #     error=unexplained_gen_d['unexplained'].dropna()-results['reg'].dropna()
    #     error=error.dropna().loc[:cutoff]
    #     rmse=np.sqrt(np.mean((error*error)))
    #     print(rmse)

        results['month'] = results.index.month


def get_nl_unexplained(_FTP_PATH):
    c='NL'
    #-- Data collection
    fund = baseload_download(auto_path_join(_FTP_PATH,f'fund_{c.lower()}'),is_dataframe=True,cet_index=True)

    act_temperature, for_temperature = get_temp(c,fund,_FTP_PATH)
    #norm_temperature = get_data_mk(f'tt {c.lower()} con °c cet min15 n','temp',start=pd.to_datetime('today').floor('D'),end=pd.to_datetime('today').to_period('Y').end_time.floor('H')+pd.DateOffset(days=365*2),get_ens=False)

    temperature = act_temperature.combine_first(for_temperature)#.combine_first(norm_temperature)
    temperature['const'] = 1
    temperature['month'] = temperature.index.month
    temperature_d = temperature.resample('D').mean()

    fcst_cons =  fund[('res','cons')]
    fcst_solar = fund[('fixed','solar')]
    fcst_wind =  fund[('fixed','wind')]

    # Actuals
    units,units_gen = get_units()
    units = units.loc[units.Country==c]
    actgen_entsoe = get_entsoe_gen(units,start_utc='2019',end_utc =None).fillna(method='ffill',limit=24)
    for u in actgen_entsoe.columns:
        actgen_entsoe[u] = actgen_entsoe[u].clip(upper=actgen_entsoe[u].quantile(0.9999))
        actgen_entsoe[u] = actgen_entsoe[u].clip(lower=0)
    act_cons =  baseload_download('/actual_data/consumption/mk',True,cet_index=True)
    act_bio =   baseload_download('/actual_data/generation/mk/biomass',True,cet_index=True)
    act_hydro = baseload_download('/actual_data/generation/mk/hydro',True,cet_index=True)
    act_solar = baseload_download('/actual_data/generation/mk/solar',True,cet_index=True)[c]
    act_wind =  baseload_download('/actual_data/generation/mk/wind',True,cet_index=True)[c]

    act_solar[c] = baseload_download('/actual_data/generation/mk/solar',True,cet_index=True).dropna()[c]
    act_wind[c] =baseload_download('/actual_data/generation/mk/wind',True,cet_index=True).dropna()[c]

    act_flows = baseload_download('/actual_data/borders/mk/flow_com',True,cet_index=True)

    act_netpos = pd.DataFrame(index = act_flows.index)
    act_netpos[c] = 0
    for l in act_flows.columns:
        c1,c2 = l.split(',')
        if c1==c:
            act_netpos[c]-= act_flows[l]

    unexplained_gen = act_cons[c]- actgen_entsoe.sum(axis=1) - \
                      act_solar[c] - act_wind[c] - act_netpos[c]
    unexplained_gen = pd.DataFrame(unexplained_gen)
    unexplained_gen.columns = ['unexplained']
    unexplained_gen_d = unexplained_gen.resample('D').mean()
    qmax = unexplained_gen_d['unexplained'].quantile(0.999)
    qlow = unexplained_gen_d['unexplained'].quantile(0.001)
    unexplained_gen_d = unexplained_gen_d.loc[unexplained_gen_d['unexplained']>=qlow]
    unexplained_gen_d = unexplained_gen_d.loc[unexplained_gen_d['unexplained']<=qmax]

    unexplained_gen['date'] = unexplained_gen.index.floor('D')
    unexplained_gen = unexplained_gen.loc[unexplained_gen.date.isin(unexplained_gen_d.index)]


    solar = (act_solar[c].combine_first(fcst_solar))#.combine_first(norm_solar[country])
    wind = (act_wind[c].combine_first(fcst_wind))#.combine_first(norm_wind[country])
    cons = (act_cons[c].combine_first(fcst_cons))#.combine_first(norm_cons[country])
    # hydro_ror = (act_hydro[f'{c.lower()}_ror'].combine_first(fcst_ror))
    # hydro_res = (act_hydro[f'{c.lower()}_res'].combine_first(fcst_res))
    # hydro_ps = (act_hydro[f'{c.lower()}_ps'].combine_first(fcst_ps))

    #-- Data preparation
    cutoff = (pd.to_datetime('today').floor('D')-pd.DateOffset(days=7)).tz_localize('CET')
    today = pd.Timestamp('today').tz_localize('CET').normalize()
    results = pd.DataFrame(index=cons.index)


    unexplained_gen_d['day'] =   unexplained_gen_d.index.dayofweek
    unexplained_gen_d['month'] = unexplained_gen_d.index.month

    temperature_d['day'] =   temperature_d.index.dayofweek
    temperature_d['month'] = temperature_d.index.month

    exog = temperature_d[['temp','const','month']]
    exog.columns = ['temperature','const','month']
    exog = exog.copy(True)
    exog['cons'] = cons.resample('D').mean()
    exog['wind'] = wind.resample('D').mean()
    exog['solar'] = solar.resample('D').mean()
    exog['weekend'] = exog.apply(lambda x: 1 if x.name.dayofweek>=5 else 0,axis=1)
    #exog['hydro_total'] = hydro['fr_total'].resample('D').mean()

    # exog['hydro_ror'] = hydro_ror.resample('D').mean()
    # exog['hydro_res'] = hydro_res.resample('D').mean()
    # exog['hydro_ps'] = hydro_ps.resample('D').mean()

    exog['winter'] = exog['month'].apply(lambda x: 1 if (x>=11 or x<=3) else 0)
    exog['winter_temperature'] = exog.apply(lambda x: x.temperature if (x.month>=11 or x.month<=3) else 0,axis=1)

    #-- Regression
    from sklearn.linear_model import LinearRegression
    cols = ['const','temperature','solar','wind','cons','weekend']

    for m in range (1,13):
        exog= exog[cols].dropna()
        endog = unexplained_gen_d.dropna()

        endog_m =endog.dropna()
        exog_m = exog.loc[exog.index.month==m]

        idx_reg= [t for t in exog_m.index if t in endog_m.index]
        exog_m = exog_m.loc[idx_reg]
        endog_m = endog_m.loc[idx_reg]

        reg = LinearRegression().fit(exog_m[cols],endog_m['unexplained'])

        idx_predict = exog.index.intersection(results.index)
        idx_predict = idx_predict[idx_predict.month==m]
        exog_m = exog.loc[idx_predict]

        results.loc[idx_predict,'reg']= reg.predict(exog_m.loc[idx_predict,cols])
        #results['lin_const']= res.predict(temperature_de_d[['const','DE']].rolling(1,center=True).mean())
#     error=unexplained_gen_d['unexplained'].dropna()-results['reg'].dropna()
#     error=error.dropna().loc[:cutoff]
#     rmse=np.sqrt(np.mean((error*error)))
#     print(rmse)

    results['month'] = results.index.month

    # Smooth out
    for m in results.month.unique():
        maxp = unexplained_gen_d.loc[unexplained_gen_d.month==m,'unexplained'].quantile(0.995)
        minp = unexplained_gen_d.loc[unexplained_gen_d.month==m,'unexplained'].quantile(0.005)
        results.loc[results.month==m,'reg'] = results.loc[results.month==m,'reg'].clip(upper=maxp)
        results.loc[results.month==m,'reg'] = results.loc[results.month==m,'reg'].clip(lower=minp)

    results_d = results.resample('D').mean()
    results_d['error'] = results_d['reg'] - unexplained_gen_d['unexplained']
    results_d.loc[results_d['error']<=results_d['error'].quantile(0.005),'error'] = results_d.loc[results_d['error']<=0,'error'].mean()
    results_d.loc[results_d['error']>=results_d['error'].quantile(0.995),'error'] = results_d.loc[results_d['error']>=0,'error'].mean()
    fillerror = results_d['error'].fillna(0).shift(24).rolling(365).mean().loc[cutoff]

    # Smooth error
    for d in [7]:
        results_d.loc[:cutoff,'adj'] = results_d['error'].fillna(0).shift(1).rolling(d).mean()
        results_d.loc[cutoff:,'adj'] = results_d['error'].fillna(fillerror).shift(1).rolling(d).mean()
        results_d['reg_adj'] = results_d['reg'] - results_d['adj']
        results_d['error_adj'] = results_d['reg_adj'] - unexplained_gen_d['unexplained']
#         print(d, np.sqrt(np.mean(results_d['error_adj'].loc[:cutoff]**2)))

    results_d['reg_adj'] = results_d['reg']
    # Get hourly results
    results['reg_adj'] = results_d['reg_adj']
    results['reg_adj'] = results['reg_adj'].fillna(method='ffill',limit=25)

    unexplained_gen = unexplained_gen.drop('date',axis=1)

    unexplained_gen['hour'] =  unexplained_gen.index.hour
    unexplained_gen['day'] =   unexplained_gen.index.dayofweek
    unexplained_gen['month'] = unexplained_gen.index.month

    shapes = {}
    for m in range(1,13):
        shapes[m]={}
        for d in range(0,7):
            unexplained_gen_m = unexplained_gen.loc[(unexplained_gen.month>=m-1) & (unexplained_gen.month<=m+1)]
            unexplained_gen_m = unexplained_gen_m.loc[unexplained_gen_m.day==d].loc[:cutoff]
            shape = unexplained_gen_m.abs().groupby(unexplained_gen_m.hour).mean().iloc[:,0]/unexplained_gen_m.abs().groupby(unexplained_gen_m.index.date).mean().iloc[:,0].mean()
            shapes[m][d]=shape

    def apply_d_shape(x):
        m = x.month
        d =x.day
        #actgen_u_m = actgen_u.loc[actgen_u.month==m]
        shape = shapes[m]
        try:
            return x['predict']*shape.loc[x.hour]
        except Exception as e:
            for m_d in [0,1,1,-2,2]:
                for h_d in [0,1,-1,2,-2]:
                    try:
                        shape = shapes[m+m_d][d]
                        return x['reg_adj']*shape.loc[x.hour+h_d]
                    except Exception as e:
                        pass

    results['hour'] =  results.index.hour
    results['day'] =   results.index.dayofweek
    results['month'] = results.index.month

    results['reg_shaped'] = results.apply(lambda x: apply_d_shape(x),axis=1)

    assert results.loc[fund.index]['reg_shaped'].isna().sum()==0

    fund[('res','unexplained')] = results.loc[fund.index]['reg_shaped']
    fund.index = fund.index.tz_convert('UTC').tz_localize(None)
    fund.index.name = 'timestamp[UTC]'
    fund = fund.reset_index()
    assert(fund.isna().sum().sum()==0)
    baseload_upload(fund,key=auto_path_join(_FTP_PATH,f'fund_{c.lower()}'),is_dataframe=True)
    return 0
