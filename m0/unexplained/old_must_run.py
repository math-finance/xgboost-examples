def get_must_run_pf(_FTP_PATH,nuke_scenario, asof=None):
    print('get_must_run_pf:', _FTP_PATH, nuke_scenario, asof)
    avail = {}
    countries_list = ['fr','de','be','nl','at']
    for c in countries_list:
        act_avail = baseload_download(f'/actual_data/availability/{c}',is_dataframe=True,cet_index=True)
        act_avail.columns = act_avail.columns.droplevel(0)
        for_avail = baseload_download(auto_path_join(_FTP_PATH,f'avail_{c}'), is_dataframe=True, cet_index=True)
        avail_c = act_avail.combine_first(for_avail)
        avail[c.upper()] = avail_c

    units_gen,units_avail = get_units()
    units = units_gen.loc[units_gen.BOX.isin(['TEMPERATURE', 'RANDOM', 'FLAT'])]
    actgen_entsoe = get_entsoe_gen(units,start_utc='2019',end_utc = None)
    #display(actgen_entsoe)
    countries_list = ['de','fr','be','nl','at']
    must_run_pf = pd.DataFrame(index = for_avail.index)
    for c in countries_list:
        cutoff = actgen_entsoe[c.upper()].dropna().index.max().floor('H')
        for index,row in units.loc[units.Country==c.upper()].iterrows():
            if row.BOX == 'FLAT':
                temp = get_prod_flat(row.AssetName,units,actgen_entsoe,avail[c.upper()],cutoff)
            if row.BOX == 'RANDOM':
                temp = get_prod_random(row.AssetName,units,actgen_entsoe,avail[c.upper()],cutoff)
            if row.BOX == 'TEMPERATURE':
                temp = get_prod_temperature(row.AssetName, units,actgen_entsoe, avail[c.upper()], cutoff, _FTP_PATH, asof=asof)
            if temp.loc[avail[c.upper()].index].isna().sum().sum()>0:
                print('error', row.AssetName,row.BOX)

            must_run_pf = must_run_pf.join(temp, how='left')

    # Nukes

    must_run_pf = get_must_run_nukes(_FTP_PATH, must_run_pf, nuke_scenario, asof=asof)

    nuke_names = ['BELLEVILLE 1', 'BELLEVILLE 2', 'CATTENOM 1', 'CATTENOM 2', 'CATTENOM 3', 'CATTENOM 4', 'CHINON 1', 'CHINON 2', 'CHINON 3', 'CHINON 4', 'CHOOZ 1', 'CHOOZ 2', 'CIVAUX 1', 'CIVAUX 2', 'CRUAS 1', 'CRUAS 2', 'CRUAS 3', 'CRUAS 4', 'DAMPIERRE 1', 'DAMPIERRE 2', 'DAMPIERRE 3', 'DAMPIERRE 4', 'FESSENHEIM 1', 'FESSENHEIM 2', 'FLAMANVILLE 1', 'FLAMANVILLE 2', 'GOLFECH 1', 'GOLFECH 2', 'GRAVELINES 1', 'GRAVELINES 2', 'GRAVELINES 3', 'GRAVELINES 4', 'GRAVELINES 5', 'GRAVELINES 6', 'BLAYAIS 1', 'BLAYAIS 2', 'BLAYAIS 3', 'BLAYAIS 4', 'BUGEY 2', 'BUGEY 3', 'BUGEY 4', 'BUGEY 5', 'TRICASTIN 1', 'TRICASTIN 2', 'TRICASTIN 3', 'TRICASTIN 4', 'NOGENT 1', 'NOGENT 2', 'PALUEL 1', 'PALUEL 2', 'PALUEL 3', 'PALUEL 4', 'PENLY 1', 'PENLY 2', 'ST ALBAN 1', 'ST ALBAN 2', 'ST LAURENT 1', 'ST LAURENT 2']
    nuke_names_subset = []
    for n in nuke_names:
        if not n in must_run_pf.columns:
            print(f'Missing data for Nuke: {n}')
        else:
            nuke_names_subset.append(n)
    nuke_names = nuke_names_subset

    try: 
        must_run_pf.loc['2024-04'] *= 0.93
    except:
        pass

    try:
        must_run_pf.loc['2024-05'] *= 0.91
    except:
        pass 

    try:
        must_run_pf.loc['2024-06'] *= 0.89
    except:
        pass
        
    print(must_run_pf.resample('D').mean().dropna()[nuke_names])
    #assert(must_run_pf.isna().sum().sum()==0), print('Nuke', must_run_pf.isna().sum(axis=0).sort_values())
    
    must_run_pf[nuke_names] = must_run_pf[nuke_names].fillna(method='ffill').fillna(method='bfill')
    #must_run_pf.loc['2023-12-12', nuke_names] = must_run_pf.loc['2023-12-10', nuke_names].values
    #must_run_pf.loc['2023-12-13', nuke_names] = must_run_pf.loc['2023-12-10', nuke_names].values

    must_run_pf.index = must_run_pf.index.tz_convert('UTC').tz_localize(None)
    must_run_pf.index.name = 'timestamp[UTC]'
    must_run_pf = must_run_pf.reset_index()
    
    assert must_run_pf.isna().sum().sum()==0, list((must_run_pf.isna().sum() !=0 ).replace(False, np.nan).dropna().index)
    baseload_upload(must_run_pf,key=auto_path_join(_FTP_PATH,f'must_run_pf'),is_dataframe=True)
    return 0

    return must_run_pf


def get_must_run_nukes(_FTP_PATH, must_run_pf, nuke_scenario, asof=None):
    #must_run_pf = baseload_download(auto_path_join(_FTP_PATH,f'must_run_pf'), is_dataframe=True, cet_index=True)
    #must_run_pf = pd.DataFrame(index = must_run_pf.index)
    
    for c in ['FR','BE','NL','BE','DE']:
        avail = baseload_download(auto_path_join(_FTP_PATH,f'avail_{c.lower()}'), is_dataframe=True,cet_index=True)
        
        if c=='FR' and not('backtest' in _FTP_PATH or 'backfit' in _FTP_PATH):
            ts = re.search(r"\d{8}T*\d{6}",_FTP_PATH)[0]
            fcst = get_ml_nuke_fcst(ts=ts, scenario=nuke_scenario, cet_index=True)

            must_run_pf = must_run_pf.join(fcst,how='left')

            if asof is None:
                ts = pd.to_datetime(ts)
            else:
                ts = asof

            #ts = pd.to_datetime('2023-10-31')
            #Backfill for pv6/pv9 from start of the week 
            backtest_folder = get_date_subfolder(key='/model_data/scenarios/backtest',date=ts)

            # Handle case when must_run_pf not in backtest folder yet 
            if auto_path_join(backtest_folder,'must_run_pf.parquet') in baseload_list(backtest_folder):
                backtest = baseload_download(auto_path_join(backtest_folder,'must_run_pf'),is_dataframe=True,cet_index=True)
            else:
                ts = pd.to_datetime(ts) -pd.DateOffset(days=1)
                #Backfill for pv6/pv9 from start of the week 
                backtest_folder = get_date_subfolder(key='/model_data/scenarios/backtest',date=ts)
                backtest = baseload_download(auto_path_join(backtest_folder,'must_run_pf'),is_dataframe=True,cet_index=True)

            for u in fcst.columns:
                must_run_pf[u] = must_run_pf[u].combine_first(backtest[u])

            must_run_pf = must_run_pf.fillna(method='ffill',limit=3)

        else:
            units = baseload_download(auto_path_join(_FTP_PATH,f'units_{c}'), is_dataframe=True)
            units = units.loc[units.Fuel=='Nuclear']

            if avail[units.AssetName].sum().sum()==0:
                pass
            else:
                t_act_gen_start = min(avail.index.min().tz_convert('UTC').tz_localize(None)-pd.Timedelta('7 days'),pd.to_datetime('today').floor('H')-pd.Timedelta('7 days'))
                t_act_gen_end = max(avail.index.max().tz_convert('UTC').tz_localize(None),pd.to_datetime('today').floor('H'))

                act_gen = get_entsoe_gen(units,t_act_gen_start,cet_index=True)
                act_avail = baseload_download(f'/actual_data/availability/{c.lower()}',is_dataframe=True,cet_index=True)

                ratio = act_gen.sum(axis=1)/ act_avail['Nuclear'].sum(axis=1)

                ratio = pd.DataFrame(ratio.dropna())
                must_run_pf = must_run_pf.copy(True)
                must_run_pf[units.AssetName] = 0

                for t in avail.index:
                    if t-pd.DateOffset(days=7) in ratio.index:
                        must_run_pf.loc[t,units.AssetName] = max(1,ratio.loc[t-pd.DateOffset(days=7),0]) * avail.loc[t,units.AssetName]
                    else:
                        must_run_pf.loc[t,units.AssetName] = 0.95 * avail.loc[t,units.AssetName]
                for u in units.AssetName:
                    must_run_pf[u] = must_run_pf[u].clip(upper = units.loc[units.AssetName ==u,'Capacity'].values[0], lower=0)
                #must_run_pf[units.AssetName]
            for u in units.AssetName:
                idx = avail.loc[avail[u]==0].index
                must_run_pf.loc[idx,u] = 0
    return must_run_pf





def get_prod_flat(unit,units,actgen_entsoe,avail,cutoff):
    temp = pd.DataFrame(index=avail.index)
    today = pd.Timestamp('today').tz_localize('CET').normalize()
    
    fuel = units.loc[units.AssetName==unit,'Fuel'].values[0]
    capacity = units.loc[units.AssetName==unit,'Capacity'].values[0]
    actgen_u = actgen_entsoe[actgen_entsoe.columns[actgen_entsoe.columns.get_level_values(2)==unit]].fillna(method='ffill',limit=24)
    avail_u = avail[[unit]]
    
    actgen_u = actgen_u.loc[:cutoff]
    avail_u = avail_u.loc[:cutoff]
    
    idx = avail_u.index.intersection(actgen_u.index).intersection(avail_u.index)
    avail_u = avail_u.loc[idx]
    actgen_u = actgen_u.loc[idx]

    idx_avail = avail_u.loc[idx].loc[avail_u.rolling(8,center=True).mean().values>=0.8*capacity].index

    temp = temp.copy(True)
    temp[unit] = actgen_u.loc[idx_avail].loc[:cutoff].iloc[:, 0] / avail_u.loc[idx_avail].loc[:cutoff].iloc[:, 0]

    gen_ratio = temp[[unit]].replace([np.inf,-np.inf],np.nan).dropna()
    gen_ratio_mean = gen_ratio.mean()

    hours_to_shift = (today - actgen_u.dropna().index[-1]).components
    hours_to_shift = hours_to_shift.days * 24 + hours_to_shift.hours
    hours_to_shift = max([14 * 24, (hours_to_shift//168+1)*168])

    delta = (gen_ratio - gen_ratio_mean).reindex(temp[unit].index).fillna(0)
    delta.index += pd.offsets.Hour(hours_to_shift)

    temp[unit] = (pd.concat([delta.loc[delta.index.hour == h].rolling(int(hours_to_shift/24)).mean() for h in range(24)]).reindex(delta.index) + gen_ratio_mean).combine_first(temp[[unit]]).iloc[:, 0]

    temp = temp.copy(True)
    avail_u = avail[[unit]]
    
    temp[unit] = temp[unit]*avail_u.iloc[:,0]
    temp[unit] = pd.concat([temp[unit],avail_u],axis=1).min(axis=1)
    return temp[[unit]]
