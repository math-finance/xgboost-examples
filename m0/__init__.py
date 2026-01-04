from base.utils import hzlog, auto_path_join
from base.utils.io import difc_read, difc_list
from edge.models.m0.ps import calc_ps_for_fund_de, calc_ps_for_fund_at, calc_ps_for_fund_be, calc_ps_for_fund_fr
from edge.models.m0.must_run import calculate_must_run_all
from edge.models.m0.solve import solve_weekly_cet



import pandas as pd

def run_compute(scenario_path):
    calc_ps_for_fund_de(scenario_path=scenario_path)
    calc_ps_for_fund_at(scenario_path=scenario_path)
    calc_ps_for_fund_fr(scenario_path=scenario_path)
    calc_ps_for_fund_be(scenario_path=scenario_path)

    #we have to make sure must_run is the last calculation to be done, as it uses the fund_calc data for unexplained gen/CHP
    calculate_must_run_all(scenario_path=scenario_path)



def get_model_inputs(scenario_path, with_ptdf):
    run_compute(scenario_path=scenario_path)
    #up to this point, no new data should be generated/saved to difc because must_run is the last calculation to be done

    fund_at = difc_read(auto_path_join(scenario_path, 'fund_at'), cet_index=True)
    fund_fr = difc_read(auto_path_join(scenario_path, 'fund_fr'), cet_index=True)
    fund_be = difc_read(auto_path_join(scenario_path, 'fund_be'), cet_index=True)
    fund_de = difc_read(auto_path_join(scenario_path, 'fund_de'), cet_index=True)
    fund_nl = difc_read(auto_path_join(scenario_path, 'fund_nl'), cet_index=True)

    master_index = fund_de.index

    fund_at_calc = difc_read(auto_path_join(scenario_path, f'fund_at_calc'), cet_index=True, not_exist_raise=False)
    fund_fr_calc = difc_read(auto_path_join(scenario_path, f'fund_fr_calc'), cet_index=True, not_exist_raise=False)
    fund_be_calc = difc_read(auto_path_join(scenario_path, f'fund_be_calc'), cet_index=True, not_exist_raise=False)
    fund_de_calc = difc_read(auto_path_join(scenario_path, f'fund_de_calc'), cet_index=True, not_exist_raise=False)
    fund_nl_calc = difc_read(auto_path_join(scenario_path, f'fund_nl_calc'), cet_index=True, not_exist_raise=False)

    fund_at = pd.concat([fund_at, fund_at_calc], axis=1).sort_index(axis=1, level=[0, 1]) 
    fund_fr = pd.concat([fund_fr, fund_fr_calc], axis=1).sort_index(axis=1, level=[0, 1]) 
    fund_be = pd.concat([fund_be, fund_be_calc], axis=1).sort_index(axis=1, level=[0, 1]) 
    fund_de = pd.concat([fund_de, fund_de_calc], axis=1).sort_index(axis=1, level=[0, 1]) 
    fund_nl = pd.concat([fund_nl, fund_nl_calc], axis=1).sort_index(axis=1, level=[0, 1]) 
    
    avail = {
        'AT': difc_read(auto_path_join(scenario_path, 'avail_at'), cet_index=True),
        'FR': difc_read(auto_path_join(scenario_path, 'avail_fr'), cet_index=True),
        'BE': difc_read(auto_path_join(scenario_path, 'avail_be'), cet_index=True),
        'DE': difc_read(auto_path_join(scenario_path, 'avail_de'), cet_index=True),
        'NL': difc_read(auto_path_join(scenario_path, 'avail_nl'), cet_index=True)
    }

    ntc         = difc_read(auto_path_join(scenario_path, 'ntc'), cet_index=True)
    f_flows     = difc_read(auto_path_join(scenario_path, 'f_flows'), cet_index=True)
    fuels       = difc_read(auto_path_join(scenario_path, 'fuels'), cet_index=True)
    px_borders  = difc_read(auto_path_join(scenario_path, 'px_borders'), cet_index=True)
    adjustments = difc_read(auto_path_join(scenario_path, 'adjustments'))

    units = {
        'AT': difc_read(auto_path_join(scenario_path, 'units_at')),
        'FR': difc_read(auto_path_join(scenario_path, 'units_fr')),
        'BE': difc_read(auto_path_join(scenario_path, 'units_be')),
        'DE': difc_read(auto_path_join(scenario_path, 'units_de')),
        'NL': difc_read(auto_path_join(scenario_path, 'units_nl')),
    }

    if with_ptdf:
        ptdf_mapping_cet = difc_read(auto_path_join(scenario_path, 'ptdf_mapping_cet'))
    else:
        ptdf_mapping_cet = None

    must_run = difc_read(auto_path_join(scenario_path, 'must_run_calc'))

    # Append must_run units
    for r, df in [('at', fund_at), ('de', fund_de), ('fr', fund_fr), ('nl', fund_nl), ('be', fund_be), ]:
        if 'fixed' in df.columns.get_level_values(0):
            for t in df['fixed'].columns:
                assetname = f'{r}_{t}'
                must_run[(r, t)] = df['fixed'][t]
                avail[f'{r.upper()}'][assetname] = 100e3

    ###### NOW transform the data into a format that can be acceptted by solves  ######

    u_coal = {}
    u_coal['DE'] = units['DE'].loc[units['DE'].Fuel=='Coal']
    u_coal['AT'] = units['AT'].loc[units['AT'].Fuel=='Coal']
    u_coal['NL'] = units['NL'].loc[units['NL'].Fuel=='Coal']
    u_coal['FR'] = units['FR'].loc[units['FR'].Fuel=='Coal']
    u_coal['BE'] = units['BE'].loc[units['BE'].Fuel=='Coal']

    u_lignite = {}
    u_lignite['DE'] = units['DE'].loc[units['DE'].Fuel=='Lignite']
    u_lignite['AT'] = units['AT'].loc[units['AT'].Fuel=='Lignite']
    u_lignite['NL'] = units['NL'].loc[units['NL'].Fuel=='Lignite']
    u_lignite['FR'] = units['FR'].loc[units['FR'].Fuel=='Lignite']
    u_lignite['BE'] = units['BE'].loc[units['BE'].Fuel=='Lignite']

    u_gas = {}
    u_gas['DE'] = units['DE'].loc[units['DE'].Fuel=='Gas']
    u_gas['AT'] = units['AT'].loc[units['AT'].Fuel=='Gas']
    u_gas['NL'] = units['NL'].loc[units['NL'].Fuel=='Gas']
    u_gas['FR'] = units['FR'].loc[units['FR'].Fuel=='Gas']
    u_gas['BE'] = units['BE'].loc[units['BE'].Fuel=='Gas']

    u_oil = {}
    u_oil['DE'] = units['DE'].loc[units['DE'].Fuel=='Oil']
    u_oil['AT'] = units['AT'].loc[units['AT'].Fuel=='Oil']
    u_oil['NL'] = units['NL'].loc[units['NL'].Fuel=='Oil']
    u_oil['FR'] = units['FR'].loc[units['FR'].Fuel=='Oil']
    u_oil['BE'] = units['BE'].loc[units['BE'].Fuel=='Oil']

    u_fixed = {}
    u_fixed['DE'] = units['DE'].loc[units['DE'].Fuel.isin(['Fixed','Nuclear'])]
    u_fixed['AT'] = units['AT'].loc[units['AT'].Fuel.isin(['Fixed','Nuclear'])]
    u_fixed['NL'] = units['NL'].loc[units['NL'].Fuel.isin(['Fixed','Nuclear'])]
    u_fixed['FR'] = units['FR'].loc[units['FR'].Fuel.isin(['Fixed','Nuclear'])]
    u_fixed['BE'] = units['BE'].loc[units['BE'].Fuel.isin(['Fixed','Nuclear'])]

    #Compute residual
    residual = pd.DataFrame(index=master_index)
    residual['DE'] = fund_de[('rdl','cons')] - fund_de['rdl'].drop('cons',axis=1).sum(axis=1)
    residual['AT'] = fund_at[('rdl','cons')] - fund_at['rdl'].drop('cons',axis=1).sum(axis=1)
    residual['NL'] = fund_nl[('rdl','cons')] - fund_nl['rdl'].drop('cons',axis=1).sum(axis=1)
    residual['FR'] = fund_fr[('rdl','cons')] - fund_fr['rdl'].drop('cons',axis=1).sum(axis=1)
    residual['BE'] = fund_be[('rdl','cons')] - fund_be['rdl'].drop('cons',axis=1).sum(axis=1)
    residual = residual.dropna()

    return {
        'residual'          : residual,
        'avail'             : avail,
        'must_run'          : must_run,
        'ntc'               : ntc,
        'f_flows'           : f_flows,
        'px_borders'        : px_borders,
        'u_lignite'         : u_lignite,
        'u_coal'            : u_coal,
        'u_gas'             : u_gas,
        'u_oil'             : u_oil,
        'u_fixed'           : u_fixed,
        'fuels'             : fuels,
        'ptdf_mapping_cet'  : ptdf_mapping_cet,
    }


def run_scenario(scenario_path, with_ptdf=True):
    model_inputs = get_model_inputs(scenario_path, with_ptdf=with_ptdf)

    print(model_inputs['ntc'])

    return
    sol_parameters, solver_name = solve_weekly_cet(**model_inputs)
