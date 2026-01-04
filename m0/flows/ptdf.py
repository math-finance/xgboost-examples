import pandas as pd
import numpy as np 
from edge.models import get_all_borders
from base.fpc import baseload_list, baseload_download

__FALLBACK_PTDF_BOUNDS = {
    '2024-06-24':[]
}


def get_ptdf_bounds(ptdf_df):
    max_flows = {}
    regions = sorted((c for c in ptdf_df.columns if c.lower() != 'ram'))
    borders = get_all_borders()
    for t in ptdf_df.index.unique():
      max_flows[t] = {}
      ptdf_by_t = ptdf_df.loc[t]
    
      for i, lhs in enumerate(regions):
          for j, rhs in enumerate(regions):
              if i >= j:  # Skip duplicate pairs and self-pairs
                  continue
    
              border = tuple(sorted([lhs, rhs]))
              if border not in borders:
                  continue
              
              max_flow_pos = float('inf')  # lhs -> rhs
              max_flow_neg = float('inf')  # rhs -> lhs
    
              ram = ptdf_by_t['RAM']
              ptdf1 = ptdf_by_t[lhs]
              ptdf2 = ptdf_by_t[rhs]
    
              # Filter out lines with no remaining capacity or negligible PTDF difference
              valid_lines = (ram > 1e-6) & ((ptdf1 - ptdf2).abs() > 1e-8)
    
              if valid_lines.any():
                  ram_valid = ram[valid_lines]
                  net_ptdf_valid = (ptdf1 - ptdf2)[valid_lines]
                  flow_limits = ram_valid / net_ptdf_valid.abs()
    
                  # Separate constraints by flow direction
                  positive_constraints = net_ptdf_valid > 0
                  negative_constraints = net_ptdf_valid < 0
    
                  if positive_constraints.any():
                      max_flow_pos = min(flow_limits[positive_constraints])
    
                  if negative_constraints.any():
                      max_flow_neg = min(flow_limits[negative_constraints])
    
              # Store results
              pair_key = (lhs, rhs)
              reverse_key = (rhs, lhs)
    
              max_flows[t][pair_key] = max_flow_pos if max_flow_pos != float('inf') else None
              max_flows[t][reverse_key] = max_flow_neg if max_flow_neg != float('inf') else None
    
    
    ptdf_bounds = pd.DataFrame(index=pd.DatetimeIndex(sorted(max_flows.keys())))
    for idx in ptdf_bounds.index:
        for (lhs, rhs), bound in max_flows[ptdf_bounds.index[0]].items():
            ptdf_bounds.loc[idx, f'{lhs},{rhs}'] = np.ceil(bound) + 1

    return ptdf_bounds.astype(int)


def load_ptdf(date):
    ptdf_path_suffix = pd.Timestamp(date).tz_localize('CET').normalize().tz_convert('UTC').strftime('%Y-%m/%Y%m%dT%H0000')
    ptdf = baseload_download(f'/model_data/scenarios/shared/ptdfs/ptdfs_early_jao/{ptdf_path_suffix}', is_dataframe=True, cet_index=True)
    ptdf_columns = ['RAM', 'ALBE', 'ALDE', 'AT', 'BE', 'CZ', 'DE', 'FR', 'HR', 'HU', 'NL', 'PL', 'RO', 'SI', 'SK']
    ptdf = ptdf[ptdf_columns]
    return ptdf.copy(True)