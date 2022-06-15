import numpy as np
import pandas as pd
import sys

# setup
fortran_file = r'C:\Users\barry\OneDrive\Desktop\CEC\09_NGAWest2.F'
subroutines = {
    'S09_BSSA_NGAWest2_2013': {
        'store_name': 'bssa14',
        'start_line': 49,
        'end_line': 386 
    },
    'S09_CB_NGAWest2_2013': {
        'store_name': 'cb14',
        'start_line': 802,
        'end_line': 913 
    },
    'S09_CY_NGAWest2_2013': {
        'store_name': 'cy14',
        'start_line': 1580,
        'end_line': 1813 
    },
    'S09_ASK_NGAWest2_2013': {
        'store_name': 'ask14',
        'start_line': 2267,
        'end_line': 2374 
    },
}

# read values
coeffs = {}
with open(fortran_file,'r') as f:
    lines = f.readlines()
    sub_count = 0
    for curr_sub in subroutines:
        store_name = subroutines[curr_sub]['store_name']
        start_line = subroutines[curr_sub]['start_line'] - 1 # python index starts at 0
        end_line = subroutines[curr_sub]['end_line']
        coeffs[store_name] = pd.DataFrame(None)
        param_val = None
        for i in range(start_line, end_line):
            line = lines[i].strip().replace(',', ' ').replace('/', '').split(' ')
            line = [val for val in line if val != '']
            if len(line) > 0 and not line[0].lower() == 'c':
                if 'Data' in line or 'data' in line:
                    if param_val is not None:
                        coeffs[store_name][param_name] = param_val
                    param_ind =  next(ind for ind,val in enumerate(line) if val.lower()=='data')+1
                    param_name = line[param_ind]
                    val_start = param_ind+1
                    param_val = [float(val) for val in line[val_start:]]
                else:
                    val_start = 1
                    param_val += [float(val) for val in line[val_start:]]
        coeffs[store_name][param_name] = param_val

# export to csv
for sub in coeffs:
    coeffs[sub].to_csv(
        os.path.join(
            r'C:\Users\barry\SlateGeotech\SlateDrive - _References\EQ Spectra\NGA-W2 Supplements',
            sub+'.csv'
        ),index=False
    )