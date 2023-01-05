
# this script tests the examples in OpenSRA

# base modules from Python
import argparse
import os
import shutil
import subprocess
import time
import sys


# -----------------------------------------------------------
# Main function
def main():

    # get OpenSRA dir
    opensra_dir = os.getcwd()
    count = 0
    while not opensra_dir.endswith('OpenSRA'):
        opensra_dir = os.path.dirname(opensra_dir)
        # in case can't locate OpenSRA dir and goes into infinite loop
        if count>5:
            print('Cannot locate OpenSRA directory - contact dev.')
        count += 1
        
    # preprocess script fpath
    preprocess_py_fpath = os.path.abspath(os.path.join(opensra_dir,"Preprocess.py"))
    # opensra script fpath
    opensra_py_fpath = os.path.abspath(os.path.join(opensra_dir,"OpenSRA.py"))

    # example wdir
    ex_list = {
        ##############################
        # above ground
        'above_ground': [
            # 'shakemap_gis_input',
            # 'ucerf_csv_input',
            # 'ucerf_with_filters_csv_input',
        ],
        ##############################
        # below ground
        # -- landslide
        'below_ground_landslide': [
            # 'level1_balboablvd_with_inventory_shakemap_(fails_in_preprocess)',
            # 'level1_bayarea_from_csv_no_polygon_shakemap',
            # 'level1_bayarea_no_polygon_ucerf',
            # 'level1_bayarea_with_inventory_shakemap',
            # 'level1_bayarea_with_inventory_ucerf',
        ],
        # -- lateral spread
        'below_ground_lateral_spread': [
            'level1_balboablvd_ucerf',
            'level1_bayarea_shakemap_1989lomaprieta_user-specified_data',
            'level1_bayarea_shakemap_1989lomaprieta-scaledto0p5g',
            'level1_bayarea_shakemap_2014southnapa',
            # 'level1_bayarea_ucerf',
            'level2_balboa_blvd_shakemap_1971sanfernando',
            'level2_balboa_blvd_shakemap_1994northridge',
            'level2_balboa_blvd_ucerf',
            'level3_cpt_alameda_ucerf_(fails_in_preprocess)',
            'level3_cpt_balboablvd_shakemap_1971sanfernando',
            'level3_cpt_balboablvd_ucerf',
        ],
        # -- settlement
        'below_ground_settlement': [
            # 'level1_bayarea_shakemap',
            # 'level1_bayarea_ucerf',
            # 'level3_cpt_balboablvd_shakemap_(fails_in_preprocess)',
            # 'level3_cpt_balboablvd_ucerf_(fails_in_preprocess)',
        ],
        # -- surface fault rupture
        'below_ground_surface_fault_rupture': [
            # 'bayarea',
            # 'losangeles',
        ],
        ##############################
        # wells and caprocks
        'wells_caprocks': [
            # 'ucerf_csv_input',
            # 'userdef_rupture_gis_input',
        ]
    }
    ex_wdir = [
        os.path.abspath(os.path.join(opensra_dir,'examples',cat,fname))
        for cat in ex_list
        for fname in ex_list[cat]
    ]
    
    # mapping for scripts
    script_map = {
        'preprocess': 'preprocessing',
        'opensra': 'risk calculations',
    }
    
    # initialize
    output = {}
    err = {}
    rc = {}
    time_init = time.time()
    time_start = time.time()
    
    # loop through each example
    for i,each in enumerate(ex_wdir):
        catname = os.path.basename(os.path.dirname(each))
        basename = os.path.basename(each)
        
        # track messages
        output[basename] = {'preprocess': None,'opensra': None}
        err[basename] = {'preprocess': None,'opensra': None}
        rc[basename] = {'preprocess': None,'opensra': None}
        print(f'\n{i+1}. Running example: {catname} - {basename}')
        
        # loop through scripts
        for phase in script_map:
            # print(f'\t- running script for "{script_map[phase]}"...')
            print(f'\t- {script_map[phase]}...')

            # run subprocess
            p = subprocess.Popen(
                ['python', locals()[f'{phase}_py_fpath'], '-w', each],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            
            # get outputs and errors
            output_curr, err_curr = p.communicate()
            rc_curr = p.returncode
            # store output messages
            output[basename][phase] = output_curr
            err[basename][phase] = err_curr
            rc[basename][phase] = rc_curr
            
            # print time
            time_spent = time.time()-time_start
            if time_spent > 60:
                print(f'\t\t- run time: {round(time_spent/60,1)} min')
            else:
                print(f'\t\t- run time: {round(time_spent,1)} sec')
            time_start = time.time()

            # print errors if any
            print(f'\t\t- error messages:')
            if len(err_curr) == 0:
                print(f'\t\t\tnone')
            else:
                print(f'\t\t\t{err_curr}')
                        

    print("\n>>>>>>>>>>>>>>>>> Finished testing all examples\n")
    total_time_spent = time.time() - time_init
    if total_time_spent > 60:
        print(f'\t- total run time: {round(total_time_spent/60,1)} min')
    else:
        print(f'\t- total run time: {round(total_time_spent,1)} sec')

# -----------------------------------------------------------
# Proc main
if __name__ == "__main__":

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        description='Run test script for examples'
    )
    
    # Define arguments

    # Parse command line input
    args = parser.parse_args()
    
    # Call "Main" function
    main()