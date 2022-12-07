
# this script tests the examples in OpenSRA

# base modules from Python
import argparse
import os
import subprocess
import time


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
    ex_list = [
        ##############################
        # above ground
        'above_ground_shakemap',
        'above_ground_ucerf',
        'above_ground_ucerf_with_filters',
        ##############################
        # below ground
        # -- landslide
        'below_ground_landslide_statewide-subset_level1_shakemap',
        'below_ground_landslide_statewide-subset_level1_ucerf',
        'below_ground_landslide_statewide-subset_level2_shakemap',
        # -- lateral spread
        'below_ground_lateral_spread_cpt_alameda_ucerf',
        'below_ground_lateral_spread_cpt_balboa_blvd_shakemap',
        # 'below_ground_lateral_spread_cpt_balboa_blvd_ucerf',
        # 'below_ground_lateral_spread_level1_balboa_blvd_shakemap',
        'below_ground_lateral_spread_level1_statewide-subset_shakemap',
        'below_ground_lateral_spread_level1_statewide-subset_ucerf',
        'below_ground_lateral_spread_level2_balboa_blvd_shakemap',
        # -- settlement
        ####'below_ground_settlement_cpt_alameda_shakemap',
        ####'below_ground_settlement_cpt_alameda_ucerf',
        ####'below_ground_settlement_cpt_balboa_blvd_shakemap',
        ####'below_ground_settlement_cpt_balboa_blvd_ucerf',
        ####'below_ground_settlement_level2_statewide-subset_shakemap',
        ####'below_ground_settlement_level2_statewide-subset_ucerf',
        # -- surface fault rupture
        # 'below_ground_fault_rupture_statewide-subset',
        ##############################
        # wells and caprocks
        'wells_caprocks_ucerf',
        'wells_caprocks_userdef_rupture',
    ]
    ex_wdir = [
        os.path.abspath(os.path.join(opensra_dir,'examples',each))
        for each in ex_list
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
    time_start = time.time()
    
    # loop through each example
    for i,each in enumerate(ex_wdir):
        basename = os.path.basename(each)
        
        # track messages
        output[basename] = {'preprocess': None,'opensra': None}
        err[basename] = {'preprocess': None,'opensra': None}
        rc[basename] = {'preprocess': None,'opensra': None}
        print(f'\n{i+1}. Running example: {basename}')
        
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