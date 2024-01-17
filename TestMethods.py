# this script tests the methods in OpenSRA against pre-built results

#
import argparse
import ast
import copy
import glob
import importlib
import inspect
import os

# list of GMPES to test

# list of methods in hierarchy
methods_to_check = {
    'IM': {
        # 'gmc': [
        #     'ASK14',
        #     'BSSA14',
        #     'CB14',
        #     'CY14',
        # ]
    },
    'EDP': {
        'liquefaction': [
            # 'PGE2022',
            # 'ZhuEtal2015',
            # 'ZhuEtal2017',
            # 'Hazus2020',
            # 'Hazus2020_with_ZhuEtal2017',
        ],
        'landslide': [
            # 'Jibson2007',
            # 'BrayMacedo2019'
        ],
        'lateral_spread': [
            # 'PGE2022',
            # 'Hazus2020',
            # 'YoudEtal2002'
        ],
        'settlement': [
            # 'PGE2022',
            # 'Hazus2020'
        ],
        'surface_fault_rupture': [
            # 'Thompson2021',
            # 'Hazus2020',
            # 'WellsCoppersmith1994'
        ],
        'wellhead_rotation': [
            # 'PantoliEtal2022',
        ]
    },
    'DM': {
        # 'pipe_strain': [
            # 'BainEtal2022',
        # ],
        'pipe_strain_lateral_spread': [
            # 'BainEtal2022_and_HutabaratEtal2022',
        ],
        'pipe_strain_settlement': [
            # 'HutabaratEtal2022',
        ],
        'pipe_strain_landslide': [
            # 'HutabaratEtal2022',
        ],
        '_pipe_strain_base': [
            # 'BainEtal2022',
            # 'HutabaratEtal2022_Normal',
            # 'HutabaratEtal2022_Reverse',
            # 'HutabaratEtal2022_SSComp',
            # 'HutabaratEtal2022_SSTens_5to85',
            # 'HutabaratEtal2022_SSTens_85to90',
            # 'HutabaratEtal2022_SSTens',
        ],
        'well_strain': [
            # 'SasakiEtal2022',
        ],
        'well_moment': [
            # 'LuuEtal2022',
        ],
        'wellhead_strain': [
            # 'PantoliEtal2022',
        ],
        'vessel_moment_ratio': [
            # 'PantoliEtal2022',
        ],
        'repair_rate': [
            # 'Hazus2020',
            # 'PGE2020',
        ],
    },
    'DV': {
        'pipe_comp_rupture': [
            # 'BainEtal2022'
        ],
        'pipe_tensile_rupture': [
            # 'BainEtal2022'
        ],
        'pipe_tensile_leakage': [
            # 'BainEtal2022'
        ],
        'wellhead_rupture': [
            # 'BainEtal2022'
        ],
        'wellhead_leakage': [
            # 'BainEtal2022'
        ],
        'caprock_leakage': [
            # 'ZhangEtal2022'
        ],
        'well_rupture_shear': [
            # 'SasakiEtal2022'
        ],
        'well_rupture_shaking': [
            # 'LuuEtal2022'
        ],
        'vessel_rupture': [
            # 'PantoliEtal2022'
        ],
        'pipe_rupture_using_repair_rate': [
            # 'Hazus2020',
        ],
    }
}

# -----------------------------------------------------------
# Main function
def main(check_nda_methods, detailed_verbose):
    
    # test methods
    _run_test(methods_to_check=methods_to_check, detailed_verbose=detailed_verbose)

    # if testing nda methods
    if check_nda_methods:
        print(f'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'Requested to check NDA methods...')
        # glob through nda folder to gather methods to test, if available
        if not 'nda' in os.listdir(os.getcwd()):
            print(f'\t- NDA directory does not exist (likely because you do not have permission to view information under NDA)')
        else:
            nda_dir = os.path.abspath(os.path.join(os.getcwd(),'nda'))
            entities = os.listdir(nda_dir)
            # for each entity
            sum_methods = 0
            for entity in entities:
                avail_py_files = glob.glob(os.path.join(nda_dir,entity,'src','*','*.py'))
                # create custom methods_to_check dict
                methods_to_check_nda = {}
                for f in avail_py_files:
                    norm_path = os.path.normpath(f)
                    path_split = norm_path.split(os.sep)
                    # gather PBEE category and add to methods dictionary
                    cat = path_split[-2].upper()
                    if not cat in methods_to_check_nda:
                        methods_to_check_nda[cat] = {}
                    if cat == 'IM':
                        print(f'\t- Testing for NDA IM methods to be implemented')
                    else:
                        # gather metric and add to methods dictionary
                        haz = path_split[-1].replace('.py','') # drop extension
                        haz_to_check = haz.replace('_','') # drop underscores
                        # check classes in .py file and gather all that do not share name as python file, and ignore BaseModel
                        mod = importlib.import_module(f'nda.{entity}.src.{cat.lower()}.{haz}')
                        methods_to_check_nda[cat][haz] = [
                            name
                            for name, _ in inspect.getmembers(mod, inspect.isclass)
                            if name != 'BaseModel' and name.lower() != haz_to_check
                        ]
                        sum_methods += len(methods_to_check_nda[cat][haz])
                if sum_methods > 0:
                    print(f'\n-----> {entity.upper()} <-----')
                    _run_test(
                        methods_to_check=methods_to_check_nda, detailed_verbose=detailed_verbose,
                        addl_path_for_nda_mod=f'nda.{entity}.'
                    )
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def _run_test(methods_to_check, detailed_verbose, addl_path_for_nda_mod=''):
    # loop through to run
    mods_dict = {}
    counter = 1
    for cat in methods_to_check:
        mods_dict[cat.lower()] = {}
        if cat == 'IM':
            if 'gmc' in methods_to_check[cat]:
                mods_dict[cat.lower()]['gmc'] = {
                    'module': importlib.import_module(f'{addl_path_for_nda_mod}src.{cat.lower()}.gmc'),
                    'method': {}
                }
                for gmpe in methods_to_check[cat]['gmc']:
                    mods_dict[cat.lower()]['gmc']['method'][gmpe] = copy.deepcopy(getattr(mods_dict[cat.lower()]['gmc']['module'], gmpe))
                    print(f'\nMethod {counter}: {cat.lower()} - gmc - {gmpe}')
                    mods_dict[cat.lower()]['gmc']['method'][gmpe].run_check()
                    counter += 1
        else:
            for haz in methods_to_check[cat]:
                mods_dict[cat.lower()][haz] = {
                    'module': importlib.import_module(f'{addl_path_for_nda_mod}src.{cat.lower()}.{haz}'),
                    'method': {}
                }
                for model in methods_to_check[cat][haz]:
                    mods_dict[cat.lower()][haz]['method'][model] = copy.deepcopy(getattr(mods_dict[cat.lower()][haz]['module'], model))
                    print(f'\nMethod {counter}: {cat.lower()} - {haz} - {model}')
                    if model == 'PGE2023' and haz != 'repair_rate':
                        dist_metric=['mean'] # sigmas vary with site
                    else:
                        dist_metric=['mean','sigma','sigma_mu'] # check sigmas
                    mods_dict[cat.lower()][haz]['method'][model].run_check(
                        verbose=True, detailed_verbose=detailed_verbose,
                        addl_path_for_nda_mod=addl_path_for_nda_mod.replace('.','\\'),
                        dist_metric=dist_metric
                    )
                    counter += 1

# -----------------------------------------------------------
# Proc main
if __name__ == "__main__":

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        description='Run test script for methods'
    )
    
    # Define arguments
    parser.add_argument('-c', '--checknda', help='Also check NDA methods', default=False)
    parser.add_argument('-d', '--detailed', help='Print detailed test messages', default=False)

    # Parse command line input
    args = parser.parse_args()
    
    # convert to boolean
    if isinstance(args.checknda,str):
        check_nda_methods = ast.literal_eval(args.checknda)
    else:
        check_nda_methods = args.checknda
    if isinstance(args.detailed,str):
        detailed_verbose = ast.literal_eval(args.detailed)
    else:
        detailed_verbose = args.detailed
    
    # Call "Main" function
    main(
        check_nda_methods = check_nda_methods,
        detailed_verbose = detailed_verbose,
    )