# this script tests the methods in OpenSRA against pre-built results

#
import os
import importlib
import copy
import argparse


# -----------------------------------------------------------
# Main function
def main(detailed_verbose):
    
    # list of GMPES to test
    
    
    # list of methods in hierarchy
    methods_to_check = {
        'IM': {
            # 'gmc': {
            #     'ASK14',
            #     'BSSA14',
            #     'CB14',
            #     'CY14',
            # }
        },
        'EDP': {
            'liquefaction': {
                'ZhuEtal2015',
                'ZhuEtal2017',
                'Hazus2020',
                'Hazus2020_with_ZhuEtal2017',
            },
            'landslide': {
                'Jibson2007',
                'BrayMacedo2019'
            },
            'lateral_spread': {
                'Hazus2020',
                'YoudEtal2002'
            },
            'settlement': {
                'Hazus2020'
            },
            'surface_fault_rupture': {
                'Thompson2021',
                'Hazus2020',
                'WellsCoppersmith1994'
            },
            'wellhead_rotation': {
                'PantoliEtal2022',
            }
        },
        'DM': {
            # 'pipe_strain': {
                # 'BainEtal2022',
            # },
            'pipe_strain_lateral_spread': {
                # 'BainEtal2022_and_HutabaratEtal2022',
            },
            'pipe_strain_settlement': {
                'HutabaratEtal2022',
            },
            'pipe_strain_landslide': {
                # 'HutabaratEtal2022',
            },
            '_pipe_strain_base_models': {
                'BainEtal2022',
                'HutabaratEtal2022_Normal',
                'HutabaratEtal2022_Reverse',
                'HutabaratEtal2022_SSComp',
                'HutabaratEtal2022_SSTens_5to85',
                'HutabaratEtal2022_SSTens_85to90',
                'HutabaratEtal2022_SSTens',
            },
            'well_strain': {
                'SasakiEtal2022',
            },
            'well_moment': {
                'LuuEtal2022',
            },
            'wellhead_strain': {
                'PantoliEtal2022',
            },
            'vessel_moment_ratio': {
                'PantoliEtal2022',
            }
        },
        'DV': {
            'pipe_comp_rupture': {
                'BainEtal2022'
            },
            'pipe_tensile_rupture': {
                'BainEtal2022'
            },
            'pipe_tensile_leakage': {
                'BainEtal2022'
            },
            'wellhead_rupture': {
                'BainEtal2022'
            },
            'wellhead_leakage': {
                'BainEtal2022'
            },
            'caprock_leakage': {
                'ZhangEtal2022'
            },
            'well_rupture_shear': {
                'SasakiEtal2022'
            },
            'well_rupture_shaking': {
                'LuuEtal2022'
            },
            'vessel_rupture': {
                'PantoliEtal2022'
            }
        }
    }

    # loop through to run
    mods_dict = {}
    counter = 1
    for cat in methods_to_check:
        mods_dict[cat.lower()] = {}
        if cat == 'IM':
            if 'gmc' in methods_to_check[cat]:
                mods_dict[cat.lower()]['gmc'] = {
                    'module': importlib.import_module(f'src.{cat.lower()}.gmc'),
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
                    'module': importlib.import_module(f'src.{cat.lower()}.{haz}'),
                    'method': {}
                }
                for model in methods_to_check[cat][haz]:
                    mods_dict[cat.lower()][haz]['method'][model] = copy.deepcopy(getattr(mods_dict[cat.lower()][haz]['module'], model))
                    print(f'\nMethod {counter}: {cat.lower()} - {haz} - {model}')
                    mods_dict[cat.lower()][haz]['method'][model].run_check(verbose=True, detailed_verbose=detailed_verbose)
                    counter += 1
                

# -----------------------------------------------------------
# Proc main
if __name__ == "__main__":

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        description='Run test script for methods'
    )
    
    # Define arguments
    # detailed versose
    parser.add_argument('-d', '--detailed', help='Print detailed test messages', default=False)

    # Parse command line input
    args = parser.parse_args()
    
    # Call "Main" function
    main(
        detailed_verbose = args.detailed,
    )