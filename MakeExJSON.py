import json
import os

# get list of folders with examples
ex_folders = []
for each in os.listdir('examples'):
    if os.path.isdir(os.path.join('examples',each)) and \
        each != 'superseded' and \
        each != 'other_data':
        # each != 'below_ground_surface_fault_rupture' and \
            ex_folders.append(each)

# setup
infra_map = {
    'above_ground': "Above Ground Components",
    'below_ground_landslide': "Below Ground - Landslide",
    'below_ground_lateral_spread': "Below Ground - Lateral Spread",
    'below_ground_settlement': "Below Ground - Liquefaction Induced Settlement",
    'below_ground_surface_fault_rupture': "Below Ground - Surface Fault Rupture",
    'wells_caprocks': "Wells and Caprocks",
    'generic_model': "Using Generic Models",
}
words_to_upper = ['cpt','ucerf','csv','gis']
words_to_skip = ['with','no','in','from']
words_to_map = {
    'shakemap': 'ShakeMap',
    'statewide-subset': 'Statewide-Subset',
    'bayarea': 'BayArea',
    'losangeles': 'LosAngeles',
    'balboablvd': 'BalboaBlvd',
    'user-specified': 'User-Specified',
    'ca': 'CA',
    'genmod': 'GenericModel in',
    'single': 'SingleIntegral',
    
    'im': 'IM',
    'edp': 'EDP',
    'dm': 'DM',
    'dv': 'DV',
    
    '1971sanfernando': '1971SanFernando',
    '1989lomaprieta': '1989LomaPrieta',
    '1989lomaprieta-scaledto0p5g': '1989LomaPrieta-ScaledTo0p5g',
    '1994northridge': '1994Northridge',
    '2014southnapa': '2014SouthNapa',
}
to_skip = [
    'level1_bayarea_from_csv_no_polygon_shakemaps',
]


# go through each file in example and modify name
out = {}
for each in ex_folders:
    counter = 0
    infra_name = infra_map[each]
    print(infra_name)
    curr_examples = os.listdir(os.path.join('examples',each))
    ex_list = []
    for ex_name in curr_examples:
        if not ex_name in to_skip:
            # get name to print under examples
            print_name_str = ex_name.replace('_', ' ')
            print_name_list = print_name_str.split()
            print_name = []
            for word in print_name_list:
                if word in words_to_upper:
                    print_name.append(word.upper())
                elif word in words_to_skip:
                    print_name.append(word)
                elif word in words_to_map:
                    print_name.append(words_to_map[word])
                elif '(' in word:
                    print_name.append(word[0]+word[1].upper()+word[2:])
                else:
                    print_name.append(word.capitalize())
            # print_name = '_'.join(print_name)
            print_name = ' '.join(print_name)
            print(f'\t{counter+1}: {print_name}')
            # make path
            ex_path = os.path.join(each, ex_name, 'Input', 'SetupConfig.json')
            ex_path = ex_path.replace('\\','/') # replace backward slash with forward slash to work in unix
            # append to list
            ex_list.append({
                "name": print_name,
                "description": "Placeholder\n",
                "inputFile": ex_path
            })
            counter += 1
    out[infra_name] = {'Examples': ex_list}
    
# export
spath = os.path.join('examples','Examples.json')
with open(spath, 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=4)