# Python packages
import os, json, sys


# Instructions:
# use function add_string to add text
# function automatically appends newline character
# for new lines, use empty string ''


# main
def main():
    # flag for Java text
    flag_includeJava = True

    # setup
    cwd = os.getcwd()
    # get list of devs from devs.json
    with open(os.path.join(cwd,'readme_src','ReadmeResource.json'),'r') as f:
        readme_resource = json.load(f)
    print(readme_resource.keys())
    devs = readme_resource['Developers'] # developers
    depends = readme_resource['PythonModules'] # dependencies
    ackns = readme_resource['Acknowledgements'] # acknowledgements
    contacts = readme_resource['Contacts'] # contacts
    py_ver = readme_resource['Python']['Version'] # Python version
    # Current Python version:
    # py_ver = sys.version
    # py_ver = py_ver[:py_ver.find(' | ')]
    
    # init string for storing text
    text = []

    # 0: for builds
    text = add_string(text, '**Builds Status**')
    text = add_string(text, '')
    text = add_string(text, '| **Windows** | **Mac** |')
    text = add_string(text, '|---|---|')
    text = add_string(text, '[![Build Status]()]()|[![Build Status]()]()')
    text = add_string(text, '')

    # 1: OpenSRA
    text = add_string(text, '# OpenSRA')
    text = add_string(text,
        'This repository contains the source code to OpenSRA_backend, ' +\
        'developed by [Slate Geotechnical Consultants](http://slategeotech.com/), with ' +\
        'assistance from [NHERI SimCenter](https://simcenter.designsafe-ci.org/) and ' +\
        '[UC Berkeley](https://ce.berkeley.edu/).'
    )
    text = add_string(text, '')

    # 1-1: Developers
    text = add_string(text, '## Developers')
    for dev in devs:
        info = devs[dev]
        string = f"{dev}, {info['Title']}, @ {info['Affiliation']}: {info['Profile']}"
        text = add_string(text, string)
        text = add_string(text, '')
        
    # 1-2: Dependencies
    text = add_string(text, '## Dependencies')
    text = add_string(text, '')
    
    # 1-2-1: Python
    text = add_string(text, '### Python')
    text = add_string(text, f'OpenSRA has been tested on **Python version {py_ver}**')
    text = add_string(text, '')
    text = add_string(text, 
        'The Python modules required for the current version of OpenSRA are listed ' +\
        'below, along with the versions used for testing. Modules were installed ' +\
        'via the "conda-forge" channel. If you are experiencing difficulty installing ' +\
        'the modules, consider working in a clean environment.'
    )
    text = add_string(text, '')
    for module in depends:
        info = depends[module]
        string = f"[{module} ({info['Version']})]({info['Reference']})"
        if info['Note'] is not None:
            string = string + f" - {info['Note']}"
        text = add_string(text, string)
        text = add_string(text, '')

    # 1-2-2: Java
    if flag_includeJava:
        # get list of devs from devs.json
        with open(os.path.join(cwd,'readme_src','JavaText.txt'),'r') as f:
            for line in f:
                text = add_string(text, line.rstrip())
        text = add_string(text, '')
        
    # 1-3: User's Guide
    text = add_string(text, "## User's Guide")
    text = add_string(text, 
        "To run OpenSRA in the command prompt, nagivate to the root " +\
        "folder of OpenSRA and run the command:"
    )
    text = add_string(text, "```")
    text = add_string(text, "python OpenSRA.py -i PATH_TO_INPUT_FOLDER")
    text = add_string(text, "```")
    text = add_string(text, '')
    text = add_string(text, "To clean results from the previous run, run the command:")
    text = add_string(text, "```")
    text = add_string(text, "python OpenSRA.py -i PATH_TO_INPUT_FOLDER -c yes")
    text = add_string(text, "```")
    text = add_string(text, '')
    
    # 1-4: Developer's Guide
    text = add_string(text, "## Developer's Guide")
    text = add_string(text, "Under development")
    text = add_string(text, '')
    
    # 1-5: Acknowledgements
    text = add_string(text, "## Acknowledgements")
    text = add_string(text, 
        "The OpenSRA development team would like to acknowledge [Dr. Wael Elhaddad " +\
        "and Dr. Kuanshi Zhong @ NHERI SimCenter]" +\
        "(https://simcenter.designsafe-ci.org/about/people/) for providing " +\
        "developmental support on the OpenSHA interface, and [Dr. Simon Kwong " +\
        "@ USGS](https://www.usgs.gov/staff-profiles/neal-simon-kwong) for providing " +\
        "technical feedback on seismic and performance-based hazard analysis."
    )
    text = add_string(text, '')

    # 1-5: Acknowledgements
    text = add_string(text, "## License")
    text = add_string(text, "Please check the license file in the root folder.")
    text = add_string(text, '')
    
    # export to README.md
    save_loc = os.path.join(os.getcwd(),'README.md')
    print(f"\n - Saving readme file to: {save_loc}")
    with open(save_loc,'w') as f:
        for line in text:
            f.write(line)

    # generate requirements.txt file
    reqs = [module+'=='+depends[module]['Version']+'\n' for module in depends]
    print(f"\n - Saving requirement file to: {os.path.join(cwd,'requirements.txt')}")
    with open(os.path.join(cwd,'requirements.txt'),'w') as f:
        for line in reqs:
            f.write(line)
            

# add string
def add_string(text, string):
    text.append(string+'\n')
    return text
    
    
# init
if __name__ == '__main__':
    main()