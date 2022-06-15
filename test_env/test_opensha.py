# import os, logging, json, sys, importlib, time, pathlib, queue
import time
# import numpy as np
# import pandas as pd

from opensha_thread import OpenSHAWrapper

def main():
    erf_name = 'Mean UCERF3 FM3.1'
    imr_name = 'Abrahamson, Silva & Kamai (2014)'
    num_threads = 4
    # opensha = None
    opensha = OpenSHAWrapper(erf_name, imr_name, num_threads)


    n_sources = 100
    # src_list = np.random.randint(int(opensha.erfs[0].getNumSources()),size=n_sources)
    # rup_list = np.zeros(src_list.shape,dtype=int)
    opensha.get_random_source(n_sources)


    filepath = r'C:\Users\barry\OneDrive\Desktop\temp\data.csv'
    opensha.get_sites(filepath)
    # data = pd.read_csv(r'C:\Users\barry\OneDrive\Desktop\temp\data.csv')
    # n_sites = data.shape[0]


    ##Site data
    ## sites = ArrayList()
    #site_params = {
    #    'Vs30': {
    #        'Name': 'Vs30',
    #        'Index': [0,1,2,3]
    #    },
    #    'Z1p0': {
    #        'Name': 'Depth 1.0 km/sec',
    #        'Index': [5,7,9,11,13,15]
    #    },
    #    'Z2p5': {
    #        'Name': 'Depth 2.5 km/sec',
    #        'Index': [4,6,8,10,12,14]
    #    }
    #}


        
    ## make OpenSHA sites
    #sites_for_opensha = []
    ## loop through each location
    #for i in range(n_sites):
    #    # convert to OpenSHA site object
    #    site = Site(Location(data['Latitude'][i],data['Longitude'][i]))
    #    # add site parameters to site
    #    for param in site_params:
    #        newParam = Parameter.clone(opensha.imrs[0].getSiteParams().getParameter(site_params[param]['Name']))
    #        try:
    #            newParam.setValue(Double(data[param][i]))
    #            if param == 'Vs30':
    #                newParamType = Parameter.clone(opensha.imrs[0].getSiteParams().getParameter('Vs30 Type'))
    #                newParamType.setValue(data['Vs30 Type'][i])
    #                site.addParameter(newParamType)
    #        except jpype.JException as exception: # if value is outside the allowed range
    #            newParam.setValue(newParam.getMin()) # values are invalid, set to minimum allowed value
    #        site.addParameter(newParam)
    #    sites_for_opensha.append(site)
        
        
    time_start = time.time()
    opensha.get_im_with_threads()
    print(time.time()-time_start)


if __name__ == '__main__':
    main()