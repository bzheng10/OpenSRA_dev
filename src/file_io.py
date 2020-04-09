#####################################################################################################################
##### Functions for input/output
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
##### Read EQHazard outputs
#####################################################################################################################
def read_json_EQHazard(path, verbose=False):
	
	## load json file
    with open(EQHaz_file, 'r') as f:
        jd = json.load(f)

    ## set up empty labels for IM dictionary
    im_data = dict([(label, dict()) for label in [
        'site_data', 'im_data', 'im_type', 'site_loc', 'eq_rup'
    ]])

    ## partition json content into a few internal variables for convenience
    eq_info = jd.get('EqRupture',None)
    gm_input = jd.get('GroundMotions',None)

	## items to get from EQHazard output
    im_list = ['SA','PGA','PGV']
    unit_list = ['g','g','cm_s']
    flag_log_list = ['ln','ln','ln']
    qnt_type_list = ['acc','acc','vel']
    
	## initialize empty arrays
    site_lat = []
    site_long = []
    data_type = []
    data_val = []
    data_source = []
    im_qnt_type = []
    im_unit = []
    im_flag_log = []

	## loop to store site and metadata information
    for i in range(len(gm_input)):
	
		## store longitude and latitude
        site_loc = gm_input[i].get('Location',None)
        if site_loc is not None:
            site_lat.append(site_loc.get('Latitude', None))
            site_long.append(site_loc.get('Longitude', None))
        else:
            site_lat.append(None)
            site_long.append(None)

		## store data type and source
        site_data = gm_input[i].get('SiteData',None)[0]
        if site_data is not None:
            data_type.append(site_data.get('Type', None))
            data_val.append(site_data.get('Value', None))
            data_source.append(site_data.get('Source', None))
        else:
            data_type.append(None)
            data_val.append(None)
            data_source.append(None)
    
	## store IM measures
    for im in im_list:
	
		## initialize empty arrays
        im_qnt = []
        im_mean = []
        im_sig_tot = []
        im_sig_intra = []
        im_sig_inter = []
        im_label = []
		
		## loop through all sites
        for i in range(len(gm_input)):
		
			## try to find target items specified in "im_list" and store them
            try:
                key_val = next(key_val for _, key_val in enumerate(gm_input[i].keys()) if im in key_val)
                im_qnt = gm_input[i][key_val]
                im_label = key_val
            except:
                im_qnt = None
                im_label = None

			## if target IM measure is available, further partition its content into mean and sigma
            if im_qnt is not None:
                im_mean.append(im_qnt.get('Mean', None))
                im_sig_tot.append(im_qnt.get('TotalStdDev', None))
                im_sig_intra.append(im_qnt.get('IntraEvStdDev', None))
                im_sig_inter.append(im_qnt.get('InterEvStdDev', None))
            else:
                im_mean.append(None)
                im_sig_tot.append(None)
                im_sig_intra.append(None)
                im_sig_inter.append(None)

		## if specral acceleration is specified, also store period
        if 'SA' in im and im_mean is not None:
            im_period = jd.get('Periods',None)
            im_data['im_data'].update({'period': im_period,
                                       im.lower()+'_mean': im_mean,
                                       im.lower()+'_sig_total': im_sig_tot,
                                       im.lower()+'_sig_intra': im_sig_intra,
                                       im.lower()+'_sig_inter': im_sig_inter})
        else:
            im_data['im_data'].update({im.lower()+'_mean': im_mean,
                                       im.lower()+'_sig_total': im_sig_tot,
                                       im.lower()+'_sig_intra': im_sig_intra,
                                       im.lower()+'_sig_inter': im_sig_inter})

		## see if other information are provided (e.g., units, if values are in log)
        try:
            next(j for i, j in enumerate(gm_input[0].keys()) if im in j)
            im_qnt_type.append(qnt_type_list[i])
            im_unit.append(unit_list[i])
            im_flag_log.append(flag_log_list[i])
        except:
            pass

	## update the IM data dictionary
    im_data['im_type'].update({'type': im_qnt_type,
                               'unit': im_unit,
                               'log_flag': im_flag_log})

    im_data['eq_rup'].update(eq_info)

    im_data['site_loc'].update({'latitude': site_lat,
                                'longitude': site_long})

    im_data['site_data'].update({'vs30': data_val,
                                 'source': data_source})

	##
    return im_data


#####################################################################################################################
##### Reading other json files that are manually created
#####################################################################################################################
def read_json_other(path, var, verbose=False):
	##
    with open(path, 'r') as f:
        jd = json.load(f)

	##
    site = jd.get('Site',None)
    site_list = site.get('SiteList',None)
	
	##
    data2get = []
	
	##
    for i in range(len(site_list)):
        if site_list[i][var] is not None:
            data2get.append(site_list[i][var])
        else:
            data2get.append(np.nan)
	
	##
    return data2get