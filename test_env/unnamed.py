#
import os, sys, time
import numpy as np
import importlib

#
sys.path.append(os.getcwd())

#
from src.base_class import BaseClass
# from src.im import spatial_corr, cross_corr
# from src.im.spatial_corr import jb09
# from src.im.cross_corr import bj08
from src.dm import *
from src.dv import *

kwargs = {}

n_site = 1
n_sample = 10
kwargs['n_site'] = n_site
kwargs['n_sample'] = n_sample
np.random.seed(1)

dm_type = 'pipe_strain'
method_name = 'BainBray2022'

# pull inputs locally
kwargs['pgd'] = np.random.randint(13, size=n_sample)*0.25 + 0.25
kwargs['d'] = np.random.randint(13, size=n_sample)*200 + 200
kwargs['l_def'] = np.random.randint(13, size=n_sample)*25 + 25
kwargs['t'] = np.random.randint(5, size=n_sample)*2 + 4
kwargs['sigma_y'] = np.ones(n_sample) * 250000
kwargs['soil_type'] = np.random.choice(['sand', 'clay'], size=n_sample)
kwargs['gamma_t'] = np.random.randint(5, size=n_sample)*100 + 150
kwargs['h_cover'] = np.ones(n_sample) * 2.4
kwargs['phi'] = np.ones(n_sample) * 35
kwargs['delta'] = np.ones(n_sample) * 0.8
kwargs['k0'] = np.ones(n_sample) * 1
kwargs['alpha'] = np.random.randint(6, size=n_sample)*0.1 + 0.5
kwargs['s_u'] = np.random.randint(12, size=n_sample)*12 + 12

time_start = time.time()
module = importlib.import_module(''.join(['src.dm.',dm_type]))
# class_ = getattr(spatial_corr, method_name)
class_ = getattr(module, method_name)
instance_ = class_()
dm_out = instance_(kwargs)
print (time.time()-time_start)


# DM-DV

dv_type = 'buckling'
method_name = 'BainBray2022'

kwargs['pressure_op'] = np.ones(n_sample) * 3200
# kwargs['eps_p'] = dm_out['eps_p']
# kwargs['eps_p'] = 10**(np.random.rand(n_sample)*(-2 - -4) + -4)*100
kwargs['eps_p'] = 10**(np.arange(-4,-2,0.2))*100
time_start = time.time()
module = importlib.import_module(''.join(['src.dv.',dv_type]))
# class_ = getattr(spatial_corr, method_name)
class_ = getattr(module, method_name)
instance_ = class_()
dv_out = instance_(kwargs)
print (time.time()-time_start)





# kwargs['T'] = 2
# # kwargs['d'] = np.array([2.3, 100, 0.4])
# kwargs['d'] = np.random.randint(0,100,n_site)
# kwargs['T1'] = 10*np.random.rand(n_site)
# kwargs['T2'] = 10*np.random.rand(n_site)
# kwargs['geo_cond'] = 1


# for i in range(1):
#     output = instance_(kwargs)
#     if i == 0:
#         print (time.time()-time_start)
#         time_start = time.time()
# print (time.time()-time_start)

# time_start = time.time()
# # for i in range(10000):
#     # output_v2 = jb09(kwargs['d'],kwargs['T'],kwargs['geo_cond'])
# output_v2 = [bj08(kwargs['T1'][i],kwargs['T2'][i]) for i in range(n_sites)]
#     # if i == 0:
#     #     print (time.time()-time_start)
#     #     time_start = time.time()
# print (time.time()-time_start)


# time_start = time.time()
# output_v2 = JayaramBaker2009_v2(kwargs['d'],kwargs['T'],kwargs['geo_cond'])
# print (time.time()-time_start)

sys.exit()

# method_name = 'JayaramBaker2009'
# method_name = 'BakerJayaram2008'

# n_sites = 10000

# kwargs = {}
# kwargs['T'] = 2
# # kwargs['d'] = np.array([2.3, 100, 0.4])
# kwargs['d'] = np.random.randint(0,100,n_sites)
# kwargs['T1'] = 10*np.random.rand(n_sites)
# kwargs['T2'] = 10*np.random.rand(n_sites)
# kwargs['geo_cond'] = 1


# time_start = time.time()
# # class_ = getattr(spatial_corr, method_name)
# class_ = getattr(cross_corr, method_name)
# instance_ = class_()
# for i in range(1):
#     output = instance_(kwargs)
#     if i == 0:
#         print (time.time()-time_start)
#         time_start = time.time()
# print (time.time()-time_start)

# time_start = time.time()
# # for i in range(10000):
#     # output_v2 = jb09(kwargs['d'],kwargs['T'],kwargs['geo_cond'])
# output_v2 = [bj08(kwargs['T1'][i],kwargs['T2'][i]) for i in range(n_sites)]
#     # if i == 0:
#     #     print (time.time()-time_start)
#     #     time_start = time.time()
# print (time.time()-time_start)


# # time_start = time.time()
# # output_v2 = JayaramBaker2009_v2(kwargs['d'],kwargs['T'],kwargs['geo_cond'])
# # print (time.time()-time_start)