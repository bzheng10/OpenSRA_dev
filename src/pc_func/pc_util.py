# functions to be used in PC

import numpy as np
from scipy.special import comb
from numba import njit, float64, int64
from numba_stats import truncnorm, norm, uniform
from src.nb_stats.stats_util import *


njit(
    # float64[:](float64[:],float64,float64,float64,float64),
    float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],int64),
    fastmath=True,
    cache=True,
    parallel=True
)
def res_to_samples(residuals, mean, sigma, low, high, dist_type):
    """
    convert lhs residuals to samples, for dist_type:
    1 = normal
    2 = lognormal
    3 = uniform
    mean, sigma, low, and high are arrays with length = dim1
    """
    # get dimensions
    dim1 = residuals.shape[0]
    dim2 = residuals.shape[1]
    samples = np.zeros((dim1,dim2))
    # if 'norm' in dist_type:
    if dist_type == 1 or dist_type == 2:
        # find where low == -np.inf or high == np.inf
        low_is_neg_inf = set(np.where(low==-np.inf)[0])
        high_is_inf = set(np.where(high==np.inf)[0])
        # union_cond = np.unique(np.hstack([low_is_neg_inf,high_is_inf]))
        union_cond = list(low_is_neg_inf.union(high_is_inf))
        ind_with_noninf_bounds = list(set(np.arange(dim1)).difference(set(union_cond)))
        # print(dim1)
        # print(len(union_cond))
        # print(len(ind_with_noninf_bounds))
        if len(ind_with_noninf_bounds) > 0:
            samples[ind_with_noninf_bounds,:] = truncnorm2_ppf_2d(
                p=norm2_cdf_2d(residuals[ind_with_noninf_bounds,:],0.0,1.0),
                xmin=low[ind_with_noninf_bounds].astype(float),
                xmax=high[ind_with_noninf_bounds].astype(float),
                loc=mean[ind_with_noninf_bounds].astype(float),
                scale=sigma[ind_with_noninf_bounds].astype(float)
            )
            # for ind in ind_with_noninf_bounds:
            #     samples[ind,:] = truncnorm.ppf(
            #         # q=norm.cdf(samples,0,1),
            #         p=norm.cdf(residuals[ind,:],0,1),
            #         xmin=low[ind],
            #         xmax=high[ind],
            #         loc=mean[ind],
            #         scale=sigma[ind]
            #     )
        # if neither low nor high are inf
        if len(union_cond) > 0:
            sigma_repeat = sigma[union_cond].repeat(dim2).reshape((-1, dim2))
            mean_repeat = mean[union_cond].repeat(dim2).reshape((-1, dim2))
            samples[union_cond,:] = residuals[union_cond,:] * sigma_repeat + mean_repeat
        # if neither low nor high are inf
        # else:
        #     mean_repeat = np.repeat(np.expand_dims(mean,axis=1), dim2, axis=1)
        #     sigma_repeat = np.repeat(np.expand_dims(sigma,axis=1), dim2, axis=1)
        #     samples = residuals*mean_repeat + mean_repeat
        # if dist_type == 'lognormal':
        if dist_type == 2:
            samples = np.exp(samples)
    # elif dist_type == 'uniform':
    elif dist_type == 3:
        # for ind in range(dim1):
            # samples[ind,:] = uniform.ppf(
            #     # q=norm.cdf(samples,0,1),
            #     p=norm.cdf(residuals[ind,:],0,1),
            #     a=low[ind],
            #     w=high[ind]-low[ind],
            # )
        samples = norm2_cdf_2d(residuals,0.0,1.0) * (high[ind]-low[ind]) + low[ind]
    return samples


# def res_to_samples(samples, mean, sigma, low, high, dist_type):
#     """
#     convert lhs residuals to samples, for dist_type:
#     1 = normal
#     2 = lognormal
#     3 = uniform
#     mean, sigma, low, and high are arrays with length = dim1
#     """
#     # get dimensions
#     dim1 = samples.shape[0]
#     dim2 = samples.shape[1]
#     # if 'norm' in dist_type:
#     if dist_type == 1 or dist_type == 2:
#         if low != -np.inf or high != np.inf:
#             # samples = truncnorm.ppf(
#             #     q=norm.cdf(samples),
#             #     a=(low-mean)/sigma,
#             #     b=(high-mean)/sigma,
#             #     loc=mean,
#             #     scale=sigma
#             # )
#             for i in range(dim1):
#             samples = truncnorm.ppf(
#                 # q=norm.cdf(samples,0,1),
#                 p=norm.cdf(samples,0,1),
#                 xmin=low,
#                 xmax=high,
#                 loc=mean,
#                 scale=sigma
#             )
#         else:
#             samples = samples*sigma + mean
#         # if dist_type == 'lognormal':
#         if dist_type == 2:
#             samples = np.exp(samples)
#     # elif dist_type == 'uniform':
#     elif dist_type == 3:
#         samples = uniform.ppf(
#             # q=norm.cdf(samples,0,1),
#             p=norm.cdf(samples,0,1),
#             a=low,
#             w=high-low,
#         )
#     return samples


# def make_domain_vector(left, right, num_pts, dist_type='lognormal'):
def make_domain_vector(left, right, num_pts, dist_type='lognormal'):
    """make logspace domain vector given left and right"""
    if dist_type == 'lognormal':
        return np.logspace(np.log10(left), np.log10(right), num_pts)
    elif dist_type == 'normal':
        return np.linspace(left, right, num_pts)
    
    
def make_normal_domain_vector(left, right, num_pts):
    """
    make linspace domain vector given left and right
    if lognormal, then provide log(left) and log(right)
    """
    return np.linspace(left, right, num_pts)


@njit(
    fastmath=True,
    cache=True,
)
def erf(x):
    """modified from https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf"""
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # Save the sign of x
    signs = np.sign(x)
    x = np.abs(x)
    # A & S 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
    return signs*y


def num_pc_terms(kl_dim, pc_order):
    """number of combinations"""
    return comb(kl_dim+pc_order, pc_order, exact=True)


# @njit(float64[:](float64[:],int64))
# @njit
def hermite_prob(x, i):
    """x = val, i = degree (up to 20); for deg = 0, poly=np.ones(x.shape)"""

    if i > 20:
        raise ValueError('Error in Hermite_Proba : degree prescribed is not available!')
    else:
        if i == 0:
            poly = np.ones(x.shape)

        elif i == 1:
            poly = x
            
        elif i == 2:
            poly = x**2 - 1
            
        elif i == 3:
            poly = x**3 - 3.*x
            
        elif i == 4:
            poly = x**4 - 6.*x**2 + 3
            
        elif i == 5:
            poly = x**5 - 10.*x**3 + 15.*x
            
        elif i == 6:
            poly = x**6 - 15.*x**4 + 45.*x**2 - 15
            
        elif i == 7:
            poly = x**7 - 21.*x**5 + 105.*x**3 - 105.*x
            
        elif i == 8:
            poly = x**8 - 28.*x**6 + 210.*x**4 - 420.*x**2 + 105
            
        elif i == 9:
            poly = x**9 - 36.*x**7 + 378.*x**5 - 1260.*x**3 + 945.*x
            
        elif i == 10:
            poly = x**10 - 45.*x**8 + 630.*x**6 - 3150.*x**4 + 4725.*x**2 - 945
            
        elif i == 11:
            poly = x**11 - 55.*x**9 + 990.*x**7 - 6930.*x**5 + 17325.*x**3 - 10395.*x
            
        elif i == 12:
            poly = x**12 - 66.*x**10 + 1485.*x**8 - 13860.*x**6 + 51975.*x**4 - 62370.*x**2 + 10395

        elif i == 13:
            poly = x**13 - 78.*x**11 + 2145.*x**9 - 25740.*x**7 + 135135.*x**5 - 270270.*x**3 + 135135.*x
            
        elif i == 14:
            poly = x**14 - 91.*x**12 + 3003.*x**10 - 45045.*x**8 + 315315.*x**6 - 945945.*x**4 + 945945.*x**2 - 135135
            
        elif i == 15:
            poly = x**15 - 105.*x**13 + 4095.*x**11 - 75075.*x**9 + 675675.*x**7 - \
                   2837835.*x**5 + 4729725.*x**3 - 2027025.*x
            
        elif i == 16:
            poly = x**16 - 120.*x**14 + 5460.*x**12 - 120120.*x**10 + 1351350.*x**8 - 7567560.*x**6 + 18918900.*x**4 - \
                   16216200.*x**2 + 2027025
            
        elif i == 17:
            poly = x**17 - 136.*x**15 + 7140.*x**13 - 185640.*x**11 + 2552550.*x**9 - 18378360.*x**7 + \
                   64324260.*x**5 - 91891800.*x**3 + 34459425.*x
               
        elif i == 18:
            poly = x**18 - 153.*x**16 + 9180.*x**14 - 278460.*x**12 + 4594590.*x**10 - \
                   41351310.*x**8 + 192972780.*x**6 - 413513100.*x**4 + 310134825.*x**2 - 34459425
               
        elif i == 19:
            poly = x**19 - 171.*x**17 + 11628.*x**15 - 406980.*x**13 + 7936110.*x**11 - 87297210.*x**9 + \
                   523783260.*x**7 - 1571349780.*x**5 + 1964187225.*x**3 - 654729075.*x
               
        elif i == 20:
            poly = x**20 - 190.*x**18 + 14535.*x**16 - 581400.*x**14 + 13226850.*x**12 - \
                   174594420.*x**10 + 1309458150.*x**8 - 5237832600.*x**6 + 9820936125.*x**4 - \
                   6547290750.*x**2 + 654729075

        return poly


def index_table_function(kl_dim, pc_order):
    """fugure implementation: simplify code with subfunctions to get index table and pc terms"""
    if pc_order > 10:
        raise ValueError('This pc_order is not available yet!!')
    
    else:
        ##########################################################################
        ######################## PC ORDER 0   ####################################
        ##########################################################################
        if pc_order == 0:
            table  =  np.zeros((1, kl_dim))
        
        ##########################################################################
        ######################## PC ORDER 1   ####################################
        ##########################################################################
        elif pc_order == 1:
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
                
            table = np.vstack([np.zeros((1,kl_dim)), index_table_1])
            
        ##########################################################################
        ######################## PC ORDER 2   ####################################
        ##########################################################################
        elif pc_order == 2:
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i]  =  index_table_1[count, i] + 1
                count += 1
                
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2])
            
        ##########################################################################
        ######################## PC ORDER 3   ####################################
        ##########################################################################    
        elif pc_order == 3:
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
            
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
            
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
            
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2, index_table_3])
        
        ##########################################################################
        ######################## PC ORDER 4   ####################################
        ##########################################################################
        elif pc_order == 4:
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            index_table_4[count, i] += 1
                            index_table_4[count, j] += 1
                            index_table_4[count, k] += 1
                            index_table_4[count, l] += 1
                            count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1

            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2, index_table_3, index_table_4])

        ##########################################################################
        ######################## PC ORDER 5   ####################################
        ##########################################################################
        elif pc_order == 5:
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
           
            table  =  np.vstack([
                np.zeros((1,kl_dim)), index_table_1, index_table_2, \
                index_table_3, index_table_4, index_table_5
            ])
        
        ##########################################################################
        ######################## PC ORDER 6   ####################################
        ##########################################################################
        elif pc_order == 6:
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
           
            table  =  np.vstack([
                np.zeros((1,kl_dim)), index_table_1, index_table_2, \
                index_table_3, index_table_4, index_table_5, index_table_6
            ])
           
        ##########################################################################
        ######################## PC ORDER 7   ####################################
        ##########################################################################
        elif pc_order == 7:
            num_pc_terms_7 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-6) - num_pc_terms(kl_dim,pc_order-7)
            index_table_7 = np.zeros([num_pc_terms_7, kl_dim])
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])

            #Order 7
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        index_table_7[count, i] += 1
                                        index_table_7[count, j] += 1
                                        index_table_7[count, k] += 1
                                        index_table_7[count, l] += 1
                                        index_table_7[count, m] += 1
                                        index_table_7[count, n] += 1
                                        index_table_7[count, o] += 1
                                        count += 1
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
            
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2,\
                index_table_3, index_table_4, index_table_5,\
                index_table_6, index_table_7])
           
        ##########################################################################
        ######################## PC ORDER 8   ####################################
        ##########################################################################
        elif pc_order == 8:
            num_pc_terms_8 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_7 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-6) - num_pc_terms(kl_dim,pc_order-7)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-7) - num_pc_terms(kl_dim,pc_order-8)
            index_table_8 = np.zeros([num_pc_terms_8, kl_dim])
            index_table_7 = np.zeros([num_pc_terms_7, kl_dim])
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            #Order 8
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            index_table_8[count, i] += 1
                                            index_table_8[count, j] += 1
                                            index_table_8[count, k] += 1
                                            index_table_8[count, l] += 1
                                            index_table_8[count, m] += 1
                                            index_table_8[count, n] += 1
                                            index_table_8[count, o] += 1
                                            index_table_8[count, p] += 1
                                            count += 1
            
            #Order 7
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        index_table_7[count, i] += 1
                                        index_table_7[count, j] += 1
                                        index_table_7[count, k] += 1
                                        index_table_7[count, l] += 1
                                        index_table_7[count, m] += 1
                                        index_table_7[count, n] += 1
                                        index_table_7[count, o] += 1
                                        count += 1
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
             
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2,\
                index_table_3, index_table_4, index_table_5,\
                index_table_6, index_table_7, index_table_8])
             
        ##########################################################################
        ######################## PC ORDER 9   ####################################
        ##########################################################################
        elif pc_order == 9:
            num_pc_terms_9 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_8 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_7 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-6) - num_pc_terms(kl_dim,pc_order-7)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-7) - num_pc_terms(kl_dim,pc_order-8)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-8) - num_pc_terms(kl_dim,pc_order-9)
            index_table_9 = np.zeros([num_pc_terms_9, kl_dim])
            index_table_8 = np.zeros([num_pc_terms_8, kl_dim])
            index_table_7 = np.zeros([num_pc_terms_7, kl_dim])
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            #Order 9
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            for q in range(p,kl_dim):
                                                index_table_9[count, i] += 1
                                                index_table_9[count, j] += 1
                                                index_table_9[count, k] += 1
                                                index_table_9[count, l] += 1
                                                index_table_9[count, m] += 1
                                                index_table_9[count, n] += 1
                                                index_table_9[count, o] += 1
                                                index_table_9[count, p] += 1
                                                index_table_9[count, q] += 1
                                                count += 1
            
            #Order 8
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            index_table_8[count, i] += 1
                                            index_table_8[count, j] += 1
                                            index_table_8[count, k] += 1
                                            index_table_8[count, l] += 1
                                            index_table_8[count, m] += 1
                                            index_table_8[count, n] += 1
                                            index_table_8[count, o] += 1
                                            index_table_8[count, p] += 1
                                            count += 1
            
            #Order 7
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        index_table_7[count, i] += 1
                                        index_table_7[count, j] += 1
                                        index_table_7[count, k] += 1
                                        index_table_7[count, l] += 1
                                        index_table_7[count, m] += 1
                                        index_table_7[count, n] += 1
                                        index_table_7[count, o] += 1
                                        count += 1
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
             
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2,\
                index_table_3, index_table_4, index_table_5,\
                index_table_6, index_table_7, index_table_8, \
                index_table_9])
             
        ##########################################################################
        ######################## PC ORDER 10  ####################################
        ##########################################################################
        elif pc_order == 10:
            num_pc_terms_10 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_9 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_8 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_7 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-6) - num_pc_terms(kl_dim,pc_order-7)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-7) - num_pc_terms(kl_dim,pc_order-8)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-8) - num_pc_terms(kl_dim,pc_order-9)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-9) - num_pc_terms(kl_dim,pc_order-10)
            index_table_10 = np.zeros([num_pc_terms_10, kl_dim])
            index_table_9 = np.zeros([num_pc_terms_9, kl_dim])
            index_table_8 = np.zeros([num_pc_terms_8, kl_dim])
            index_table_7 = np.zeros([num_pc_terms_7, kl_dim])
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            #Order 10
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            for q in range(p,kl_dim):
                                                for r in range(q,kl_dim):
                                                    index_table_10[count, i] += 1
                                                    index_table_10[count, j] += 1
                                                    index_table_10[count, k] += 1
                                                    index_table_10[count, l] += 1
                                                    index_table_10[count, m] += 1
                                                    index_table_10[count, n] += 1
                                                    index_table_10[count, o] += 1
                                                    index_table_10[count, p] += 1
                                                    index_table_10[count, q] += 1
                                                    index_table_10[count, r] += 1
                                                    count += 1
                                                    
            #Order 9
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            for q in range(p,kl_dim):
                                                index_table_9[count, i] += 1
                                                index_table_9[count, j] += 1
                                                index_table_9[count, k] += 1
                                                index_table_9[count, l] += 1
                                                index_table_9[count, m] += 1
                                                index_table_9[count, n] += 1
                                                index_table_9[count, o] += 1
                                                index_table_9[count, p] += 1
                                                index_table_9[count, q] += 1
                                                count += 1
            
            #Order 8
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            index_table_8[count, i] += 1
                                            index_table_8[count, j] += 1
                                            index_table_8[count, k] += 1
                                            index_table_8[count, l] += 1
                                            index_table_8[count, m] += 1
                                            index_table_8[count, n] += 1
                                            index_table_8[count, o] += 1
                                            index_table_8[count, p] += 1
                                            count += 1

            #Order 7
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        index_table_7[count, i] += 1
                                        index_table_7[count, j] += 1
                                        index_table_7[count, k] += 1
                                        index_table_7[count, l] += 1
                                        index_table_7[count, m] += 1
                                        index_table_7[count, n] += 1
                                        index_table_7[count, o] += 1
                                        count += 1
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
            
            table = np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2,\
                index_table_3, index_table_4, index_table_5,\
                index_table_6, index_table_7, index_table_8, \
                index_table_9, index_table_10])
        
        ##########################################################################
        ######################## END          ####################################
        ##########################################################################
        return table