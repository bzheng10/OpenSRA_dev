import numpy as np
from .pc_util import erf
from numba import njit, float64
from .pc_coeffs_first_int import pc_coeffs_first_int

@njit(
    float64[:,:,:](float64[:,:],float64[:],float64[:],float64[:],
                   float64[:],float64[:],float64[:],float64[:]),
    fastmath=True,
    cache=True
)
def pc_coeffs_cdf_single_int(v, muY, sigmaMuY, sigmaY, \
        amuZ, bmuZ, sigmaMuZ, sigmaZ): #
    """
    
    Notations (for simplicity):

    # Y = IM
    # Z = EDP
    # T = DM
    # V = DV

    ### Input types and dimensions:

    # v         row vector (n_v * 1), where n_v are the discretized values of the decision variable LN DV (x-axis) (in natural log units)

    # muY       column vector (1 * n_sites) with mean IM along sites
    # sigmaMuY  column vector (1 * n_sites) with epistemic uncertainty sigma IM along sites
    # sigmaY    column vector (1 * n_sites) with aleatory variability sigma IM along sites

    # amuZ      column vector (1 * n_sites) with slope of conditioning mean median model EDP|IM along sites
    # bmuZ      column vector (1 * n_sites) with intercept of conditioning mean median model EDP|IM along sites
    # sigmaMuZ  column vector (1 * n_sites) with epistemic uncertainty sigma EDP along sites
    # sigmaZ    column vector (1 * n_sites) with aleatory variability sigma EDP along sites

    # amuT      column vector (1 * n_sites) with slope of conditioning mean median model DM|EDP along sites
    # bmuT      column vector (1 * n_sites) with intercept of conditioning mean median model DM|EDP along sites
    # sigmaMuT  column vector (1 * n_sites) with epistemic uncertainty sigma DM along sites
    # sigmaT    column vector (1 * n_sites) with aleatory variability sigma DM along sites

    # amuV      column vector (1 * n_sites) with slope of conditioning mean median model DV|DM along sites
    # bmuV      column vector (1 * n_sites) with intercept of conditioning mean median model DV|DM along sites
    # sigmaMuV  column vector (1 * n_sites) with epistemic uncertainty sigma DV along sites
    # sigmaV    column vector (1 * n_sites) with aleatory variability sigma DV along sites

    ### Output types and dimensions

    # P_output = [P_1; ...;P_70] is a Matlab array containing 70 stacked arrays,
    # P_i, i = 1,...70
    #
    # Each array P_i is a (n_v * n_sites) array of type double, containing the 
    # Polynomial Chaos coefficients of the risk curve, such that P_i(j, k) 
    # represents the PC coefficient of DV value ln(v(j)) at site k.
    
    """
    
    # dimensions
    n_sites = len(muY)
    n_pts_v = len(v)
    
    ## PC Coefficients from First Integration!!
    P_output_1 = pc_coeffs_first_int(
        muY, sigmaMuY, sigmaY, \
        amuZ, bmuZ, sigmaMuZ, sigmaZ)
    
    # pull results
    p1c0 = np.zeros((1,n_sites))
    p2c0 = np.zeros((1,n_sites))
    p2c1 = np.zeros((1,n_sites))
    p3c0 = np.zeros((1,n_sites))
    p3c1 = np.zeros((1,n_sites))
    p4c0 = np.zeros((1,n_sites))
    p4c1 = np.zeros((1,n_sites))
    p4c2 = np.zeros((1,n_sites))
    p5c0 = np.zeros((1,n_sites))
    p5c1 = np.zeros((1,n_sites))
    p5c2 = np.zeros((1,n_sites))
    p6c0 = np.zeros((1,n_sites))
    p6c1 = np.zeros((1,n_sites))
    p6c2 = np.zeros((1,n_sites))
    p7c0 = np.zeros((1,n_sites))
    p7c1 = np.zeros((1,n_sites))
    p7c2 = np.zeros((1,n_sites))
    p7c3 = np.zeros((1,n_sites))
    p8c0 = np.zeros((1,n_sites))
    p8c1 = np.zeros((1,n_sites))
    p8c2 = np.zeros((1,n_sites))
    p8c3 = np.zeros((1,n_sites))
    p9c0 = np.zeros((1,n_sites))
    p9c1 = np.zeros((1,n_sites))
    p9c2 = np.zeros((1,n_sites))
    p9c3 = np.zeros((1,n_sites))
    p10c0 = np.zeros((1,n_sites))
    p10c1 = np.zeros((1,n_sites))
    p10c2 = np.zeros((1,n_sites))
    p10c3 = np.zeros((1,n_sites))
    p11c0 = np.zeros((1,n_sites))
    p11c1 = np.zeros((1,n_sites))
    p11c2 = np.zeros((1,n_sites))
    p11c3 = np.zeros((1,n_sites))
    p11c4 = np.zeros((1,n_sites))
    p12c0 = np.zeros((1,n_sites))
    p12c1 = np.zeros((1,n_sites))
    p12c2 = np.zeros((1,n_sites))
    p12c3 = np.zeros((1,n_sites))
    p12c4 = np.zeros((1,n_sites))
    p13c0 = np.zeros((1,n_sites))
    p13c1 = np.zeros((1,n_sites))
    p13c2 = np.zeros((1,n_sites))
    p13c3 = np.zeros((1,n_sites))
    p13c4 = np.zeros((1,n_sites))
    p14c0 = np.zeros((1,n_sites))
    p14c1 = np.zeros((1,n_sites))
    p14c2 = np.zeros((1,n_sites))
    p14c3 = np.zeros((1,n_sites))
    p14c4 = np.zeros((1,n_sites))
    p15c0 = np.zeros((1,n_sites))
    p15c1 = np.zeros((1,n_sites))
    p15c2 = np.zeros((1,n_sites))
    p15c3 = np.zeros((1,n_sites))
    p15c4 = np.zeros((1,n_sites))
    
    p1c0[0,:] = P_output_1[0, :]
    p2c0[0,:] = P_output_1[1, :]
    p2c1[0,:] = P_output_1[2, :]
    p3c0[0,:] = P_output_1[3, :]
    p3c1[0,:] = P_output_1[4, :]
    p4c0[0,:] = P_output_1[5, :]
    p4c1[0,:] = P_output_1[6, :]
    p4c2[0,:] = P_output_1[7, :]
    p5c0[0,:] = P_output_1[8, :]
    p5c1[0,:] = P_output_1[9, :]
    p5c2[0,:] = P_output_1[10, :]
    p6c0[0,:] = P_output_1[11, :]
    p6c1[0,:] = P_output_1[12, :]
    p6c2[0,:] = P_output_1[13, :]
    p7c0[0,:] = P_output_1[14, :]
    p7c1[0,:] = P_output_1[15, :]
    p7c2[0,:] = P_output_1[16, :]
    p7c3[0,:] = P_output_1[17, :]
    p8c0[0,:] = P_output_1[18, :]
    p8c1[0,:] = P_output_1[19, :]
    p8c2[0,:] = P_output_1[20, :]
    p8c3[0,:] = P_output_1[21, :]
    p9c0[0,:] = P_output_1[22, :]
    p9c1[0,:] = P_output_1[23, :]
    p9c2[0,:] = P_output_1[24, :]
    p9c3[0,:] = P_output_1[25, :]
    p10c0[0,:] = P_output_1[26, :]
    p10c1[0,:] = P_output_1[27, :]
    p10c2[0,:] = P_output_1[28, :]
    p10c3[0,:] = P_output_1[29, :]
    p11c0[0,:] = P_output_1[30, :]
    p11c1[0,:] = P_output_1[31, :]
    p11c2[0,:] = P_output_1[32, :]
    p11c3[0,:] = P_output_1[33, :]
    p11c4[0,:] = P_output_1[34, :]
    p12c0[0,:] = P_output_1[35, :]
    p12c1[0,:] = P_output_1[36, :]
    p12c2[0,:] = P_output_1[37, :]
    p12c3[0,:] = P_output_1[38, :]
    p12c4[0,:] = P_output_1[39, :]
    p13c0[0,:] = P_output_1[40, :]
    p13c1[0,:] = P_output_1[41, :]
    p13c2[0,:] = P_output_1[42, :]
    p13c3[0,:] = P_output_1[43, :]
    p13c4[0,:] = P_output_1[44, :]
    p14c0[0,:] = P_output_1[45, :]
    p14c1[0,:] = P_output_1[46, :]
    p14c2[0,:] = P_output_1[47, :]
    p14c3[0,:] = P_output_1[48, :]
    p14c4[0,:] = P_output_1[49, :]
    p15c0[0,:] = P_output_1[50, :]
    p15c1[0,:] = P_output_1[51, :]
    p15c2[0,:] = P_output_1[52, :]
    p15c3[0,:] = P_output_1[53, :]
    p15c4[0,:] = P_output_1[54, :]
    
    # precalculate
    a = (-1/2)*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)**(-1)
    b = (bmuZ+amuZ*muY)*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+ \
        sigmaZ**2)**(-1)

    # resize
    a = np.expand_dims(a,axis=0)
    b = np.expand_dims(b,axis=0)
    # v = np.expand_dims(v,axis=1)
    
    # more calculations with broadcasting, after resizing
    expCst = np.exp(1.0)**((1/4)*a**(-1)*(b+2*a*v)**2)
    ErfCst = erf((b+2*a*v)/(2 *(-a)**(1/2)))
    
    # get Polynomial Coefficients;
    P_1 = (-1/2)*((-1)*a)**(-1/2)*((-1)+ErfCst)*p1c0*np.pi**(1/2)
    P_2 = (1/4)*a**(-1)*(2*expCst*p2c1+((-1)*a)**(-3/2)*a*((-1)+ErfCst)*( \
        2*a*p2c0+(-1)*b*p2c1)*np.pi**(1/2))
    P_3 = (1/4)*a**(-1)*(2*expCst*p3c1+((-1)*a)**(-3/2)*a*((-1)+ErfCst)*( \
        2*a*p3c0+(-1)*b*p3c1)*np.pi**(1/2))
    P_4 = (1/8)*((-1)*a)**(-5/2)*(b*p4c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
        *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p4c0*np.pi**( \
        1/2)+2*a*(((-1)+ErfCst)*(b*p4c1+p4c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p4c1+p4c2*v)))
    P_5 = (1/8)*((-1)*a)**(-5/2)*(b*p5c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
        *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p5c0*np.pi**( \
        1/2)+2*a*(((-1)+ErfCst)*(b*p5c1+p5c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p5c1+p5c2*v)))
    P_6 = (1/8)*((-1)*a)**(-5/2)*(b*p6c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
        *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p6c0*np.pi**( \
        1/2)+2*a*(((-1)+ErfCst)*(b*p6c1+p6c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p6c1+p6c2*v)))
    P_7 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p7c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p7c0*np.pi**(1/2)+ \
        (-2)*a*(4*((-1)*a)**(1/2)*expCst*p7c3+b**2*((-1)+ErfCst)*p7c2* \
        np.pi**(1/2)+3*b*((-1)+ErfCst)*p7c3*np.pi**(1/2)+2*((-1)*a)**(1/2)*b* \
        expCst*(p7c2+p7c3*v))+4*a**2*(((-1)+ErfCst)*(b*p7c1+p7c2)*np.pi**( \
        1/2)+2*((-1)*a)**(1/2)*expCst*(p7c1+v*(p7c2+p7c3*v))))
    P_8 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p8c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p8c0*np.pi**(1/2)+ \
        (-2)*a*(4*((-1)*a)**(1/2)*expCst*p8c3+b**2*((-1)+ErfCst)*p8c2* \
        np.pi**(1/2)+3*b*((-1)+ErfCst)*p8c3*np.pi**(1/2)+2*((-1)*a)**(1/2)*b* \
        expCst*(p8c2+p8c3*v))+4*a**2*(((-1)+ErfCst)*(b*p8c1+p8c2)*np.pi**( \
        1/2)+2*((-1)*a)**(1/2)*expCst*(p8c1+v*(p8c2+p8c3*v))))
    P_9 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p9c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p9c0*np.pi**(1/2)+ \
        (-2)*a*(4*((-1)*a)**(1/2)*expCst*p9c3+b**2*((-1)+ErfCst)*p9c2* \
        np.pi**(1/2)+3*b*((-1)+ErfCst)*p9c3*np.pi**(1/2)+2*((-1)*a)**(1/2)*b* \
        expCst*(p9c2+p9c3*v))+4*a**2*(((-1)+ErfCst)*(b*p9c1+p9c2)*np.pi**( \
        1/2)+2*((-1)*a)**(1/2)*expCst*(p9c1+v*(p9c2+p9c3*v))))
    P_10 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p10c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p10c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p10c3+b**2*((-1)+ErfCst)* \
        p10c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p10c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p10c2+p10c3*v))+4*a**2*(((-1)+ErfCst)*(b*p10c1+ \
        p10c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p10c1+v*(p10c2+p10c3* \
        v))))
    P_11 = (1/32)*((-1)*a)**(-9/2)*(b**3*p11c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p11c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p11c4+b**2*(( \
        -1)+ErfCst)*p11c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p11c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p11c3+p11c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p11c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p11c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p11c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p11c3+3* \
        p11c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p11c2+v*(p11c3+p11c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p11c1+p11c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p11c1+v*(p11c2+v*(p11c3+p11c4*v)))))
    P_12 = (1/32)*((-1)*a)**(-9/2)*(b**3*p12c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p12c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p12c4+b**2*(( \
        -1)+ErfCst)*p12c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p12c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p12c3+p12c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p12c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p12c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p12c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p12c3+3* \
        p12c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p12c2+v*(p12c3+p12c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p12c1+p12c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p12c1+v*(p12c2+v*(p12c3+p12c4*v)))))
    P_13 = (1/32)*((-1)*a)**(-9/2)*(b**3*p13c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p13c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p13c4+b**2*(( \
        -1)+ErfCst)*p13c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p13c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p13c3+p13c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p13c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p13c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p13c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p13c3+3* \
        p13c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p13c2+v*(p13c3+p13c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p13c1+p13c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p13c1+v*(p13c2+v*(p13c3+p13c4*v)))))
    P_14 = (1/32)*((-1)*a)**(-9/2)*(b**3*p14c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p14c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p14c4+b**2*(( \
        -1)+ErfCst)*p14c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p14c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p14c3+p14c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p14c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p14c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p14c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p14c3+3* \
        p14c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p14c2+v*(p14c3+p14c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p14c1+p14c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p14c1+v*(p14c2+v*(p14c3+p14c4*v)))))
    P_15 = (1/32)*((-1)*a)**(-9/2)*(b**3*p15c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p15c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p15c4+b**2*(( \
        -1)+ErfCst)*p15c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p15c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p15c3+p15c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p15c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p15c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p15c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p15c3+3* \
        p15c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p15c2+v*(p15c3+p15c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p15c1+p15c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p15c1+v*(p15c2+v*(p15c3+p15c4*v)))))
    
    # store to output
    # P_output = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13,  P_14, P_15))
    P_output = np.zeros((n_pts_v, n_sites, 15))
    P_output[:,:,0] = P_1
    P_output[:,:,1] = P_2
    P_output[:,:,2] = P_3
    P_output[:,:,3] = P_4
    P_output[:,:,4] = P_5
    P_output[:,:,5] = P_6
    P_output[:,:,6] = P_7
    P_output[:,:,7] = P_8
    P_output[:,:,8] = P_9
    P_output[:,:,9] = P_10
    P_output[:,:,10] = P_11
    P_output[:,:,11] = P_12
    P_output[:,:,12] = P_13
    P_output[:,:,13] = P_14
    P_output[:,:,14] = P_15
    
    # return
    return P_output