import numpy as np
from scipy.special import erf
from numba import vectorize


@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
def PC_Coeffs_Simpler_CDF_Single_Integral_array_function(v, muY, sigmaMuY, sigmaY, \
        amuZ,, bmuZ,, sigmaMuZ,, sigmaZ), #
    ## PC Coefficients from First Integration!!;
    P_output_1  =  PC_Coeffs_Simpler_First_Integration_function(muY, sigmaMuY, \
        sigmaY,, amuZ,, bmuZ,, sigmaMuZ,, sigmaZ)
    p1c0 = P_output_1[0, :]
    p2c0 = P_output_1[1, :]
    p2c1 = P_output_1[2, :]
    p3c0 = P_output_1[3, :]
    p3c1 = P_output_1[4, :]
    p4c0 = P_output_1[5, :]
    p4c1 = P_output_1[6, :]
    p4c2 = P_output_1[7, :]
    p5c0 = P_output_1[8, :]
    p5c1 = P_output_1[9, :]
    p5c2 = P_output_1[10, :]
    p6c0 = P_output_1[11, :]
    p6c1 = P_output_1[12, :]
    p6c2 = P_output_1[13, :]
    p7c0 = P_output_1[14, :]
    p7c1 = P_output_1[15, :]
    p7c2 = P_output_1[16, :]
    p7c3 = P_output_1[17, :]
    p8c0 = P_output_1[18, :]
    p8c1 = P_output_1[19, :]
    p8c2 = P_output_1[20, :]
    p8c3 = P_output_1[21, :]
    p9c0 = P_output_1[22, :]
    p9c1 = P_output_1[23, :]
    p9c2 = P_output_1[24, :]
    p9c3 = P_output_1[25, :]
    p10c0 = P_output_1[26, :]
    p10c1 = P_output_1[27, :]
    p10c2 = P_output_1[28, :]
    p10c3 = P_output_1[29, :]
    p11c0 = P_output_1[30, :]
    p11c1 = P_output_1[31, :]
    p11c2 = P_output_1[32, :]
    p11c3 = P_output_1[33, :]
    p11c4 = P_output_1[34, :]
    p12c0 = P_output_1[35, :]
    p12c1 = P_output_1[36, :]
    p12c2 = P_output_1[37, :]
    p12c3 = P_output_1[38, :]
    p12c4 = P_output_1[39, :]
    p13c0 = P_output_1[40, :]
    p13c1 = P_output_1[41, :]
    p13c2 = P_output_1[42, :]
    p13c3 = P_output_1[43, :]
    p13c4 = P_output_1[44, :]
    p14c0 = P_output_1[45, :]
    p14c1 = P_output_1[46, :]
    p14c2 = P_output_1[47, :]
    p14c3 = P_output_1[48, :]
    p14c4 = P_output_1[49, :]
    p15c0 = P_output_1[50, :]
    p15c1 = P_output_1[51, :]
    p15c2 = P_output_1[52, :]
    p15c3 = P_output_1[53, :]
    p15c4 = P_output_1[54, :]
    v  =  v.T
    a = (-1/2)*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)**(-1) \
        
    b = (bmuZ+amuZ*muY)*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+ \
        sigmaZ**2)**(-1)
    
    
    
    n_sites  =  muY.shape[1]
    
    n_pts_v  =  len(v)
    
    
    
    P_1  =  np.zeros((n_pts_v, n_sites))
    P_2  =  np.zeros((n_pts_v, n_sites))
    P_3  =  np.zeros((n_pts_v, n_sites))
    P_4  =  np.zeros((n_pts_v, n_sites))
    P_5  =  np.zeros((n_pts_v, n_sites))
    P_6  =  np.zeros((n_pts_v, n_sites))
    P_7  =  np.zeros((n_pts_v, n_sites))
    P_8  =  np.zeros((n_pts_v, n_sites))
    P_9  =  np.zeros((n_pts_v, n_sites))
    P_10  =  np.zeros((n_pts_v, n_sites))
    P_11  =  np.zeros((n_pts_v, n_sites))
    P_12  =  np.zeros((n_pts_v, n_sites))
    P_13  =  np.zeros((n_pts_v, n_sites))
    P_14  =  np.zeros((n_pts_v, n_sites))
    P_15  =  np.zeros((n_pts_v, n_sites))
    
    #;
    ## Polynomial Coefficients;
    #;
    
    for i in range(n_pts_v):
        
        
        ErfCst = erf((b+2*a*v[i])/(2 *(-a)**(1/2)))
        expCst = np.exp(1)**((1/4)*a**(-1)*(b+2*a*v[i])**2)
        
        P_1[i, :] = (-1/2)*((-1)*a)**(-1/2)*((-1)+ErfCst)*p1c0*np.pi**(1/2)
        P_2[i, :] = (1/4)*a**(-1)*(2*expCst*p2c1+((-1)*a)**(-3/2)*a*((-1)+ErfCst)*( \
            2*a*p2c0+(-1)*b*p2c1)*np.pi**(1/2))
        P_3[i, :] = (1/4)*a**(-1)*(2*expCst*p3c1+((-1)*a)**(-3/2)*a*((-1)+ErfCst)*( \
            2*a*p3c0+(-1)*b*p3c1)*np.pi**(1/2))
        P_4[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p4c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
            *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p4c0*np.pi**( \
            1/2)+2*a*(((-1)+ErfCst)*(b*p4c1+p4c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p4c1+p4c2*v[i])))
        P_5[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p5c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
            *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p5c0*np.pi**( \
            1/2)+2*a*(((-1)+ErfCst)*(b*p5c1+p5c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p5c1+p5c2*v[i])))
        P_6[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p6c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
            *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p6c0*np.pi**( \
            1/2)+2*a*(((-1)+ErfCst)*(b*p6c1+p6c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p6c1+p6c2*v[i])))
        P_7[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p7c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p7c0*np.pi**(1/2)+ \
            (-2)*a*(4*((-1)*a)**(1/2)*expCst*p7c3+b**2*((-1)+ErfCst)*p7c2* \
            np.pi**(1/2)+3*b*((-1)+ErfCst)*p7c3*np.pi**(1/2)+2*((-1)*a)**(1/2)*b* \
            expCst*(p7c2+p7c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p7c1+p7c2)*np.pi**( \
            1/2)+2*((-1)*a)**(1/2)*expCst*(p7c1+v[i]*(p7c2+p7c3*v[i]))))
        P_8[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p8c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p8c0*np.pi**(1/2)+ \
            (-2)*a*(4*((-1)*a)**(1/2)*expCst*p8c3+b**2*((-1)+ErfCst)*p8c2* \
            np.pi**(1/2)+3*b*((-1)+ErfCst)*p8c3*np.pi**(1/2)+2*((-1)*a)**(1/2)*b* \
            expCst*(p8c2+p8c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p8c1+p8c2)*np.pi**( \
            1/2)+2*((-1)*a)**(1/2)*expCst*(p8c1+v[i]*(p8c2+p8c3*v[i]))))
        P_9[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p9c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p9c0*np.pi**(1/2)+ \
            (-2)*a*(4*((-1)*a)**(1/2)*expCst*p9c3+b**2*((-1)+ErfCst)*p9c2* \
            np.pi**(1/2)+3*b*((-1)+ErfCst)*p9c3*np.pi**(1/2)+2*((-1)*a)**(1/2)*b* \
            expCst*(p9c2+p9c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p9c1+p9c2)*np.pi**( \
            1/2)+2*((-1)*a)**(1/2)*expCst*(p9c1+v[i]*(p9c2+p9c3*v[i]))))
        P_10[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p10c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p10c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p10c3+b**2*((-1)+ErfCst)* \
            p10c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p10c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p10c2+p10c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p10c1+ \
            p10c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p10c1+v[i]*(p10c2+p10c3* \
            v[i]))))
        P_11[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p11c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p11c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p11c4+b**2*(( \
            -1)+ErfCst)*p11c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p11c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p11c3+p11c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p11c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p11c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p11c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p11c3+3* \
            p11c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p11c2+v[i]*(p11c3+p11c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p11c1+p11c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p11c1+v[i]*(p11c2+v[i]*(p11c3+p11c4*v[i])))))
        P_12[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p12c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p12c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p12c4+b**2*(( \
            -1)+ErfCst)*p12c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p12c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p12c3+p12c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p12c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p12c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p12c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p12c3+3* \
            p12c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p12c2+v[i]*(p12c3+p12c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p12c1+p12c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p12c1+v[i]*(p12c2+v[i]*(p12c3+p12c4*v[i])))))
        P_13[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p13c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p13c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p13c4+b**2*(( \
            -1)+ErfCst)*p13c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p13c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p13c3+p13c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p13c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p13c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p13c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p13c3+3* \
            p13c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p13c2+v[i]*(p13c3+p13c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p13c1+p13c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p13c1+v[i]*(p13c2+v[i]*(p13c3+p13c4*v[i])))))
        P_14[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p14c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p14c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p14c4+b**2*(( \
            -1)+ErfCst)*p14c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p14c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p14c3+p14c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p14c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p14c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p14c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p14c3+3* \
            p14c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p14c2+v[i]*(p14c3+p14c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p14c1+p14c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p14c1+v[i]*(p14c2+v[i]*(p14c3+p14c4*v[i])))))
        P_15[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p15c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p15c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p15c4+b**2*(( \
            -1)+ErfCst)*p15c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p15c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p15c3+p15c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p15c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p15c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p15c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p15c3+3* \
            p15c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p15c2+v[i]*(p15c3+p15c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p15c1+p15c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p15c1+v[i]*(p15c2+v[i]*(p15c3+p15c4*v[i])))))
        
        
        
        
        
        
        P_output = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13, \
            P_14, P_15))
        
         #
    #;
    #;
    return P_output