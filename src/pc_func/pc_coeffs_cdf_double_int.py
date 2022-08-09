import numpy as np
from .pc_util import erf
from numba import njit, float64
from .pc_coeffs_second_int import pc_coeffs_second_int

@njit(
    float64[:,:,:](float64[:,:],float64[:],float64[:],float64[:],
                   float64[:],float64[:],float64[:],float64[:],
                   float64[:],float64[:],float64[:],float64[:]),
    fastmath=True,
    cache=True
)
def pc_coeffs_cdf_double_int(v, muY, sigmaMuY, sigmaY, \
        amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT): #
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
    
    ## PC Coefficients from Second Integration!!;
    P_output_2 = pc_coeffs_second_int(
        muY, sigmaMuY, sigmaY, \
        amuZ, bmuZ, sigmaMuZ, sigmaZ, \
        amuT, bmuT, sigmaMuT, sigmaT)
    
    # pull results
    p1c0 = np.zeros((1,n_sites))
    p2c0 = np.zeros((1,n_sites))
    p2c1 = np.zeros((1,n_sites))
    p3c0 = np.zeros((1,n_sites))
    p3c1 = np.zeros((1,n_sites))
    p4c0 = np.zeros((1,n_sites))
    p4c1 = np.zeros((1,n_sites))
    p5c0 = np.zeros((1,n_sites))
    p5c1 = np.zeros((1,n_sites))
    p5c2 = np.zeros((1,n_sites))
    p6c0 = np.zeros((1,n_sites))
    p6c1 = np.zeros((1,n_sites))
    p6c2 = np.zeros((1,n_sites))
    p7c0 = np.zeros((1,n_sites))
    p7c1 = np.zeros((1,n_sites))
    p7c2 = np.zeros((1,n_sites))
    p8c0 = np.zeros((1,n_sites))
    p8c1 = np.zeros((1,n_sites))
    p8c2 = np.zeros((1,n_sites))
    p9c0 = np.zeros((1,n_sites))
    p9c1 = np.zeros((1,n_sites))
    p9c2 = np.zeros((1,n_sites))
    p10c0 = np.zeros((1,n_sites))
    p10c1 = np.zeros((1,n_sites))
    p10c2 = np.zeros((1,n_sites))
    p11c0 = np.zeros((1,n_sites))
    p11c1 = np.zeros((1,n_sites))
    p11c2 = np.zeros((1,n_sites))
    p11c3 = np.zeros((1,n_sites))
    p12c0 = np.zeros((1,n_sites))
    p12c1 = np.zeros((1,n_sites))
    p12c2 = np.zeros((1,n_sites))
    p12c3 = np.zeros((1,n_sites))
    p13c0 = np.zeros((1,n_sites))
    p13c1 = np.zeros((1,n_sites))
    p13c2 = np.zeros((1,n_sites))
    p13c3 = np.zeros((1,n_sites))
    p14c0 = np.zeros((1,n_sites))
    p14c1 = np.zeros((1,n_sites))
    p14c2 = np.zeros((1,n_sites))
    p14c3 = np.zeros((1,n_sites))
    p15c0 = np.zeros((1,n_sites))
    p15c1 = np.zeros((1,n_sites))
    p15c2 = np.zeros((1,n_sites))
    p15c3 = np.zeros((1,n_sites))
    p16c0 = np.zeros((1,n_sites))
    p16c1 = np.zeros((1,n_sites))
    p16c2 = np.zeros((1,n_sites))
    p16c3 = np.zeros((1,n_sites))
    p17c0 = np.zeros((1,n_sites))
    p17c1 = np.zeros((1,n_sites))
    p17c2 = np.zeros((1,n_sites))
    p17c3 = np.zeros((1,n_sites))
    p18c0 = np.zeros((1,n_sites))
    p18c1 = np.zeros((1,n_sites))
    p18c2 = np.zeros((1,n_sites))
    p18c3 = np.zeros((1,n_sites))
    p19c0 = np.zeros((1,n_sites))
    p19c1 = np.zeros((1,n_sites))
    p19c2 = np.zeros((1,n_sites))
    p19c3 = np.zeros((1,n_sites))
    p20c0 = np.zeros((1,n_sites))
    p20c1 = np.zeros((1,n_sites))
    p20c2 = np.zeros((1,n_sites))
    p20c3 = np.zeros((1,n_sites))
    p21c0 = np.zeros((1,n_sites))
    p21c1 = np.zeros((1,n_sites))
    p21c2 = np.zeros((1,n_sites))
    p21c3 = np.zeros((1,n_sites))
    p21c4 = np.zeros((1,n_sites))
    p22c0 = np.zeros((1,n_sites))
    p22c1 = np.zeros((1,n_sites))
    p22c2 = np.zeros((1,n_sites))
    p22c3 = np.zeros((1,n_sites))
    p22c4 = np.zeros((1,n_sites))
    p23c0 = np.zeros((1,n_sites))
    p23c1 = np.zeros((1,n_sites))
    p23c2 = np.zeros((1,n_sites))
    p23c3 = np.zeros((1,n_sites))
    p23c4 = np.zeros((1,n_sites))
    p24c0 = np.zeros((1,n_sites))
    p24c1 = np.zeros((1,n_sites))
    p24c2 = np.zeros((1,n_sites))
    p24c3 = np.zeros((1,n_sites))
    p24c4 = np.zeros((1,n_sites))
    p25c0 = np.zeros((1,n_sites))
    p25c1 = np.zeros((1,n_sites))
    p25c2 = np.zeros((1,n_sites))
    p25c3 = np.zeros((1,n_sites))
    p25c4 = np.zeros((1,n_sites))
    p26c0 = np.zeros((1,n_sites))
    p26c1 = np.zeros((1,n_sites))
    p26c2 = np.zeros((1,n_sites))
    p26c3 = np.zeros((1,n_sites))
    p26c4 = np.zeros((1,n_sites))
    p27c0 = np.zeros((1,n_sites))
    p27c1 = np.zeros((1,n_sites))
    p27c2 = np.zeros((1,n_sites))
    p27c3 = np.zeros((1,n_sites))
    p27c4 = np.zeros((1,n_sites))
    p28c0 = np.zeros((1,n_sites))
    p28c1 = np.zeros((1,n_sites))
    p28c2 = np.zeros((1,n_sites))
    p28c3 = np.zeros((1,n_sites))
    p28c4 = np.zeros((1,n_sites))
    p29c0 = np.zeros((1,n_sites))
    p29c1 = np.zeros((1,n_sites))
    p29c2 = np.zeros((1,n_sites))
    p29c3 = np.zeros((1,n_sites))
    p29c4 = np.zeros((1,n_sites))
    p30c0 = np.zeros((1,n_sites))
    p30c1 = np.zeros((1,n_sites))
    p30c2 = np.zeros((1,n_sites))
    p30c3 = np.zeros((1,n_sites))
    p30c4 = np.zeros((1,n_sites))
    p31c0 = np.zeros((1,n_sites))
    p31c1 = np.zeros((1,n_sites))
    p31c2 = np.zeros((1,n_sites))
    p31c3 = np.zeros((1,n_sites))
    p31c4 = np.zeros((1,n_sites))
    p32c0 = np.zeros((1,n_sites))
    p32c1 = np.zeros((1,n_sites))
    p32c2 = np.zeros((1,n_sites))
    p32c3 = np.zeros((1,n_sites))
    p32c4 = np.zeros((1,n_sites))
    p33c0 = np.zeros((1,n_sites))
    p33c1 = np.zeros((1,n_sites))
    p33c2 = np.zeros((1,n_sites))
    p33c3 = np.zeros((1,n_sites))
    p33c4 = np.zeros((1,n_sites))
    p34c0 = np.zeros((1,n_sites))
    p34c1 = np.zeros((1,n_sites))
    p34c2 = np.zeros((1,n_sites))
    p34c3 = np.zeros((1,n_sites))
    p34c4 = np.zeros((1,n_sites))
    p35c0 = np.zeros((1,n_sites))
    p35c1 = np.zeros((1,n_sites))
    p35c2 = np.zeros((1,n_sites))
    p35c3 = np.zeros((1,n_sites))
    p35c4 = np.zeros((1,n_sites))
    
    # pull results and reshape
    p1c0[0,:] = P_output_2[0, :]
    p2c0[0,:] = P_output_2[1, :]
    p2c1[0,:] = P_output_2[2, :]
    p3c0[0,:] = P_output_2[3, :]
    p3c1[0,:] = P_output_2[4, :]
    p4c0[0,:] = P_output_2[5, :]
    p4c1[0,:] = P_output_2[6, :]
    p5c0[0,:] = P_output_2[7, :]
    p5c1[0,:] = P_output_2[8, :]
    p5c2[0,:] = P_output_2[9, :]
    p6c0[0,:] = P_output_2[10, :]
    p6c1[0,:] = P_output_2[11, :]
    p6c2[0,:] = P_output_2[12, :]
    p7c0[0,:] = P_output_2[13, :]
    p7c1[0,:] = P_output_2[14, :]
    p7c2[0,:] = P_output_2[15, :]
    p8c0[0,:] = P_output_2[16, :]
    p8c1[0,:] = P_output_2[17, :]
    p8c2[0,:] = P_output_2[18, :]
    p9c0[0,:] = P_output_2[19, :]
    p9c1[0,:] = P_output_2[20, :]
    p9c2[0,:] = P_output_2[21, :]
    p10c0[0,:] = P_output_2[22, :]
    p10c1[0,:] = P_output_2[23, :]
    p10c2[0,:] = P_output_2[24, :]
    p11c0[0,:] = P_output_2[25, :]
    p11c1[0,:] = P_output_2[26, :]
    p11c2[0,:] = P_output_2[27, :]
    p11c3[0,:] = P_output_2[28, :]
    p12c0[0,:] = P_output_2[29, :]
    p12c1[0,:] = P_output_2[30, :]
    p12c2[0,:] = P_output_2[31, :]
    p12c3[0,:] = P_output_2[32, :]
    p13c0[0,:] = P_output_2[33, :]
    p13c1[0,:] = P_output_2[34, :]
    p13c2[0,:] = P_output_2[35, :]
    p13c3[0,:] = P_output_2[36, :]
    p14c0[0,:] = P_output_2[37, :]
    p14c1[0,:] = P_output_2[38, :]
    p14c2[0,:] = P_output_2[39, :]
    p14c3[0,:] = P_output_2[40, :]
    p15c0[0,:] = P_output_2[41, :]
    p15c1[0,:] = P_output_2[42, :]
    p15c2[0,:] = P_output_2[43, :]
    p15c3[0,:] = P_output_2[44, :]
    p16c0[0,:] = P_output_2[45, :]
    p16c1[0,:] = P_output_2[46, :]
    p16c2[0,:] = P_output_2[47, :]
    p16c3[0,:] = P_output_2[48, :]
    p17c0[0,:] = P_output_2[49, :]
    p17c1[0,:] = P_output_2[50, :]
    p17c2[0,:] = P_output_2[51, :]
    p17c3[0,:] = P_output_2[52, :]
    p18c0[0,:] = P_output_2[53, :]
    p18c1[0,:] = P_output_2[54, :]
    p18c2[0,:] = P_output_2[55, :]
    p18c3[0,:] = P_output_2[56, :]
    p19c0[0,:] = P_output_2[57, :]
    p19c1[0,:] = P_output_2[58, :]
    p19c2[0,:] = P_output_2[59, :]
    p19c3[0,:] = P_output_2[60, :]
    p20c0[0,:] = P_output_2[61, :]
    p20c1[0,:] = P_output_2[62, :]
    p20c2[0,:] = P_output_2[63, :]
    p20c3[0,:] = P_output_2[64, :]
    p21c0[0,:] = P_output_2[65, :]
    p21c1[0,:] = P_output_2[66, :]
    p21c2[0,:] = P_output_2[67, :]
    p21c3[0,:] = P_output_2[68, :]
    p21c4[0,:] = P_output_2[69, :]
    p22c0[0,:] = P_output_2[70, :]
    p22c1[0,:] = P_output_2[71, :]
    p22c2[0,:] = P_output_2[72, :]
    p22c3[0,:] = P_output_2[73, :]
    p22c4[0,:] = P_output_2[74, :]
    p23c0[0,:] = P_output_2[75, :]
    p23c1[0,:] = P_output_2[76, :]
    p23c2[0,:] = P_output_2[77, :]
    p23c3[0,:] = P_output_2[78, :]
    p23c4[0,:] = P_output_2[79, :]
    p24c0[0,:] = P_output_2[80, :]
    p24c1[0,:] = P_output_2[81, :]
    p24c2[0,:] = P_output_2[82, :]
    p24c3[0,:] = P_output_2[83, :]
    p24c4[0,:] = P_output_2[84, :]
    p25c0[0,:] = P_output_2[85, :]
    p25c1[0,:] = P_output_2[86, :]
    p25c2[0,:] = P_output_2[87, :]
    p25c3[0,:] = P_output_2[88, :]
    p25c4[0,:] = P_output_2[89, :]
    p26c0[0,:] = P_output_2[90, :]
    p26c1[0,:] = P_output_2[91, :]
    p26c2[0,:] = P_output_2[92, :]
    p26c3[0,:] = P_output_2[93, :]
    p26c4[0,:] = P_output_2[94, :]
    p27c0[0,:] = P_output_2[95, :]
    p27c1[0,:] = P_output_2[96, :]
    p27c2[0,:] = P_output_2[97, :]
    p27c3[0,:] = P_output_2[98, :]
    p27c4[0,:] = P_output_2[99, :]
    p28c0[0,:] = P_output_2[100, :]
    p28c1[0,:] = P_output_2[101, :]
    p28c2[0,:] = P_output_2[102, :]
    p28c3[0,:] = P_output_2[103, :]
    p28c4[0,:] = P_output_2[104, :]
    p29c0[0,:] = P_output_2[105, :]
    p29c1[0,:] = P_output_2[106, :]
    p29c2[0,:] = P_output_2[107, :]
    p29c3[0,:] = P_output_2[108, :]
    p29c4[0,:] = P_output_2[109, :]
    p30c0[0,:] = P_output_2[110, :]
    p30c1[0,:] = P_output_2[111, :]
    p30c2[0,:] = P_output_2[112, :]
    p30c3[0,:] = P_output_2[113, :]
    p30c4[0,:] = P_output_2[114, :]
    p31c0[0,:] = P_output_2[115, :]
    p31c1[0,:] = P_output_2[116, :]
    p31c2[0,:] = P_output_2[117, :]
    p31c3[0,:] = P_output_2[118, :]
    p31c4[0,:] = P_output_2[119, :]
    p32c0[0,:] = P_output_2[120, :]
    p32c1[0,:] = P_output_2[121, :]
    p32c2[0,:] = P_output_2[122, :]
    p32c3[0,:] = P_output_2[123, :]
    p32c4[0,:] = P_output_2[124, :]
    p33c0[0,:] = P_output_2[125, :]
    p33c1[0,:] = P_output_2[126, :]
    p33c2[0,:] = P_output_2[127, :]
    p33c3[0,:] = P_output_2[128, :]
    p33c4[0,:] = P_output_2[129, :]
    p34c0[0,:] = P_output_2[130, :]
    p34c1[0,:] = P_output_2[131, :]
    p34c2[0,:] = P_output_2[132, :]
    p34c3[0,:] = P_output_2[133, :]
    p34c4[0,:] = P_output_2[134, :]
    p35c0[0,:] = P_output_2[135, :]
    p35c1[0,:] = P_output_2[136, :]
    p35c2[0,:] = P_output_2[137, :]
    p35c3[0,:] = P_output_2[138, :]
    p35c4[0,:] = P_output_2[139, :]
    
    # precalculate
    a = (-1/2)*(sigmaMuT**2+sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*( \
        sigmaMuY**2+sigmaY**2)+sigmaZ**2))**(-1)
    b = (bmuT+amuT*(bmuZ+amuZ*muY))*(sigmaMuT**2+sigmaT**2+amuT**2*( \
        sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2))**(-1)

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
    P_4 = (1/4)*a**(-1)*(2*expCst*p4c1+((-1)*a)**(-3/2)*a*((-1)+ErfCst)*( \
        2*a*p4c0+(-1)*b*p4c1)*np.pi**(1/2))
    P_5 = (1/8)*((-1)*a)**(-5/2)*(b*p5c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
        *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p5c0*np.pi**( \
        1/2)+2*a*(((-1)+ErfCst)*(b*p5c1+p5c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p5c1+p5c2*v)))
    P_6 = (1/8)*((-1)*a)**(-5/2)*(b*p6c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
        *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p6c0*np.pi**( \
        1/2)+2*a*(((-1)+ErfCst)*(b*p6c1+p6c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p6c1+p6c2*v)))
    P_7 = (1/8)*((-1)*a)**(-5/2)*(b*p7c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
        *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p7c0*np.pi**( \
        1/2)+2*a*(((-1)+ErfCst)*(b*p7c1+p7c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p7c1+p7c2*v)))
    P_8 = (1/8)*((-1)*a)**(-5/2)*(b*p8c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
        *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p8c0*np.pi**( \
        1/2)+2*a*(((-1)+ErfCst)*(b*p8c1+p8c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p8c1+p8c2*v)))
    P_9 = (1/8)*((-1)*a)**(-5/2)*(b*p9c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
        *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p9c0*np.pi**( \
        1/2)+2*a*(((-1)+ErfCst)*(b*p9c1+p9c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p9c1+p9c2*v)))
    P_10 = (1/8)*((-1)*a)**(-5/2)*(b*p10c2*((-2)*((-1)*a)**(1/2)*expCst+( \
        -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p10c0* \
        np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p10c1+p10c2)*np.pi**(1/2)+2*((-1)*a) \
        **(1/2)*expCst*(p10c1+p10c2*v)))
    P_11 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p11c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p11c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p11c3+b**2*((-1)+ErfCst)* \
        p11c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p11c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p11c2+p11c3*v))+4*a**2*(((-1)+ErfCst)*(b*p11c1+ \
        p11c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p11c1+v*(p11c2+p11c3* \
        v))))
    P_12 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p12c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p12c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p12c3+b**2*((-1)+ErfCst)* \
        p12c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p12c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p12c2+p12c3*v))+4*a**2*(((-1)+ErfCst)*(b*p12c1+ \
        p12c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p12c1+v*(p12c2+p12c3* \
        v))))
    P_13 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p13c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p13c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p13c3+b**2*((-1)+ErfCst)* \
        p13c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p13c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p13c2+p13c3*v))+4*a**2*(((-1)+ErfCst)*(b*p13c1+ \
        p13c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p13c1+v*(p13c2+p13c3* \
        v))))
    P_14 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p14c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p14c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p14c3+b**2*((-1)+ErfCst)* \
        p14c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p14c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p14c2+p14c3*v))+4*a**2*(((-1)+ErfCst)*(b*p14c1+ \
        p14c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p14c1+v*(p14c2+p14c3* \
        v))))
    P_15 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p15c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p15c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p15c3+b**2*((-1)+ErfCst)* \
        p15c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p15c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p15c2+p15c3*v))+4*a**2*(((-1)+ErfCst)*(b*p15c1+ \
        p15c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p15c1+v*(p15c2+p15c3* \
        v))))
    P_16 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p16c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p16c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p16c3+b**2*((-1)+ErfCst)* \
        p16c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p16c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p16c2+p16c3*v))+4*a**2*(((-1)+ErfCst)*(b*p16c1+ \
        p16c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p16c1+v*(p16c2+p16c3* \
        v))))
    P_17 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p17c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p17c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p17c3+b**2*((-1)+ErfCst)* \
        p17c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p17c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p17c2+p17c3*v))+4*a**2*(((-1)+ErfCst)*(b*p17c1+ \
        p17c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p17c1+v*(p17c2+p17c3* \
        v))))
    P_18 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p18c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p18c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p18c3+b**2*((-1)+ErfCst)* \
        p18c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p18c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p18c2+p18c3*v))+4*a**2*(((-1)+ErfCst)*(b*p18c1+ \
        p18c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p18c1+v*(p18c2+p18c3* \
        v))))
    P_19 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p19c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p19c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p19c3+b**2*((-1)+ErfCst)* \
        p19c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p19c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p19c2+p19c3*v))+4*a**2*(((-1)+ErfCst)*(b*p19c1+ \
        p19c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p19c1+v*(p19c2+p19c3* \
        v))))
    P_20 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p20c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p20c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p20c3+b**2*((-1)+ErfCst)* \
        p20c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p20c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p20c2+p20c3*v))+4*a**2*(((-1)+ErfCst)*(b*p20c1+ \
        p20c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p20c1+v*(p20c2+p20c3* \
        v))))
    P_21 = (1/32)*((-1)*a)**(-9/2)*(b**3*p21c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p21c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p21c4+b**2*(( \
        -1)+ErfCst)*p21c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p21c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p21c3+p21c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p21c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p21c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p21c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p21c3+3* \
        p21c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p21c2+v*(p21c3+p21c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p21c1+p21c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p21c1+v*(p21c2+v*(p21c3+p21c4*v)))))
    P_22 = (1/32)*((-1)*a)**(-9/2)*(b**3*p22c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p22c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p22c4+b**2*(( \
        -1)+ErfCst)*p22c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p22c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p22c3+p22c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p22c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p22c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p22c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p22c3+3* \
        p22c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p22c2+v*(p22c3+p22c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p22c1+p22c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p22c1+v*(p22c2+v*(p22c3+p22c4*v)))))
    P_23 = (1/32)*((-1)*a)**(-9/2)*(b**3*p23c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p23c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p23c4+b**2*(( \
        -1)+ErfCst)*p23c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p23c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p23c3+p23c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p23c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p23c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p23c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p23c3+3* \
        p23c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p23c2+v*(p23c3+p23c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p23c1+p23c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p23c1+v*(p23c2+v*(p23c3+p23c4*v)))))
    P_24 = (1/32)*((-1)*a)**(-9/2)*(b**3*p24c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p24c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p24c4+b**2*(( \
        -1)+ErfCst)*p24c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p24c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p24c3+p24c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p24c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p24c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p24c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p24c3+3* \
        p24c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p24c2+v*(p24c3+p24c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p24c1+p24c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p24c1+v*(p24c2+v*(p24c3+p24c4*v)))))
    P_25 = (1/32)*((-1)*a)**(-9/2)*(b**3*p25c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p25c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p25c4+b**2*(( \
        -1)+ErfCst)*p25c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p25c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p25c3+p25c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p25c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p25c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p25c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p25c3+3* \
        p25c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p25c2+v*(p25c3+p25c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p25c1+p25c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p25c1+v*(p25c2+v*(p25c3+p25c4*v)))))
    P_26 = (1/32)*((-1)*a)**(-9/2)*(b**3*p26c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p26c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p26c4+b**2*(( \
        -1)+ErfCst)*p26c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p26c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p26c3+p26c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p26c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p26c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p26c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p26c3+3* \
        p26c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p26c2+v*(p26c3+p26c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p26c1+p26c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p26c1+v*(p26c2+v*(p26c3+p26c4*v)))))
    P_27 = (1/32)*((-1)*a)**(-9/2)*(b**3*p27c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p27c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p27c4+b**2*(( \
        -1)+ErfCst)*p27c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p27c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p27c3+p27c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p27c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p27c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p27c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p27c3+3* \
        p27c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p27c2+v*(p27c3+p27c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p27c1+p27c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p27c1+v*(p27c2+v*(p27c3+p27c4*v)))))
    P_28 = (1/32)*((-1)*a)**(-9/2)*(b**3*p28c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p28c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p28c4+b**2*(( \
        -1)+ErfCst)*p28c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p28c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p28c3+p28c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p28c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p28c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p28c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p28c3+3* \
        p28c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p28c2+v*(p28c3+p28c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p28c1+p28c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p28c1+v*(p28c2+v*(p28c3+p28c4*v)))))
    P_29 = (1/32)*((-1)*a)**(-9/2)*(b**3*p29c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p29c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p29c4+b**2*(( \
        -1)+ErfCst)*p29c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p29c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p29c3+p29c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p29c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p29c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p29c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p29c3+3* \
        p29c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p29c2+v*(p29c3+p29c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p29c1+p29c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p29c1+v*(p29c2+v*(p29c3+p29c4*v)))))
    P_30 = (1/32)*((-1)*a)**(-9/2)*(b**3*p30c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p30c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p30c4+b**2*(( \
        -1)+ErfCst)*p30c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p30c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p30c3+p30c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p30c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p30c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p30c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p30c3+3* \
        p30c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p30c2+v*(p30c3+p30c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p30c1+p30c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p30c1+v*(p30c2+v*(p30c3+p30c4*v)))))
    P_31 = (1/32)*((-1)*a)**(-9/2)*(b**3*p31c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p31c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p31c4+b**2*(( \
        -1)+ErfCst)*p31c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p31c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p31c3+p31c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p31c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p31c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p31c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p31c3+3* \
        p31c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p31c2+v*(p31c3+p31c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p31c1+p31c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p31c1+v*(p31c2+v*(p31c3+p31c4*v)))))
    P_32 = (1/32)*((-1)*a)**(-9/2)*(b**3*p32c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p32c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p32c4+b**2*(( \
        -1)+ErfCst)*p32c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p32c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p32c3+p32c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p32c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p32c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p32c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p32c3+3* \
        p32c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p32c2+v*(p32c3+p32c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p32c1+p32c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p32c1+v*(p32c2+v*(p32c3+p32c4*v)))))
    P_33 = (1/32)*((-1)*a)**(-9/2)*(b**3*p33c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p33c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p33c4+b**2*(( \
        -1)+ErfCst)*p33c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p33c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p33c3+p33c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p33c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p33c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p33c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p33c3+3* \
        p33c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p33c2+v*(p33c3+p33c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p33c1+p33c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p33c1+v*(p33c2+v*(p33c3+p33c4*v)))))
    P_34 = (1/32)*((-1)*a)**(-9/2)*(b**3*p34c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p34c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p34c4+b**2*(( \
        -1)+ErfCst)*p34c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p34c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p34c3+p34c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p34c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p34c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p34c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p34c3+3* \
        p34c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p34c2+v*(p34c3+p34c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p34c1+p34c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p34c1+v*(p34c2+v*(p34c3+p34c4*v)))))
    P_35 = (1/32)*((-1)*a)**(-9/2)*(b**3*p35c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p35c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p35c4+b**2*(( \
        -1)+ErfCst)*p35c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p35c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p35c3+p35c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p35c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p35c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p35c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p35c3+3* \
        p35c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p35c2+v*(p35c3+p35c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p35c1+p35c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p35c1+v*(p35c2+v*(p35c3+p35c4*v)))))
        
    # store to output
    # P_output = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13,
    #     P_14, P_15, P_16, P_17, P_18, P_19, P_20, P_21, P_22, P_23, P_24, P_25,
    #     P_26, P_27, P_28, P_29, P_30, P_31, P_32, P_33, P_34, P_35))
    P_output = np.zeros((n_pts_v, n_sites, 35))
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
    P_output[:,:,15] = P_16
    P_output[:,:,16] = P_17
    P_output[:,:,17] = P_18
    P_output[:,:,18] = P_19
    P_output[:,:,19] = P_20
    P_output[:,:,20] = P_21
    P_output[:,:,21] = P_22
    P_output[:,:,22] = P_23
    P_output[:,:,23] = P_24
    P_output[:,:,24] = P_25
    P_output[:,:,25] = P_26
    P_output[:,:,26] = P_27
    P_output[:,:,27] = P_28
    P_output[:,:,28] = P_29
    P_output[:,:,29] = P_30
    P_output[:,:,30] = P_31
    P_output[:,:,31] = P_32
    P_output[:,:,32] = P_33
    P_output[:,:,33] = P_34
    P_output[:,:,34] = P_35
    
    # return
    return P_output
