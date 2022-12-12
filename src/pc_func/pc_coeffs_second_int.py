import numpy as np
from numba import njit, float64
from .pc_coeffs_first_int import pc_coeffs_first_int

@njit(
    float64[:,:](float64[:],float64[:],float64[:],
                 float64[:],float64[:],float64[:],float64[:],
                 float64[:],float64[:],float64[:],float64[:]),
    fastmath=True,
    cache=True
)
def pc_coeffs_second_int(
    muY, sigmaMuY, sigmaY, \
    amuZ, bmuZ, sigmaMuZ, sigmaZ,
    amuT, bmuT, sigmaMuT, sigmaT
):
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
    n_sites  =  len(muY)
    
    ## PC Coefficients from First Integration!!;
    P_output_1 = pc_coeffs_first_int(muY, sigmaMuY, \
        sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ)
    
    # pull results and reshape
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
    ##;
    ## Total Constant;
    #;
    a1 = (1/4)*((-2)*amuT**2*(sigmaMuT**2+sigmaT**2)**(-1)+(-2)*(sigmaMuZ**2+ \
        amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)**(-1))
    b00 = (sigmaMuT**2+sigmaT**2)**(-1)*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+ \
        sigmaY**2)+sigmaZ**2)**(-1)*(bmuZ*(sigmaMuT**2+sigmaT**2)+amuZ*muY*( \
        sigmaMuT**2+sigmaT**2)+(-1)*amuT*amuZ**2*bmuT*(sigmaMuY**2+ \
        sigmaY**2)+(-1)*amuT*bmuT*(sigmaMuZ**2+sigmaZ**2))
    b01 = amuT*(sigmaMuT**2+sigmaT**2)**(-1)

    P_1 =  np.zeros((1, n_sites))
    P_2 =  np.zeros((2, n_sites))
    P_3 =  np.zeros((2, n_sites))
    P_4 =  np.zeros((2, n_sites))
    P_5 =  np.zeros((3, n_sites))
    P_6 =  np.zeros((3, n_sites))
    P_7 =  np.zeros((3, n_sites))
    P_8 =  np.zeros((3, n_sites))
    P_9 =  np.zeros((3, n_sites))
    P_10 =  np.zeros((3, n_sites))
    P_11 =  np.zeros((4, n_sites))
    P_12 =  np.zeros((4, n_sites))
    P_13 =  np.zeros((4, n_sites))
    P_14 =  np.zeros((4, n_sites))
    P_15 =  np.zeros((4, n_sites))
    P_16 =  np.zeros((4, n_sites))
    P_17 =  np.zeros((4, n_sites))
    P_18 =  np.zeros((4, n_sites))
    P_19 =  np.zeros((4, n_sites))
    P_20 =  np.zeros((4, n_sites))
    P_21 =  np.zeros((5, n_sites))
    P_22 =  np.zeros((5, n_sites))
    P_23 =  np.zeros((5, n_sites))
    P_24 =  np.zeros((5, n_sites))
    P_25 =  np.zeros((5, n_sites))
    P_26 =  np.zeros((5, n_sites))
    P_27 =  np.zeros((5, n_sites))
    P_28 =  np.zeros((5, n_sites))
    P_29 =  np.zeros((5, n_sites))
    P_30 =  np.zeros((5, n_sites))
    P_31 =  np.zeros((5, n_sites))
    P_32 =  np.zeros((5, n_sites))
    P_33 =  np.zeros((5, n_sites))
    P_34 =  np.zeros((5, n_sites))
    P_35 =  np.zeros((5, n_sites))
    #;
    ## Polynomial Constants;
    #;
    constantTerm2 = 2**(-1/2)*((-1)*a1)**(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-1/2)* \
        sigmaT**(-1)
    P_C_1 = constantTerm2
    P_C_2 = (-1)*constantTerm2*(2*a1*sigmaMuT**2+2*a1*sigmaT**2)**(-1)
    P_C_3 = (-1/2)*a1**(-1)*constantTerm2
    P_C_4 = (-1/2)*a1**(-1)*constantTerm2
    P_C_5 = (1/8)*a1**(-2)*constantTerm2*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT*(sigmaMuT**2+sigmaT**2)**(-3)
    P_C_6 = (1/4)*a1**(-2)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_7 = (1/4)*a1**(-2)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_8 = (1/4)*a1**(-2)*constantTerm2
    P_C_9 = (1/4)*a1**(-2)*constantTerm2
    P_C_10 = (1/4)*a1**(-2)*constantTerm2
    P_C_11 = (-1/48)*a1**(-3)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-3)
    P_C_12 = (-1/16)*a1**(-3)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-2)
    P_C_13 = (-1/16)*a1**(-3)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-2)
    P_C_14 = (-1/8)*a1**(-3)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_15 = (-1/8)*a1**(-3)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_16 = (-1/8)*a1**(-3)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_17 = (-1/8)*a1**(-3)*constantTerm2
    P_C_18 = (-1/8)*a1**(-3)*constantTerm2
    P_C_19 = (-1/8)*a1**(-3)*constantTerm2
    P_C_20 = (-1/8)*a1**(-3)*constantTerm2
    P_C_21 = (1/384)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-4)
    P_C_22 = (1/96)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-3)
    P_C_23 = (1/96)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-3)
    P_C_24 = (1/32)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-2)
    P_C_25 = (1/32)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-2)
    P_C_26 = (1/32)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-2)
    P_C_27 = (1/16)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_28 = (1/16)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_29 = (1/16)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_30 = (1/16)*a1**(-4)*constantTerm2*(sigmaMuT**2+sigmaT**2)**(-1)
    P_C_31 = (1/16)*a1**(-4)*constantTerm2
    P_C_32 = (1/16)*a1**(-4)*constantTerm2
    P_C_33 = (1/16)*a1**(-4)*constantTerm2
    P_C_34 = (1/16)*a1**(-4)*constantTerm2
    P_C_35 = (1/16)*a1**(-4)*constantTerm2
    #;
    ## Polynomial Coefficients;
    #;
    P_1[0,:] = (P_C_1)*p1c0
    P_2[0,:] = (P_C_2)*((-1)*amuT*b00+2*a1*bmuT)*p1c0*sigmaMuT
    P_2[1,:] = (-1)*(P_C_2)*(2*a1+amuT*b01)*p1c0*sigmaMuT
    P_3[0,:] = (P_C_3)*((-2)*a1*p2c0+b00*p2c1)
    P_3[1,:] = (P_C_3)*b01*p2c1
    P_4[0,:] = (P_C_4)*((-2)*a1*p3c0+b00*p3c1)
    P_4[1,:] = (P_C_4)*b01*p3c1
    P_5[0,:] = (P_C_5)*p1c0*sigmaMuT**2*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT*(amuT**2*b00**2+(-2)*a1*amuT*(amuT+2*b00*bmuT)+4*a1**2*( \
        bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2))
    P_5[1,:] = (-2)*(P_C_5)*(2*a1+amuT*b01)*((-1)*amuT*b00+2*a1*bmuT)*p1c0* \
        sigmaMuT**2*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT
    P_5[2,:] = (P_C_5)*(2*a1+amuT*b01)**2*p1c0*sigmaMuT**2*(1+sigmaMuT**2* \
        sigmaT**(-2))**(1/2)*sigmaT
    P_6[0,:] = (P_C_6)*((-4)*a1**2*bmuT*p2c0+(-1)*amuT*b00**2*p2c1+2*a1*(b00* \
        bmuT*p2c1+amuT*(b00*p2c0+p2c1)))*sigmaMuT
    P_6[1,:] = 2*(P_C_6)*(2*a1**2*p2c0+(-1)*amuT*b00*b01*p2c1+a1*(amuT*b01* \
        p2c0+(-1)*b00*p2c1+b01*bmuT*p2c1))*sigmaMuT
    P_6[2,:] = (-1)*(P_C_6)*b01*(2*a1+amuT*b01)*p2c1*sigmaMuT
    P_7[0,:] = (P_C_7)*((-4)*a1**2*bmuT*p3c0+(-1)*amuT*b00**2*p3c1+2*a1*(b00* \
        bmuT*p3c1+amuT*(b00*p3c0+p3c1)))*sigmaMuT
    P_7[1,:] = 2*(P_C_7)*(2*a1**2*p3c0+(-1)*amuT*b00*b01*p3c1+a1*(amuT*b01* \
        p3c0+(-1)*b00*p3c1+b01*bmuT*p3c1))*sigmaMuT
    P_7[2,:] = (-1)*(P_C_7)*b01*(2*a1+amuT*b01)*p3c1*sigmaMuT
    P_8[0,:] = (P_C_8)*(4*a1**2*p4c0+b00**2*p4c2+(-2)*a1*(b00*p4c1+p4c2))
    P_8[1,:] = (P_C_8)*((-2)*a1*b01*p4c1+2*b00*b01*p4c2)
    P_8[2,:] = (P_C_8)*b01**2*p4c2
    P_9[0,:] = (P_C_9)*(4*a1**2*p5c0+b00**2*p5c2+(-2)*a1*(b00*p5c1+p5c2))
    P_9[1,:] = (P_C_9)*((-2)*a1*b01*p5c1+2*b00*b01*p5c2)
    P_9[2,:] = (P_C_9)*b01**2*p5c2
    P_10[0,:] = (P_C_10)*(4*a1**2*p6c0+b00**2*p6c2+(-2)*a1*(b00*p6c1+p6c2))
    P_10[1,:] = (P_C_10)*((-2)*a1*b01*p6c1+2*b00*b01*p6c2)
    P_10[2,:] = (P_C_10)*b01**2*p6c2
    P_11[0,:] = (P_C_11)*((-1)*amuT*b00+2*a1*bmuT)*p1c0*sigmaMuT**3*(amuT**2* \
        b00**2+(-2)*a1*amuT*(3*amuT+2*b00*bmuT)+4*a1**2*(bmuT**2+(-3)*( \
        sigmaMuT**2+sigmaT**2)))
    P_11[1,:] = (-3)*(P_C_11)*(2*a1+amuT*b01)*p1c0*sigmaMuT**3*(amuT**2*b00**2+( \
        -2)*a1*amuT*(amuT+2*b00*bmuT)+4*a1**2*(bmuT**2+(-1)*sigmaMuT**2+ \
        (-1)*sigmaT**2))
    P_11[2,:] = 3*(P_C_11)*(2*a1+amuT*b01)**2*((-1)*amuT*b00+2*a1*bmuT)*p1c0* \
        sigmaMuT**3
    P_11[3,:] = (-1)*(P_C_11)*(2*a1+amuT*b01)**3*p1c0*sigmaMuT**3
    P_12[0,:] = (P_C_12)*sigmaMuT**2*(amuT**2*b00**3*p2c1+(-2)*a1*amuT*b00*( \
        amuT*b00*p2c0+3*amuT*p2c1+2*b00*bmuT*p2c1)+8*a1**3*p2c0*((-1) \
        *bmuT**2+sigmaMuT**2+sigmaT**2)+4*a1**2*(amuT**2*p2c0+2*amuT* \
        bmuT*(b00*p2c0+p2c1)+b00*p2c1*(bmuT**2+(-1)*sigmaMuT**2+(-1)* \
        sigmaT**2)))
    P_12[1,:] = (-1)*(P_C_12)*sigmaMuT**2*((-16)*a1**3*bmuT*p2c0+(-3)*amuT**2* \
        b00**2*b01*p2c1+2*a1*amuT*((-2)*b00*(b00+(-2)*b01*bmuT)*p2c1+ \
        amuT*b01*(2*b00*p2c0+3*p2c1))+4*a1**2*(2*b00*bmuT*p2c1+2* \
        amuT*(b00*p2c0+(-1)*b01*bmuT*p2c0+p2c1)+b01*p2c1*((-1)*bmuT**2+ \
        sigmaMuT**2+sigmaT**2)))
    P_12[2,:] = (-1)*(P_C_12)*(2*a1+amuT*b01)*(4*a1**2*p2c0+(-3)*amuT*b00* \
        b01*p2c1+2*a1*(amuT*b01*p2c0+(-1)*b00*p2c1+2*b01*bmuT*p2c1))* \
        sigmaMuT**2
    P_12[3,:] = (P_C_12)*b01*(2*a1+amuT*b01)**2*p2c1*sigmaMuT**2
    P_13[0,:] = (P_C_13)*sigmaMuT**2*(amuT**2*b00**3*p3c1+(-2)*a1*amuT*b00*( \
        amuT*b00*p3c0+3*amuT*p3c1+2*b00*bmuT*p3c1)+8*a1**3*p3c0*((-1) \
        *bmuT**2+sigmaMuT**2+sigmaT**2)+4*a1**2*(amuT**2*p3c0+2*amuT* \
        bmuT*(b00*p3c0+p3c1)+b00*p3c1*(bmuT**2+(-1)*sigmaMuT**2+(-1)* \
        sigmaT**2)))
    P_13[1,:] = (-1)*(P_C_13)*sigmaMuT**2*((-16)*a1**3*bmuT*p3c0+(-3)*amuT**2* \
        b00**2*b01*p3c1+2*a1*amuT*((-2)*b00*(b00+(-2)*b01*bmuT)*p3c1+ \
        amuT*b01*(2*b00*p3c0+3*p3c1))+4*a1**2*(2*b00*bmuT*p3c1+2* \
        amuT*(b00*p3c0+(-1)*b01*bmuT*p3c0+p3c1)+b01*p3c1*((-1)*bmuT**2+ \
        sigmaMuT**2+sigmaT**2)))
    P_13[2,:] = (-1)*(P_C_13)*(2*a1+amuT*b01)*(4*a1**2*p3c0+(-3)*amuT*b00* \
        b01*p3c1+2*a1*(amuT*b01*p3c0+(-1)*b00*p3c1+2*b01*bmuT*p3c1))* \
        sigmaMuT**2
    P_13[3,:] = (P_C_13)*b01*(2*a1+amuT*b01)**2*p3c1*sigmaMuT**2
    P_14[0,:] = (P_C_14)*(8*a1**3*bmuT*p4c0+(-1)*amuT*b00**3*p4c2+2*a1*b00*( \
        amuT*b00*p4c1+3*amuT*p4c2+b00*bmuT*p4c2)+(-4)*a1**2*(amuT*( \
        b00*p4c0+p4c1)+bmuT*(b00*p4c1+p4c2)))*sigmaMuT
    P_14[1,:] = (P_C_14)*((-8)*a1**3*p4c0+(-4)*a1**2*(amuT*b01*p4c0+(-1)*b00* \
        p4c1+b01*bmuT*p4c1+(-1)*p4c2)+(-3)*amuT*b00**2*b01*p4c2+(-2)* \
        a1*b00*(b00+(-2)*b01*bmuT)*p4c2+a1*amuT*(4*b00*b01*p4c1+6* \
        b01*p4c2))*sigmaMuT
    P_14[2,:] = (P_C_14)*b01*(4*a1**2*p4c1+(-3)*amuT*b00*b01*p4c2+2*a1*(amuT* \
        b01*p4c1+(-2)*b00*p4c2+b01*bmuT*p4c2))*sigmaMuT
    P_14[3,:] = (-1)*(P_C_14)*b01**2*(2*a1+amuT*b01)*p4c2*sigmaMuT
    P_15[0,:] = (P_C_15)*(8*a1**3*bmuT*p5c0+(-1)*amuT*b00**3*p5c2+2*a1*b00*( \
        amuT*b00*p5c1+3*amuT*p5c2+b00*bmuT*p5c2)+(-4)*a1**2*(amuT*( \
        b00*p5c0+p5c1)+bmuT*(b00*p5c1+p5c2)))*sigmaMuT
    P_15[1,:] = (P_C_15)*((-8)*a1**3*p5c0+(-4)*a1**2*(amuT*b01*p5c0+(-1)*b00* \
        p5c1+b01*bmuT*p5c1+(-1)*p5c2)+(-3)*amuT*b00**2*b01*p5c2+(-2)* \
        a1*b00*(b00+(-2)*b01*bmuT)*p5c2+a1*amuT*(4*b00*b01*p5c1+6* \
        b01*p5c2))*sigmaMuT
    P_15[2,:] = (P_C_15)*b01*(4*a1**2*p5c1+(-3)*amuT*b00*b01*p5c2+2*a1*(amuT* \
        b01*p5c1+(-2)*b00*p5c2+b01*bmuT*p5c2))*sigmaMuT
    P_15[3,:] = (-1)*(P_C_15)*b01**2*(2*a1+amuT*b01)*p5c2*sigmaMuT
    P_16[0,:] = (P_C_16)*(8*a1**3*bmuT*p6c0+(-1)*amuT*b00**3*p6c2+2*a1*b00*( \
        amuT*b00*p6c1+3*amuT*p6c2+b00*bmuT*p6c2)+(-4)*a1**2*(amuT*( \
        b00*p6c0+p6c1)+bmuT*(b00*p6c1+p6c2)))*sigmaMuT
    P_16[1,:] = (P_C_16)*((-8)*a1**3*p6c0+(-4)*a1**2*(amuT*b01*p6c0+(-1)*b00* \
        p6c1+b01*bmuT*p6c1+(-1)*p6c2)+(-3)*amuT*b00**2*b01*p6c2+(-2)* \
        a1*b00*(b00+(-2)*b01*bmuT)*p6c2+a1*amuT*(4*b00*b01*p6c1+6* \
        b01*p6c2))*sigmaMuT
    P_16[2,:] = (P_C_16)*b01*(4*a1**2*p6c1+(-3)*amuT*b00*b01*p6c2+2*a1*(amuT* \
        b01*p6c1+(-2)*b00*p6c2+b01*bmuT*p6c2))*sigmaMuT
    P_16[3,:] = (-1)*(P_C_16)*b01**2*(2*a1+amuT*b01)*p6c2*sigmaMuT
    P_17[0,:] = (P_C_17)*((-8)*a1**3*p7c0+4*a1**2*(b00*p7c1+p7c2)+b00**3*p7c3+( \
        -2)*a1*b00*(b00*p7c2+3*p7c3))
    P_17[1,:] = (P_C_17)*b01*(4*a1**2*p7c1+(-4)*a1*b00*p7c2+(-6)*a1*p7c3+3* \
        b00**2*p7c3)
    P_17[2,:] = (P_C_17)*b01**2*((-2)*a1*p7c2+3*b00*p7c3)
    P_17[3,:] = (P_C_17)*b01**3*p7c3
    P_18[0,:] = (P_C_18)*((-8)*a1**3*p8c0+4*a1**2*(b00*p8c1+p8c2)+b00**3*p8c3+( \
        -2)*a1*b00*(b00*p8c2+3*p8c3))
    P_18[1,:] = (P_C_18)*b01*(4*a1**2*p8c1+(-4)*a1*b00*p8c2+(-6)*a1*p8c3+3* \
        b00**2*p8c3)
    P_18[2,:] = (P_C_18)*b01**2*((-2)*a1*p8c2+3*b00*p8c3)
    P_18[3,:] = (P_C_18)*b01**3*p8c3
    P_19[0,:] = (P_C_19)*((-8)*a1**3*p9c0+4*a1**2*(b00*p9c1+p9c2)+b00**3*p9c3+( \
        -2)*a1*b00*(b00*p9c2+3*p9c3))
    P_19[1,:] = (P_C_19)*b01*(4*a1**2*p9c1+(-4)*a1*b00*p9c2+(-6)*a1*p9c3+3* \
        b00**2*p9c3)
    P_19[2,:] = (P_C_19)*b01**2*((-2)*a1*p9c2+3*b00*p9c3)
    P_19[3,:] = (P_C_19)*b01**3*p9c3
    P_20[0,:] = (P_C_20)*((-8)*a1**3*p10c0+4*a1**2*(b00*p10c1+p10c2)+b00**3* \
        p10c3+(-2)*a1*b00*(b00*p10c2+3*p10c3))
    P_20[1,:] = (P_C_20)*b01*(4*a1**2*p10c1+(-4)*a1*b00*p10c2+(-6)*a1*p10c3+3* \
        b00**2*p10c3)
    P_20[2,:] = (P_C_20)*b01**2*((-2)*a1*p10c2+3*b00*p10c3)
    P_20[3,:] = (P_C_20)*b01**3*p10c3
    P_21[0,:] = (P_C_21)*p1c0*sigmaMuT**4*(amuT**4*b00**4+(-4)*a1*amuT**3* \
        b00**2*(3*amuT+2*b00*bmuT)+12*a1**2*amuT**2*(amuT**2+4*amuT* \
        b00*bmuT+2*b00**2*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2))+16* \
        a1**4*(bmuT**4+(-6)*bmuT**2*(sigmaMuT**2+sigmaT**2)+3*(sigmaMuT**2+ \
        sigmaT**2)**2)+(-16)*a1**3*amuT*(3*amuT*(bmuT**2+(-1)*sigmaMuT**2+ \
        (-1)*sigmaT**2)+2*b00*bmuT*(bmuT**2+(-3)*(sigmaMuT**2+sigmaT**2)))) \
        
    P_21[1,:] = (-4)*(P_C_21)*(2*a1+amuT*b01)*((-1)*amuT*b00+2*a1*bmuT)*p1c0* \
        sigmaMuT**4*(amuT**2*b00**2+(-2)*a1*amuT*(3*amuT+2*b00*bmuT)+4* \
        a1**2*(bmuT**2+(-3)*(sigmaMuT**2+sigmaT**2)))
    P_21[2,:] = 6*(P_C_21)*(2*a1+amuT*b01)**2*p1c0*sigmaMuT**4*(amuT**2*b00**2+( \
        -2)*a1*amuT*(amuT+2*b00*bmuT)+4*a1**2*(bmuT**2+(-1)*sigmaMuT**2+ \
        (-1)*sigmaT**2))
    P_21[3,:] = (-4)*(P_C_21)*(2*a1+amuT*b01)**3*((-1)*amuT*b00+2*a1*bmuT)* \
        p1c0*sigmaMuT**4
    P_21[4,:] = (P_C_21)*(2*a1+amuT*b01)**4*p1c0*sigmaMuT**4
    P_22[0,:] = (P_C_22)*sigmaMuT**3*((-1)*amuT**3*b00**4*p2c1+2*a1*amuT**2* \
        b00**2*(amuT*b00*p2c0+6*amuT*p2c1+3*b00*bmuT*p2c1)+(-12)* \
        a1**2*amuT*(amuT**2*(b00*p2c0+p2c1)+amuT*b00*bmuT*(b00*p2c0+3* \
        p2c1)+b00**2*p2c1*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2))+(-16)* \
        a1**4*bmuT*p2c0*(bmuT**2+(-3)*(sigmaMuT**2+sigmaT**2))+8*a1**3*( \
        3*amuT**2*bmuT*p2c0+3*amuT*(b00*p2c0+p2c1)*(bmuT**2+(-1)* \
        sigmaMuT**2+(-1)*sigmaT**2)+b00*bmuT*p2c1*(bmuT**2+(-3)*( \
        sigmaMuT**2+sigmaT**2))))
    P_22[1,:] = 2*(P_C_22)*sigmaMuT**3*((-2)*amuT**3*b00**3*b01*p2c1+3*a1* \
        amuT**2*b00*((-1)*b00*(b00+(-3)*b01*bmuT)*p2c1+amuT*b01*(b00* \
        p2c0+4*p2c1))+24*a1**4*p2c0*(bmuT**2+(-1)*sigmaMuT**2+(-1)* \
        sigmaT**2)+(-6)*a1**2*amuT*(amuT**2*b01*p2c0+amuT*((-1)*b00**2* \
        p2c0+2*b00*b01*bmuT*p2c0+(-3)*b00*p2c1+3*b01*bmuT*p2c1)+(-2)* \
        b00*p2c1*(b00*bmuT+b01*((-1)*bmuT**2+sigmaMuT**2+sigmaT**2)))+(-4) \
        *a1**3*(3*amuT**2*p2c0+p2c1*((-1)*b01*bmuT**3+3*b00*(bmuT**2+( \
        -1)*sigmaMuT**2+(-1)*sigmaT**2)+3*b01*bmuT*(sigmaMuT**2+sigmaT**2)) \
        +3*amuT*(2*b00*bmuT*p2c0+2*bmuT*p2c1+b01*p2c0*((-1)*bmuT**2+ \
        sigmaMuT**2+sigmaT**2))))
    P_22[2,:] = (-6)*(P_C_22)*(2*a1+amuT*b01)*sigmaMuT**3*(4*a1**3*bmuT*p2c0+ \
        amuT**2*b00**2*b01*p2c1+(-1)*a1*amuT*((-1)*b00*(b00+(-3)*b01* \
        bmuT)*p2c1+amuT*b01*(b00*p2c0+2*p2c1))+(-2)*a1**2*(b00*bmuT* \
        p2c1+amuT*(b00*p2c0+(-1)*b01*bmuT*p2c0+p2c1)+b01*p2c1*((-1)* \
        bmuT**2+sigmaMuT**2+sigmaT**2)))
    P_22[3,:] = 2*(P_C_22)*(2*a1+amuT*b01)**2*(2*a1**2*p2c0+(-2)*amuT*b00* \
        b01*p2c1+a1*(amuT*b01*p2c0+(-1)*b00*p2c1+3*b01*bmuT*p2c1))* \
        sigmaMuT**3
    P_22[4,:] = (-1)*(P_C_22)*b01*(2*a1+amuT*b01)**3*p2c1*sigmaMuT**3
    P_23[0,:] = (P_C_23)*sigmaMuT**3*((-1)*amuT**3*b00**4*p3c1+2*a1*amuT**2* \
        b00**2*(amuT*b00*p3c0+6*amuT*p3c1+3*b00*bmuT*p3c1)+(-12)* \
        a1**2*amuT*(amuT**2*(b00*p3c0+p3c1)+amuT*b00*bmuT*(b00*p3c0+3* \
        p3c1)+b00**2*p3c1*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2))+(-16)* \
        a1**4*bmuT*p3c0*(bmuT**2+(-3)*(sigmaMuT**2+sigmaT**2))+8*a1**3*( \
        3*amuT**2*bmuT*p3c0+3*amuT*(b00*p3c0+p3c1)*(bmuT**2+(-1)* \
        sigmaMuT**2+(-1)*sigmaT**2)+b00*bmuT*p3c1*(bmuT**2+(-3)*( \
        sigmaMuT**2+sigmaT**2))))
    P_23[1,:] = 2*(P_C_23)*sigmaMuT**3*((-2)*amuT**3*b00**3*b01*p3c1+3*a1* \
        amuT**2*b00*((-1)*b00*(b00+(-3)*b01*bmuT)*p3c1+amuT*b01*(b00* \
        p3c0+4*p3c1))+24*a1**4*p3c0*(bmuT**2+(-1)*sigmaMuT**2+(-1)* \
        sigmaT**2)+(-6)*a1**2*amuT*(amuT**2*b01*p3c0+amuT*((-1)*b00**2* \
        p3c0+2*b00*b01*bmuT*p3c0+(-3)*b00*p3c1+3*b01*bmuT*p3c1)+(-2)* \
        b00*p3c1*(b00*bmuT+b01*((-1)*bmuT**2+sigmaMuT**2+sigmaT**2)))+(-4) \
        *a1**3*(3*amuT**2*p3c0+p3c1*((-1)*b01*bmuT**3+3*b00*(bmuT**2+( \
        -1)*sigmaMuT**2+(-1)*sigmaT**2)+3*b01*bmuT*(sigmaMuT**2+sigmaT**2)) \
        +3*amuT*(2*b00*bmuT*p3c0+2*bmuT*p3c1+b01*p3c0*((-1)*bmuT**2+ \
        sigmaMuT**2+sigmaT**2))))
    P_23[2,:] = (-6)*(P_C_23)*(2*a1+amuT*b01)*sigmaMuT**3*(4*a1**3*bmuT*p3c0+ \
        amuT**2*b00**2*b01*p3c1+(-1)*a1*amuT*((-1)*b00*(b00+(-3)*b01* \
        bmuT)*p3c1+amuT*b01*(b00*p3c0+2*p3c1))+(-2)*a1**2*(b00*bmuT* \
        p3c1+amuT*(b00*p3c0+(-1)*b01*bmuT*p3c0+p3c1)+b01*p3c1*((-1)* \
        bmuT**2+sigmaMuT**2+sigmaT**2)))
    P_23[3,:] = 2*(P_C_23)*(2*a1+amuT*b01)**2*(2*a1**2*p3c0+(-2)*amuT*b00* \
        b01*p3c1+a1*(amuT*b01*p3c0+(-1)*b00*p3c1+3*b01*bmuT*p3c1))* \
        sigmaMuT**3
    P_23[4,:] = (-1)*(P_C_23)*b01*(2*a1+amuT*b01)**3*p3c1*sigmaMuT**3
    P_24[0,:] = (P_C_24)*sigmaMuT**2*(amuT**2*b00**4*p4c2+(-2)*a1*amuT*b00**2*( \
        amuT*b00*p4c1+6*amuT*p4c2+2*b00*bmuT*p4c2)+16*a1**4*p4c0*( \
        bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)+4*a1**2*(2*amuT*b00* \
        bmuT*(b00*p4c1+3*p4c2)+amuT**2*(b00**2*p4c0+3*b00*p4c1+3*p4c2)+ \
        b00**2*p4c2*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2))+(-8)*a1**3* \
        (amuT**2*p4c0+2*amuT*bmuT*(b00*p4c0+p4c1)+(b00*p4c1+p4c2)*( \
        bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)))
    P_24[1,:] = 2*(P_C_24)*sigmaMuT**2*((-16)*a1**4*bmuT*p4c0+2*amuT**2*b00**3* \
        b01*p4c2+(-1)*a1*amuT*b00*((-2)*b00*(b00+(-3)*b01*bmuT)*p4c2+ \
        3*amuT*b01*(b00*p4c1+4*p4c2))+4*a1**3*(2*b00*bmuT*p4c1+(-1)* \
        b01*bmuT**2*p4c1+2*amuT*(b00*p4c0+(-1)*b01*bmuT*p4c0+p4c1)+2* \
        bmuT*p4c2+b01*p4c1*sigmaMuT**2+b01*p4c1*sigmaT**2)+2*a1**2*( \
        amuT**2*b01*(2*b00*p4c0+3*p4c1)+amuT*((-2)*b00**2*p4c1+4*b00* \
        b01*bmuT*p4c1+(-6)*b00*p4c2+6*b01*bmuT*p4c2)+(-2)*b00*p4c2*( \
        b00*bmuT+b01*((-1)*bmuT**2+sigmaMuT**2+sigmaT**2))))
    P_24[2,:] = 2*(P_C_24)*sigmaMuT**2*(8*a1**4*p4c0+3*amuT**2*b00**2*b01**2* \
        p4c2+a1**3*(8*amuT*b01*p4c0+(-4)*(b00*p4c1+(-2)*b01*bmuT*p4c1+ \
        p4c2))+(-3)*a1*amuT*b01*((-2)*b00*(b00+(-1)*b01*bmuT)*p4c2+ \
        amuT*b01*(b00*p4c1+2*p4c2))+2*a1**2*(amuT**2*b01**2*p4c0+2* \
        amuT*b01*((-2)*b00*p4c1+b01*bmuT*p4c1+(-3)*p4c2)+p4c2*(b00**2+( \
        -4)*b00*b01*bmuT+b01**2*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)) \
        ))
    P_24[3,:] = (-2)*(P_C_24)*b01*(2*a1+amuT*b01)*(2*a1**2*p4c1+(-2)*amuT* \
        b00*b01*p4c2+a1*(amuT*b01*p4c1+(-2)*b00*p4c2+2*b01*bmuT*p4c2)) \
        *sigmaMuT**2
    P_24[4,:] = (P_C_24)*b01**2*(2*a1+amuT*b01)**2*p4c2*sigmaMuT**2
    P_25[0,:] = (P_C_25)*sigmaMuT**2*(amuT**2*b00**4*p5c2+(-2)*a1*amuT*b00**2*( \
        amuT*b00*p5c1+6*amuT*p5c2+2*b00*bmuT*p5c2)+16*a1**4*p5c0*( \
        bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)+4*a1**2*(2*amuT*b00* \
        bmuT*(b00*p5c1+3*p5c2)+amuT**2*(b00**2*p5c0+3*b00*p5c1+3*p5c2)+ \
        b00**2*p5c2*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2))+(-8)*a1**3* \
        (amuT**2*p5c0+2*amuT*bmuT*(b00*p5c0+p5c1)+(b00*p5c1+p5c2)*( \
        bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)))
    P_25[1,:] = 2*(P_C_25)*sigmaMuT**2*((-16)*a1**4*bmuT*p5c0+2*amuT**2*b00**3* \
        b01*p5c2+(-1)*a1*amuT*b00*((-2)*b00*(b00+(-3)*b01*bmuT)*p5c2+ \
        3*amuT*b01*(b00*p5c1+4*p5c2))+4*a1**3*(2*b00*bmuT*p5c1+(-1)* \
        b01*bmuT**2*p5c1+2*amuT*(b00*p5c0+(-1)*b01*bmuT*p5c0+p5c1)+2* \
        bmuT*p5c2+b01*p5c1*sigmaMuT**2+b01*p5c1*sigmaT**2)+2*a1**2*( \
        amuT**2*b01*(2*b00*p5c0+3*p5c1)+amuT*((-2)*b00**2*p5c1+4*b00* \
        b01*bmuT*p5c1+(-6)*b00*p5c2+6*b01*bmuT*p5c2)+(-2)*b00*p5c2*( \
        b00*bmuT+b01*((-1)*bmuT**2+sigmaMuT**2+sigmaT**2))))
    P_25[2,:] = 2*(P_C_25)*sigmaMuT**2*(8*a1**4*p5c0+3*amuT**2*b00**2*b01**2* \
        p5c2+a1**3*(8*amuT*b01*p5c0+(-4)*(b00*p5c1+(-2)*b01*bmuT*p5c1+ \
        p5c2))+(-3)*a1*amuT*b01*((-2)*b00*(b00+(-1)*b01*bmuT)*p5c2+ \
        amuT*b01*(b00*p5c1+2*p5c2))+2*a1**2*(amuT**2*b01**2*p5c0+2* \
        amuT*b01*((-2)*b00*p5c1+b01*bmuT*p5c1+(-3)*p5c2)+p5c2*(b00**2+( \
        -4)*b00*b01*bmuT+b01**2*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)) \
        ))
    P_25[3,:] = (-2)*(P_C_25)*b01*(2*a1+amuT*b01)*(2*a1**2*p5c1+(-2)*amuT* \
        b00*b01*p5c2+a1*(amuT*b01*p5c1+(-2)*b00*p5c2+2*b01*bmuT*p5c2)) \
        *sigmaMuT**2
    P_25[4,:] = (P_C_25)*b01**2*(2*a1+amuT*b01)**2*p5c2*sigmaMuT**2
    P_26[0,:] = (P_C_26)*sigmaMuT**2*(amuT**2*b00**4*p6c2+(-2)*a1*amuT*b00**2*( \
        amuT*b00*p6c1+6*amuT*p6c2+2*b00*bmuT*p6c2)+16*a1**4*p6c0*( \
        bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)+4*a1**2*(2*amuT*b00* \
        bmuT*(b00*p6c1+3*p6c2)+amuT**2*(b00**2*p6c0+3*b00*p6c1+3*p6c2)+ \
        b00**2*p6c2*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2))+(-8)*a1**3* \
        (amuT**2*p6c0+2*amuT*bmuT*(b00*p6c0+p6c1)+(b00*p6c1+p6c2)*( \
        bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)))
    P_26[1,:] = 2*(P_C_26)*sigmaMuT**2*((-16)*a1**4*bmuT*p6c0+2*amuT**2*b00**3* \
        b01*p6c2+(-1)*a1*amuT*b00*((-2)*b00*(b00+(-3)*b01*bmuT)*p6c2+ \
        3*amuT*b01*(b00*p6c1+4*p6c2))+4*a1**3*(2*b00*bmuT*p6c1+(-1)* \
        b01*bmuT**2*p6c1+2*amuT*(b00*p6c0+(-1)*b01*bmuT*p6c0+p6c1)+2* \
        bmuT*p6c2+b01*p6c1*sigmaMuT**2+b01*p6c1*sigmaT**2)+2*a1**2*( \
        amuT**2*b01*(2*b00*p6c0+3*p6c1)+amuT*((-2)*b00**2*p6c1+4*b00* \
        b01*bmuT*p6c1+(-6)*b00*p6c2+6*b01*bmuT*p6c2)+(-2)*b00*p6c2*( \
        b00*bmuT+b01*((-1)*bmuT**2+sigmaMuT**2+sigmaT**2))))
    P_26[2,:] = 2*(P_C_26)*sigmaMuT**2*(8*a1**4*p6c0+3*amuT**2*b00**2*b01**2* \
        p6c2+a1**3*(8*amuT*b01*p6c0+(-4)*(b00*p6c1+(-2)*b01*bmuT*p6c1+ \
        p6c2))+(-3)*a1*amuT*b01*((-2)*b00*(b00+(-1)*b01*bmuT)*p6c2+ \
        amuT*b01*(b00*p6c1+2*p6c2))+2*a1**2*(amuT**2*b01**2*p6c0+2* \
        amuT*b01*((-2)*b00*p6c1+b01*bmuT*p6c1+(-3)*p6c2)+p6c2*(b00**2+( \
        -4)*b00*b01*bmuT+b01**2*(bmuT**2+(-1)*sigmaMuT**2+(-1)*sigmaT**2)) \
        ))
    P_26[3,:] = (-2)*(P_C_26)*b01*(2*a1+amuT*b01)*(2*a1**2*p6c1+(-2)*amuT* \
        b00*b01*p6c2+a1*(amuT*b01*p6c1+(-2)*b00*p6c2+2*b01*bmuT*p6c2)) \
        *sigmaMuT**2
    P_26[4,:] = (P_C_26)*b01**2*(2*a1+amuT*b01)**2*p6c2*sigmaMuT**2
    P_27[0,:] = (P_C_27)*((-16)*a1**4*bmuT*p7c0+8*a1**3*(amuT*(b00*p7c0+p7c1)+ \
        bmuT*(b00*p7c1+p7c2))+(-1)*amuT*b00**4*p7c3+2*a1*b00**2*(amuT* \
        b00*p7c2+6*amuT*p7c3+b00*bmuT*p7c3)+(-4)*a1**2*(b00*bmuT*(b00* \
        p7c2+3*p7c3)+amuT*(b00**2*p7c1+3*b00*p7c2+3*p7c3)))*sigmaMuT
    P_27[1,:] = 2*(P_C_27)*(8*a1**4*p7c0+4*a1**3*(amuT*b01*p7c0+(-1)*b00*p7c1+ \
        b01*bmuT*p7c1+(-1)*p7c2)+(-2)*amuT*b00**3*b01*p7c3+2*a1**2*( \
        b00**2*p7c2+(-2)*b00*b01*bmuT*p7c2+(-1)*amuT*b01*(2*b00*p7c1+ \
        3*p7c2)+3*b00*p7c3+(-3)*b01*bmuT*p7c3)+a1*b00*((-1)*b00*(b00+( \
        -3)*b01*bmuT)*p7c3+3*amuT*b01*(b00*p7c2+4*p7c3)))*sigmaMuT
    P_27[2,:] = 2*(P_C_27)*b01*((-4)*a1**3*p7c1+(-2)*a1**2*(amuT*b01*p7c1+(-2) \
        *b00*p7c2+b01*bmuT*p7c2+(-3)*p7c3)+(-3)*amuT*b00**2*b01*p7c3+ \
        3*a1*(b00*((-1)*b00+b01*bmuT)*p7c3+amuT*b01*(b00*p7c2+2*p7c3)) \
        )*sigmaMuT
    P_27[3,:] = 2*(P_C_27)*b01**2*(2*a1**2*p7c2+(-2)*amuT*b00*b01*p7c3+a1*( \
        amuT*b01*p7c2+(-3)*b00*p7c3+b01*bmuT*p7c3))*sigmaMuT
    P_27[4,:] = (-1)*(P_C_27)*b01**3*(2*a1+amuT*b01)*p7c3*sigmaMuT
    P_28[0,:] = (P_C_28)*((-16)*a1**4*bmuT*p8c0+8*a1**3*(amuT*(b00*p8c0+p8c1)+ \
        bmuT*(b00*p8c1+p8c2))+(-1)*amuT*b00**4*p8c3+2*a1*b00**2*(amuT* \
        b00*p8c2+6*amuT*p8c3+b00*bmuT*p8c3)+(-4)*a1**2*(b00*bmuT*(b00* \
        p8c2+3*p8c3)+amuT*(b00**2*p8c1+3*b00*p8c2+3*p8c3)))*sigmaMuT
    P_28[1,:] = 2*(P_C_28)*(8*a1**4*p8c0+4*a1**3*(amuT*b01*p8c0+(-1)*b00*p8c1+ \
        b01*bmuT*p8c1+(-1)*p8c2)+(-2)*amuT*b00**3*b01*p8c3+2*a1**2*( \
        b00**2*p8c2+(-2)*b00*b01*bmuT*p8c2+(-1)*amuT*b01*(2*b00*p8c1+ \
        3*p8c2)+3*b00*p8c3+(-3)*b01*bmuT*p8c3)+a1*b00*((-1)*b00*(b00+( \
        -3)*b01*bmuT)*p8c3+3*amuT*b01*(b00*p8c2+4*p8c3)))*sigmaMuT
    P_28[2,:] = 2*(P_C_28)*b01*((-4)*a1**3*p8c1+(-2)*a1**2*(amuT*b01*p8c1+(-2) \
        *b00*p8c2+b01*bmuT*p8c2+(-3)*p8c3)+(-3)*amuT*b00**2*b01*p8c3+ \
        3*a1*(b00*((-1)*b00+b01*bmuT)*p8c3+amuT*b01*(b00*p8c2+2*p8c3)) \
        )*sigmaMuT
    P_28[3,:] = 2*(P_C_28)*b01**2*(2*a1**2*p8c2+(-2)*amuT*b00*b01*p8c3+a1*( \
        amuT*b01*p8c2+(-3)*b00*p8c3+b01*bmuT*p8c3))*sigmaMuT
    P_28[4,:] = (-1)*(P_C_28)*b01**3*(2*a1+amuT*b01)*p8c3*sigmaMuT
    P_29[0,:] = (P_C_29)*((-16)*a1**4*bmuT*p9c0+8*a1**3*(amuT*(b00*p9c0+p9c1)+ \
        bmuT*(b00*p9c1+p9c2))+(-1)*amuT*b00**4*p9c3+2*a1*b00**2*(amuT* \
        b00*p9c2+6*amuT*p9c3+b00*bmuT*p9c3)+(-4)*a1**2*(b00*bmuT*(b00* \
        p9c2+3*p9c3)+amuT*(b00**2*p9c1+3*b00*p9c2+3*p9c3)))*sigmaMuT
    P_29[1,:] = 2*(P_C_29)*(8*a1**4*p9c0+4*a1**3*(amuT*b01*p9c0+(-1)*b00*p9c1+ \
        b01*bmuT*p9c1+(-1)*p9c2)+(-2)*amuT*b00**3*b01*p9c3+2*a1**2*( \
        b00**2*p9c2+(-2)*b00*b01*bmuT*p9c2+(-1)*amuT*b01*(2*b00*p9c1+ \
        3*p9c2)+3*b00*p9c3+(-3)*b01*bmuT*p9c3)+a1*b00*((-1)*b00*(b00+( \
        -3)*b01*bmuT)*p9c3+3*amuT*b01*(b00*p9c2+4*p9c3)))*sigmaMuT
    P_29[2,:] = 2*(P_C_29)*b01*((-4)*a1**3*p9c1+(-2)*a1**2*(amuT*b01*p9c1+(-2) \
        *b00*p9c2+b01*bmuT*p9c2+(-3)*p9c3)+(-3)*amuT*b00**2*b01*p9c3+ \
        3*a1*(b00*((-1)*b00+b01*bmuT)*p9c3+amuT*b01*(b00*p9c2+2*p9c3)) \
        )*sigmaMuT
    P_29[3,:] = 2*(P_C_29)*b01**2*(2*a1**2*p9c2+(-2)*amuT*b00*b01*p9c3+a1*( \
        amuT*b01*p9c2+(-3)*b00*p9c3+b01*bmuT*p9c3))*sigmaMuT
    P_29[4,:] = (-1)*(P_C_29)*b01**3*(2*a1+amuT*b01)*p9c3*sigmaMuT
    P_30[0,:] = (P_C_30)*((-16)*a1**4*bmuT*p10c0+8*a1**3*(amuT*(b00*p10c0+p10c1) \
        +bmuT*(b00*p10c1+p10c2))+(-1)*amuT*b00**4*p10c3+2*a1*b00**2*( \
        amuT*b00*p10c2+6*amuT*p10c3+b00*bmuT*p10c3)+(-4)*a1**2*(b00* \
        bmuT*(b00*p10c2+3*p10c3)+amuT*(b00**2*p10c1+3*b00*p10c2+3*p10c3) \
        ))*sigmaMuT
    P_30[1,:] = 2*(P_C_30)*(8*a1**4*p10c0+4*a1**3*(amuT*b01*p10c0+(-1)*b00* \
        p10c1+b01*bmuT*p10c1+(-1)*p10c2)+(-2)*amuT*b00**3*b01*p10c3+2* \
        a1**2*(b00**2*p10c2+(-2)*b00*b01*bmuT*p10c2+(-1)*amuT*b01*(2* \
        b00*p10c1+3*p10c2)+3*b00*p10c3+(-3)*b01*bmuT*p10c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuT)*p10c3+3*amuT*b01*(b00*p10c2+4* \
        p10c3)))*sigmaMuT
    P_30[2,:] = 2*(P_C_30)*b01*((-4)*a1**3*p10c1+(-2)*a1**2*(amuT*b01*p10c1+( \
        -2)*b00*p10c2+b01*bmuT*p10c2+(-3)*p10c3)+(-3)*amuT*b00**2*b01* \
        p10c3+3*a1*(b00*((-1)*b00+b01*bmuT)*p10c3+amuT*b01*(b00*p10c2+ \
        2*p10c3)))*sigmaMuT
    P_30[3,:] = 2*(P_C_30)*b01**2*(2*a1**2*p10c2+(-2)*amuT*b00*b01*p10c3+a1*( \
        amuT*b01*p10c2+(-3)*b00*p10c3+b01*bmuT*p10c3))*sigmaMuT
    P_30[4,:] = (-1)*(P_C_30)*b01**3*(2*a1+amuT*b01)*p10c3*sigmaMuT
    P_31[0,:] = (P_C_31)*(16*a1**4*p11c0+(-8)*a1**3*(b00*p11c1+p11c2)+b00**4* \
        p11c4+4*a1**2*(b00**2*p11c2+3*b00*p11c3+3*p11c4)+(-2)*a1* \
        b00**2*(b00*p11c3+6*p11c4))
    P_31[1,:] = 2*(P_C_31)*b01*((-4)*a1**3*p11c1+a1**2*(4*b00*p11c2+6*p11c3)+ \
        2*b00**3*p11c4+(-3)*a1*b00*(b00*p11c3+4*p11c4))
    P_31[2,:] = 2*(P_C_31)*b01**2*(2*a1**2*p11c2+3*b00**2*p11c4+(-3)*a1*(b00* \
        p11c3+2*p11c4))
    P_31[3,:] = (-2)*(P_C_31)*b01**3*(a1*p11c3+(-2)*b00*p11c4)
    P_31[4,:] = (P_C_31)*b01**4*p11c4
    P_32[0,:] = (P_C_32)*(16*a1**4*p12c0+(-8)*a1**3*(b00*p12c1+p12c2)+b00**4* \
        p12c4+4*a1**2*(b00**2*p12c2+3*b00*p12c3+3*p12c4)+(-2)*a1* \
        b00**2*(b00*p12c3+6*p12c4))
    P_32[1,:] = 2*(P_C_32)*b01*((-4)*a1**3*p12c1+a1**2*(4*b00*p12c2+6*p12c3)+ \
        2*b00**3*p12c4+(-3)*a1*b00*(b00*p12c3+4*p12c4))
    P_32[2,:] = 2*(P_C_32)*b01**2*(2*a1**2*p12c2+3*b00**2*p12c4+(-3)*a1*(b00* \
        p12c3+2*p12c4))
    P_32[3,:] = (-2)*(P_C_32)*b01**3*(a1*p12c3+(-2)*b00*p12c4)
    P_32[4,:] = (P_C_32)*b01**4*p12c4
    P_33[0,:] = (P_C_33)*(16*a1**4*p13c0+(-8)*a1**3*(b00*p13c1+p13c2)+b00**4* \
        p13c4+4*a1**2*(b00**2*p13c2+3*b00*p13c3+3*p13c4)+(-2)*a1* \
        b00**2*(b00*p13c3+6*p13c4))
    P_33[1,:] = 2*(P_C_33)*b01*((-4)*a1**3*p13c1+a1**2*(4*b00*p13c2+6*p13c3)+ \
        2*b00**3*p13c4+(-3)*a1*b00*(b00*p13c3+4*p13c4))
    P_33[2,:] = 2*(P_C_33)*b01**2*(2*a1**2*p13c2+3*b00**2*p13c4+(-3)*a1*(b00* \
        p13c3+2*p13c4))
    P_33[3,:] = (-2)*(P_C_33)*b01**3*(a1*p13c3+(-2)*b00*p13c4)
    P_33[4,:] = (P_C_33)*b01**4*p13c4
    P_34[0,:] = (P_C_34)*(16*a1**4*p14c0+(-8)*a1**3*(b00*p14c1+p14c2)+b00**4* \
        p14c4+4*a1**2*(b00**2*p14c2+3*b00*p14c3+3*p14c4)+(-2)*a1* \
        b00**2*(b00*p14c3+6*p14c4))
    P_34[1,:] = 2*(P_C_34)*b01*((-4)*a1**3*p14c1+a1**2*(4*b00*p14c2+6*p14c3)+ \
        2*b00**3*p14c4+(-3)*a1*b00*(b00*p14c3+4*p14c4))
    P_34[2,:] = 2*(P_C_34)*b01**2*(2*a1**2*p14c2+3*b00**2*p14c4+(-3)*a1*(b00* \
        p14c3+2*p14c4))
    P_34[3,:] = (-2)*(P_C_34)*b01**3*(a1*p14c3+(-2)*b00*p14c4)
    P_34[4,:] = (P_C_34)*b01**4*p14c4
    P_35[0,:] = (P_C_35)*(16*a1**4*p15c0+(-8)*a1**3*(b00*p15c1+p15c2)+b00**4* \
        p15c4+4*a1**2*(b00**2*p15c2+3*b00*p15c3+3*p15c4)+(-2)*a1* \
        b00**2*(b00*p15c3+6*p15c4))
    P_35[1,:] = 2*(P_C_35)*b01*((-4)*a1**3*p15c1+a1**2*(4*b00*p15c2+6*p15c3)+ \
        2*b00**3*p15c4+(-3)*a1*b00*(b00*p15c3+4*p15c4))
    P_35[2,:] = 2*(P_C_35)*b01**2*(2*a1**2*p15c2+3*b00**2*p15c4+(-3)*a1*(b00* \
        p15c3+2*p15c4))
    P_35[3,:] = (-2)*(P_C_35)*b01**3*(a1*p15c3+(-2)*b00*p15c4)
    P_35[4,:] = (P_C_35)*b01**4*p15c4
    P_output = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13, \
        P_14, P_15, P_16, P_17, P_18, P_19, P_20, P_21, P_22, P_23, P_24, P_25, \
        P_26, P_27, P_28, P_29, P_30, P_31, P_32, P_33, P_34, P_35))
    
    #
    return P_output
