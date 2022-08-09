import numpy as np
from scipy.special import erf
from numba import vectorize


# @vectorize([int32(int32, int32),
            # int64(int64, int64),
            # float32(float32, float32),
            # float64(float64, float64)])
def PC_Coeffs_CDF_Risk_array_function(v, muY, sigmaMuY, sigmaY, amuZ, bmuZ,  \
        sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT, amuV, bmuV, sigmaMuV,  \
        sigmaV):
    
    ### Notations (for simplicity):
    
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
    
    
    
    ##
    ###########################################################################
    ########################## FIRST INTEGRATION #############################
    ###########################################################################
    
    
    # Total Constant;
    #;
    a1 = (-1/2)*(sigmaMuY**2+sigmaY**2)**(-1)*(sigmaMuZ**2+sigmaZ**2)**(-1)*( \
        sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)
    b00 = muY*(sigmaMuY**2+sigmaY**2)**(-1)+(-1)*amuZ*bmuZ*(sigmaMuZ**2+ \
        sigmaZ**2)**(-1)
    b01 = amuZ*(sigmaMuZ**2+sigmaZ**2)**(-1)
    n_sites  =  muY.shape[1]
    P_1 =  np.zeros((1, n_sites))
    P_2 =  np.zeros((2, n_sites))
    P_3 =  np.zeros((2, n_sites))
    P_4 =  np.zeros((3, n_sites))
    P_5 =  np.zeros((3, n_sites))
    P_6 =  np.zeros((3, n_sites))
    P_7 =  np.zeros((4, n_sites))
    P_8 =  np.zeros((4, n_sites))
    P_9 =  np.zeros((4, n_sites))
    P_10 =  np.zeros((4, n_sites))
    P_11 =  np.zeros((5, n_sites))
    P_12 =  np.zeros((5, n_sites))
    P_13 =  np.zeros((5, n_sites))
    P_14 =  np.zeros((5, n_sites))
    P_15 =  np.zeros((5, n_sites))
    #;
    ## Polynomial Constants;
    #;
    constantTerm1 = (1/2)*((-1)*a1)**(-1/2)*np.pi**(-1/2)*(1+sigmaMuY**2*sigmaY**(-2))**( \
        -1/2)*sigmaY**(-1)*(1+sigmaMuZ**2*sigmaZ**(-2))**(-1/2)*sigmaZ**(-1) \
        
    P_C_1 = constantTerm1
    P_C_2 = (-1/2)*a1**(-1)*constantTerm1*(sigmaMuZ**2+sigmaZ**2)**(-1)
    P_C_3 = (-1)*constantTerm1*(2*a1*sigmaMuY**2+2*a1*sigmaY**2)**(-1)
    P_C_4 = (1/8)*a1**(-2)*constantTerm1*(sigmaMuZ**2+sigmaZ**2)**(-2)
    P_C_5 = (1/4)*a1**(-2)*constantTerm1*(sigmaMuY**2+sigmaY**2)**(-1)*( \
        sigmaMuZ**2+sigmaZ**2)**(-1)
    P_C_6 = (1/8)*a1**(-2)*constantTerm1*(sigmaMuY**2+sigmaY**2)**(-2)
    P_C_7 = (-1/48)*a1**(-3)*constantTerm1*(sigmaMuZ**2+sigmaZ**2)**(-3)
    P_C_8 = (-1/16)*a1**(-3)*constantTerm1*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)* \
        sigmaY*(sigmaMuY**2+sigmaY**2)**(-2)*(1+sigmaMuZ**2*sigmaZ**(-2))**( \
        1/2)*sigmaZ*(sigmaMuZ**2+sigmaZ**2)**(-3)
    P_C_9 = (-1/16)*a1**(-3)*constantTerm1*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)* \
        sigmaY*(sigmaMuY**2+sigmaY**2)**(-3)*(1+sigmaMuZ**2*sigmaZ**(-2))**( \
        1/2)*sigmaZ*(sigmaMuZ**2+sigmaZ**2)**(-2)
    P_C_10 = (-1/48)*a1**(-3)*constantTerm1*(sigmaMuY**2+sigmaY**2)**(-3)
    P_C_11 = (1/384)*a1**(-4)*constantTerm1*(sigmaMuZ**2+sigmaZ**2)**(-4)
    P_C_12 = (1/96)*a1**(-4)*constantTerm1*(sigmaMuY**2+sigmaY**2)**(-1)*( \
        sigmaMuZ**2+sigmaZ**2)**(-3)
    P_C_13 = (1/64)*a1**(-4)*constantTerm1*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)* \
        sigmaY*(sigmaMuY**2+sigmaY**2)**(-3)*(1+sigmaMuZ**2*sigmaZ**(-2))**( \
        1/2)*sigmaZ*(sigmaMuZ**2+sigmaZ**2)**(-3)
    P_C_14 = (1/96)*a1**(-4)*constantTerm1*(sigmaMuY**2+sigmaY**2)**(-3)*( \
        sigmaMuZ**2+sigmaZ**2)**(-1)
    P_C_15 = (1/384)*a1**(-4)*constantTerm1*(sigmaMuY**2+sigmaY**2)**(-4)
    #;
    ## Polynomial Coefficients;
    #;
    P_1[0,:] = (P_C_1)
    P_2[0,:] = (P_C_2)*((-1)*amuZ*b00*sigmaMuZ+2*a1*bmuZ*sigmaMuZ)
    P_2[1,:] = (-1)*(P_C_2)*(2*a1+amuZ*b01)*sigmaMuZ
    P_3[0,:] = (P_C_3)*(b00+2*a1*muY)*sigmaMuY
    P_3[1,:] = (P_C_3)*b01*sigmaMuY
    P_4[0,:] = (P_C_4)*sigmaMuZ**2*(amuZ**2*b00**2+(-2)*a1*amuZ*(amuZ+2*b00* \
        bmuZ)+4*a1**2*(bmuZ**2+(-1)*sigmaMuZ**2+(-1)*sigmaZ**2))
    P_4[1,:] = (-2)*(P_C_4)*(2*a1+amuZ*b01)*((-1)*amuZ*b00+2*a1*bmuZ)* \
        sigmaMuZ**2
    P_4[2,:] = (P_C_4)*(2*a1+amuZ*b01)**2*sigmaMuZ**2
    P_5[0,:] = (P_C_5)*((-1)*amuZ*b00**2+4*a1**2*bmuZ*muY+2*a1*(amuZ+b00*bmuZ+ \
        (-1)*amuZ*b00*muY))*sigmaMuY*sigmaMuZ
    P_5[1,:] = (-2)*(P_C_5)*(amuZ*b00*b01+2*a1**2*muY+a1*(b00+(-1)*b01*bmuZ+ \
        amuZ*b01*muY))*sigmaMuY*sigmaMuZ
    P_5[2,:] = (-1)*(P_C_5)*b01*(2*a1+amuZ*b01)*sigmaMuY*sigmaMuZ
    P_6[0,:] = (P_C_6)*sigmaMuY**2*((-2)*a1+b00**2+4*a1*b00*muY+(-4)*a1**2*(( \
        -1)*muY**2+sigmaMuY**2+sigmaY**2))
    P_6[1,:] = 2*(P_C_6)*b01*(b00+2*a1*muY)*sigmaMuY**2
    P_6[2,:] = (P_C_6)*b01**2*sigmaMuY**2
    P_7[0,:] = (P_C_7)*((-1)*amuZ*b00+2*a1*bmuZ)*sigmaMuZ**3*(amuZ**2*b00**2+( \
        -2)*a1*amuZ*(3*amuZ+2*b00*bmuZ)+4*a1**2*(bmuZ**2+(-3)*( \
        sigmaMuZ**2+sigmaZ**2)))
    P_7[1,:] = (-3)*(P_C_7)*(2*a1+amuZ*b01)*sigmaMuZ**3*(amuZ**2*b00**2+(-2)* \
        a1*amuZ*(amuZ+2*b00*bmuZ)+4*a1**2*(bmuZ**2+(-1)*sigmaMuZ**2+(-1) \
        *sigmaZ**2))
    P_7[2,:] = 3*(P_C_7)*(2*a1+amuZ*b01)**2*((-1)*amuZ*b00+2*a1*bmuZ)* \
        sigmaMuZ**3
    P_7[3,:] = (-1)*(P_C_7)*(2*a1+amuZ*b01)**3*sigmaMuZ**3
    P_8[0,:] = (-1)*(P_C_8)*sigmaMuY*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**( \
        1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ*((-1)* \
        amuZ**2*b00**3+2*a1*amuZ*b00*(2*b00*bmuZ+amuZ*(3+(-1)*b00*muY) \
        )+8*a1**3*muY*((-1)*bmuZ**2+sigmaMuZ**2+sigmaZ**2)+4*a1**2*( \
        amuZ**2*muY+2*amuZ*bmuZ*((-1)+b00*muY)+b00*((-1)*bmuZ**2+ \
        sigmaMuZ**2+sigmaZ**2)))
    P_8[1,:] = (-1)*(P_C_8)*sigmaMuY*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**( \
        1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ*((-3)* \
        amuZ**2*b00**2*b01+16*a1**3*bmuZ*muY+2*a1*amuZ*((-2)*b00**2+3* \
        amuZ*b01+4*b00*b01*bmuZ+(-2)*amuZ*b00*b01*muY)+a1**2*(8*b00* \
        bmuZ+amuZ*(8+(-8)*b00*muY+8*b01*bmuZ*muY)+4*b01*((-1)*bmuZ**2+ \
        sigmaMuZ**2+sigmaZ**2)))
    P_8[2,:] = (P_C_8)*(2*a1+amuZ*b01)*(3*amuZ*b00*b01+4*a1**2*muY+2*a1*( \
        b00+(-2)*b01*bmuZ+amuZ*b01*muY))*sigmaMuY*sigmaMuZ**2*(1+ \
        sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2)) \
        **(1/2)*sigmaZ
    P_8[3,:] = (P_C_8)*b01*(2*a1+amuZ*b01)**2*sigmaMuY*sigmaMuZ**2*(1+ \
        sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2)) \
        **(1/2)*sigmaZ
    P_9[0,:] = (P_C_9)*sigmaMuY**2*sigmaMuZ*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)* \
        sigmaY*((-1)*amuZ*b00**3+2*a1*b00*(b00*bmuZ+amuZ*(3+(-2)*b00* \
        muY))+8*a1**3*bmuZ*(muY**2+(-1)*sigmaMuY**2+(-1)*sigmaY**2)+4* \
        a1**2*(bmuZ*((-1)+2*b00*muY)+amuZ*(2*muY+(-1)*b00*muY**2+b00*( \
        sigmaMuY**2+sigmaY**2))))*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ \
        
    P_9[1,:] = (P_C_9)*sigmaMuY**2*sigmaMuZ*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)* \
        sigmaY*((-2)*a1*b00**2+6*a1*amuZ*b01+(-3)*amuZ*b00**2*b01+4* \
        a1*b00*b01*(bmuZ+(-2)*amuZ*muY)+8*a1**3*((-1)*muY**2+ \
        sigmaMuY**2+sigmaY**2)+a1**2*(4+(-8)*b00*muY+8*b01*bmuZ*muY+4* \
        amuZ*b01*((-1)*muY**2+sigmaMuY**2+sigmaY**2)))*(1+sigmaMuZ**2* \
        sigmaZ**(-2))**(1/2)*sigmaZ
    P_9[2,:] = (-1)*(P_C_9)*b01*(3*amuZ*b00*b01+8*a1**2*muY+a1*(4*b00+(-2)* \
        b01*bmuZ+4*amuZ*b01*muY))*sigmaMuY**2*sigmaMuZ*(1+sigmaMuY**2* \
        sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)* \
        sigmaZ
    P_9[3,:] = (-1)*(P_C_9)*b01**2*(2*a1+amuZ*b01)*sigmaMuY**2*sigmaMuZ*(1+ \
        sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2)) \
        **(1/2)*sigmaZ
    P_10[0,:] = (P_C_10)*(b00+2*a1*muY)*sigmaMuY**3*(b00**2+a1*((-6)+4*b00*muY)+ \
        4*a1**2*(muY**2+(-3)*(sigmaMuY**2+sigmaY**2)))
    P_10[1,:] = (-3)*(P_C_10)*b01*sigmaMuY**3*((-1)*b00**2+a1*(2+(-4)*b00*muY)+ \
        4*a1**2*((-1)*muY**2+sigmaMuY**2+sigmaY**2))
    P_10[2,:] = 3*(P_C_10)*b01**2*(b00+2*a1*muY)*sigmaMuY**3
    P_10[3,:] = (P_C_10)*b01**3*sigmaMuY**3
    P_11[0,:] = (P_C_11)*sigmaMuZ**4*(amuZ**4*b00**4+(-4)*a1*amuZ**3*b00**2*(3* \
        amuZ+2*b00*bmuZ)+12*a1**2*amuZ**2*(amuZ**2+4*amuZ*b00*bmuZ+2* \
        b00**2*(bmuZ**2+(-1)*sigmaMuZ**2+(-1)*sigmaZ**2))+16*a1**4*( \
        bmuZ**4+(-6)*bmuZ**2*(sigmaMuZ**2+sigmaZ**2)+3*(sigmaMuZ**2+ \
        sigmaZ**2)**2)+(-16)*a1**3*amuZ*(3*amuZ*(bmuZ**2+(-1)*sigmaMuZ**2+ \
        (-1)*sigmaZ**2)+2*b00*bmuZ*(bmuZ**2+(-3)*(sigmaMuZ**2+sigmaZ**2)))) \
        
    P_11[1,:] = (-4)*(P_C_11)*(2*a1+amuZ*b01)*((-1)*amuZ*b00+2*a1*bmuZ)* \
        sigmaMuZ**4*(amuZ**2*b00**2+(-2)*a1*amuZ*(3*amuZ+2*b00*bmuZ)+4* \
        a1**2*(bmuZ**2+(-3)*(sigmaMuZ**2+sigmaZ**2)))
    P_11[2,:] = 6*(P_C_11)*(2*a1+amuZ*b01)**2*sigmaMuZ**4*(amuZ**2*b00**2+(-2)* \
        a1*amuZ*(amuZ+2*b00*bmuZ)+4*a1**2*(bmuZ**2+(-1)*sigmaMuZ**2+(-1) \
        *sigmaZ**2))
    P_11[3,:] = (-4)*(P_C_11)*(2*a1+amuZ*b01)**3*((-1)*amuZ*b00+2*a1*bmuZ)* \
        sigmaMuZ**4
    P_11[4,:] = (P_C_11)*(2*a1+amuZ*b01)**4*sigmaMuZ**4
    P_12[0,:] = (P_C_12)*sigmaMuY*sigmaMuZ**3*((-1)*amuZ**3*b00**4+2*a1*amuZ**2* \
        b00**2*(3*b00*bmuZ+amuZ*(6+(-1)*b00*muY))+16*a1**4*bmuZ*muY*( \
        bmuZ**2+(-3)*(sigmaMuZ**2+sigmaZ**2))+(-8)*a1**3*((-1)*b00*bmuZ**3+ \
        3*amuZ**2*bmuZ*muY+3*amuZ*((-1)+b00*muY)*(bmuZ**2+(-1)* \
        sigmaMuZ**2+(-1)*sigmaZ**2)+3*b00*bmuZ*(sigmaMuZ**2+sigmaZ**2))+12* \
        a1**2*amuZ*(amuZ*b00*bmuZ*((-3)+b00*muY)+amuZ**2*((-1)+b00*muY)+ \
        b00**2*((-1)*bmuZ**2+sigmaMuZ**2+sigmaZ**2)))
    P_12[1,:] = 2*(P_C_12)*sigmaMuY*sigmaMuZ**3*((-2)*amuZ**3*b00**3*b01+(-3)* \
        a1*amuZ**2*b00*(b00**2+(-4)*amuZ*b01+b00*b01*((-3)*bmuZ+amuZ* \
        muY))+(-24)*a1**4*muY*(bmuZ**2+(-1)*sigmaMuZ**2+(-1)*sigmaZ**2)+4* \
        a1**3*(3*amuZ**2*muY+3*b00*((-1)*bmuZ**2+sigmaMuZ**2+sigmaZ**2)+ \
        b01*bmuZ*(bmuZ**2+(-3)*(sigmaMuZ**2+sigmaZ**2))+amuZ*((-3)*b01* \
        bmuZ**2*muY+6*bmuZ*((-1)+b00*muY)+3*b01*muY*(sigmaMuZ**2+ \
        sigmaZ**2)))+6*a1**2*amuZ*(amuZ**2*b01*muY+amuZ*((-3)*b01*bmuZ+( \
        -1)*b00**2*muY+b00*(3+2*b01*bmuZ*muY))+2*b00*(b00*bmuZ+b01*(( \
        -1)*bmuZ**2+sigmaMuZ**2+sigmaZ**2))))
    P_12[2,:] = 6*(P_C_12)*(2*a1+amuZ*b01)*sigmaMuY*sigmaMuZ**3*((-1)*amuZ**2* \
        b00**2*b01+4*a1**3*bmuZ*muY+(-1)*a1*amuZ*(b00**2+(-2)*amuZ*b01+ \
        b00*b01*((-3)*bmuZ+amuZ*muY))+2*a1**2*(b00*bmuZ+amuZ*(1+(-1)* \
        b00*muY+b01*bmuZ*muY)+b01*((-1)*bmuZ**2+sigmaMuZ**2+sigmaZ**2)))
    P_12[3,:] = (-2)*(P_C_12)*(2*a1+amuZ*b01)**2*(2*amuZ*b00*b01+2*a1**2*muY+ \
        a1*(b00+(-3)*b01*bmuZ+amuZ*b01*muY))*sigmaMuY*sigmaMuZ**3
    P_12[4,:] = (-1)*(P_C_12)*b01*(2*a1+amuZ*b01)**3*sigmaMuY*sigmaMuZ**3
    P_13[0,:] = (P_C_13)*sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2) \
        *sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ*(amuZ**2* \
        b00**4+4*a1*amuZ*b00**2*((-1)*b00*bmuZ+amuZ*((-3)+b00*muY))+16* \
        a1**4*(muY**2+(-1)*sigmaMuY**2+(-1)*sigmaY**2)*(bmuZ**2+(-1)* \
        sigmaMuZ**2+(-1)*sigmaZ**2)+4*a1**2*(2*amuZ*b00*bmuZ*(3+(-2)* \
        b00*muY)+amuZ**2*(3+(-6)*b00*muY+b00**2*(muY**2+(-1)*sigmaMuY**2+( \
        -1)*sigmaY**2))+b00**2*(bmuZ**2+(-1)*sigmaMuZ**2+(-1)*sigmaZ**2))+ \
        8*a1**3*(bmuZ**2*((-1)+2*b00*muY)+amuZ**2*((-1)*muY**2+ \
        sigmaMuY**2+sigmaY**2)+2*amuZ*bmuZ*(2*muY+(-1)*b00*muY**2+b00*( \
        sigmaMuY**2+sigmaY**2))+(-1)*((-1)+2*b00*muY)*(sigmaMuZ**2+ \
        sigmaZ**2)))
    P_13[1,:] = 4*(P_C_13)*sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**( \
        1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ*(amuZ**2* \
        b00**3*b01+a1*amuZ*b00*(b00**2+(-6)*amuZ*b01+(-3)*b00*b01*( \
        bmuZ+(-1)*amuZ*muY))+8*a1**4*bmuZ*((-1)*muY**2+sigmaMuY**2+ \
        sigmaY**2)+4*a1**3*(bmuZ+(-2)*amuZ*muY+(-2)*b00*bmuZ*muY+b01* \
        bmuZ**2*muY+amuZ*b00*muY**2+(-1)*amuZ*b00*(sigmaMuY**2+sigmaY**2)+ \
        amuZ*b01*bmuZ*((-1)*muY**2+sigmaMuY**2+sigmaY**2)+(-1)*b01*muY*( \
        sigmaMuZ**2+sigmaZ**2))+2*a1**2*(amuZ*(3*b01*bmuZ+2*b00**2*muY+( \
        -1)*b00*(3+4*b01*bmuZ*muY))+amuZ**2*b01*((-3)*muY+b00*muY**2+( \
        -1)*b00*(sigmaMuY**2+sigmaY**2))+(-1)*b00*(b00*bmuZ+b01*((-1)* \
        bmuZ**2+sigmaMuZ**2+sigmaZ**2))))
    P_13[2,:] = 2*(P_C_13)*sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**( \
        1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ*(3* \
        amuZ**2*b00**2*b01**2+6*a1*amuZ*b01*(b00**2+(-1)*amuZ*b01+b00* \
        b01*((-1)*bmuZ+amuZ*muY))+8*a1**4*(muY**2+(-1)*sigmaMuY**2+(-1)* \
        sigmaY**2)+4*a1**3*((-1)+2*b00*muY+(-2)*b01*(2*bmuZ*muY+amuZ*(( \
        -1)*muY**2+sigmaMuY**2+sigmaY**2)))+2*a1**2*(b00**2+(-4)*b00*b01*( \
        bmuZ+(-2)*amuZ*muY)+b01*((-2)*amuZ*(3+2*b01*bmuZ*muY)+amuZ**2* \
        b01*(muY**2+(-1)*sigmaMuY**2+(-1)*sigmaY**2)+b01*(bmuZ**2+(-1)* \
        sigmaMuZ**2+(-1)*sigmaZ**2))))
    P_13[3,:] = 4*(P_C_13)*b01*(2*a1+amuZ*b01)*(amuZ*b00*b01+2*a1**2*muY+a1*( \
        b00+(-1)*b01*bmuZ+amuZ*b01*muY))*sigmaMuY**2*sigmaMuZ**2*(1+ \
        sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2)) \
        **(1/2)*sigmaZ
    P_13[4,:] = (P_C_13)*b01**2*(2*a1+amuZ*b01)**2*sigmaMuY**2*sigmaMuZ**2*(1+ \
        sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2)) \
        **(1/2)*sigmaZ
    P_14[0,:] = (P_C_14)*sigmaMuY**3*sigmaMuZ*((-1)*amuZ*b00**4+2*a1*b00**2*( \
        b00*bmuZ+amuZ*(6+(-3)*b00*muY))+(-12)*a1**2*(amuZ+(-3)*amuZ* \
        b00*muY+b00*bmuZ*(1+(-1)*b00*muY)+amuZ*b00**2*(muY**2+(-1)* \
        sigmaMuY**2+(-1)*sigmaY**2))+16*a1**4*bmuZ*muY*(muY**2+(-3)*( \
        sigmaMuY**2+sigmaY**2))+8*a1**3*(3*bmuZ*((-1)*muY+b00*muY**2+(-1) \
        *b00*(sigmaMuY**2+sigmaY**2))+amuZ*(3*muY**2+(-1)*b00*muY**3+(-3) \
        *(sigmaMuY**2+sigmaY**2)+3*b00*muY*(sigmaMuY**2+sigmaY**2))))
    P_14[1,:] = 2*(P_C_14)*sigmaMuY**3*sigmaMuZ*((-2)*amuZ*b00**3*b01+(-1)*a1* \
        b00*(b00**2+(-12)*amuZ*b01+(-3)*b00*b01*(bmuZ+(-3)*amuZ*muY))+( \
        -8)*a1**4*(muY**3+(-3)*muY*(sigmaMuY**2+sigmaY**2))+(-4)*a1**3*( \
        3*(b00+(-1)*b01*bmuZ)*muY**2+amuZ*b01*muY**3+(-3)*(b00+(-1)* \
        b01*bmuZ)*(sigmaMuY**2+sigmaY**2)+(-3)*muY*(1+amuZ*b01*( \
        sigmaMuY**2+sigmaY**2)))+(-6)*a1**2*(b00**2*muY+b01*(bmuZ+(-3)* \
        amuZ*muY)+(-1)*b00*(1+2*b01*(bmuZ*muY+amuZ*((-1)*muY**2+ \
        sigmaMuY**2+sigmaY**2)))))
    P_14[2,:] = 6*(P_C_14)*b01*sigmaMuY**3*sigmaMuZ*((-1)*amuZ*b00**2*b01+a1*(( \
        -1)*b00**2+2*amuZ*b01+b00*b01*(bmuZ+(-3)*amuZ*muY))+4*a1**3*(( \
        -1)*muY**2+sigmaMuY**2+sigmaY**2)+2*a1**2*(1+(-2)*b00*muY+b01* \
        bmuZ*muY+amuZ*b01*((-1)*muY**2+sigmaMuY**2+sigmaY**2)))
    P_14[3,:] = (-2)*(P_C_14)*b01**2*(2*amuZ*b00*b01+6*a1**2*muY+a1*(3*b00+( \
        -1)*b01*bmuZ+3*amuZ*b01*muY))*sigmaMuY**3*sigmaMuZ
    P_14[4,:] = (-1)*(P_C_14)*b01**3*(2*a1+amuZ*b01)*sigmaMuY**3*sigmaMuZ
    P_15[0,:] = (P_C_15)*sigmaMuY**4*(b00**4+4*a1*b00**2*((-3)+2*b00*muY)+12* \
        a1**2*(1+(-4)*b00*muY+2*b00**2*(muY**2+(-1)*sigmaMuY**2+(-1)* \
        sigmaY**2))+16*a1**3*((-3)*muY**2+2*b00*muY**3+3*(sigmaMuY**2+ \
        sigmaY**2)+(-6)*b00*muY*(sigmaMuY**2+sigmaY**2))+16*a1**4*(muY**4+( \
        -6)*muY**2*(sigmaMuY**2+sigmaY**2)+3*(sigmaMuY**2+sigmaY**2)**2))
    P_15[1,:] = 4*(P_C_15)*b01*(b00+2*a1*muY)*sigmaMuY**4*(b00**2+a1*((-6)+4* \
        b00*muY)+4*a1**2*(muY**2+(-3)*(sigmaMuY**2+sigmaY**2)))
    P_15[2,:] = 6*(P_C_15)*b01**2*sigmaMuY**4*(b00**2+a1*((-2)+4*b00*muY)+4* \
        a1**2*(muY**2+(-1)*sigmaMuY**2+(-1)*sigmaY**2))
    P_15[3,:] = 4*(P_C_15)*b01**3*(b00+2*a1*muY)*sigmaMuY**4
    P_15[4,:] = (P_C_15)*b01**4*sigmaMuY**4
    P_output_1 = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13, \
        P_14, P_15))
    
    
    
    ##
    ###########################################################################
    ########################## SECOND INTEGRATION #############################
    ###########################################################################
    
    
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
    n_sites  =  muY.shape[1]
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
    P_output_2 = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13, \
        P_14, P_15, P_16, P_17, P_18, P_19, P_20, P_21, P_22, P_23, P_24, P_25, \
        P_26, P_27, P_28, P_29, P_30, P_31, P_32, P_33, P_34, P_35))
    
    
    ##
    ###########################################################################
    ########################## THIRD INTEGRATION #############################
    ###########################################################################
    
    p1c0 = P_output_2[0, :]
    p2c0 = P_output_2[1, :]
    p2c1 = P_output_2[2, :]
    p3c0 = P_output_2[3, :]
    p3c1 = P_output_2[4, :]
    p4c0 = P_output_2[5, :]
    p4c1 = P_output_2[6, :]
    p5c0 = P_output_2[7, :]
    p5c1 = P_output_2[8, :]
    p5c2 = P_output_2[9, :]
    p6c0 = P_output_2[10, :]
    p6c1 = P_output_2[11, :]
    p6c2 = P_output_2[12, :]
    p7c0 = P_output_2[13, :]
    p7c1 = P_output_2[14, :]
    p7c2 = P_output_2[15, :]
    p8c0 = P_output_2[16, :]
    p8c1 = P_output_2[17, :]
    p8c2 = P_output_2[18, :]
    p9c0 = P_output_2[19, :]
    p9c1 = P_output_2[20, :]
    p9c2 = P_output_2[21, :]
    p10c0 = P_output_2[22, :]
    p10c1 = P_output_2[23, :]
    p10c2 = P_output_2[24, :]
    p11c0 = P_output_2[25, :]
    p11c1 = P_output_2[26, :]
    p11c2 = P_output_2[27, :]
    p11c3 = P_output_2[28, :]
    p12c0 = P_output_2[29, :]
    p12c1 = P_output_2[30, :]
    p12c2 = P_output_2[31, :]
    p12c3 = P_output_2[32, :]
    p13c0 = P_output_2[33, :]
    p13c1 = P_output_2[34, :]
    p13c2 = P_output_2[35, :]
    p13c3 = P_output_2[36, :]
    p14c0 = P_output_2[37, :]
    p14c1 = P_output_2[38, :]
    p14c2 = P_output_2[39, :]
    p14c3 = P_output_2[40, :]
    p15c0 = P_output_2[41, :]
    p15c1 = P_output_2[42, :]
    p15c2 = P_output_2[43, :]
    p15c3 = P_output_2[44, :]
    p16c0 = P_output_2[45, :]
    p16c1 = P_output_2[46, :]
    p16c2 = P_output_2[47, :]
    p16c3 = P_output_2[48, :]
    p17c0 = P_output_2[49, :]
    p17c1 = P_output_2[50, :]
    p17c2 = P_output_2[51, :]
    p17c3 = P_output_2[52, :]
    p18c0 = P_output_2[53, :]
    p18c1 = P_output_2[54, :]
    p18c2 = P_output_2[55, :]
    p18c3 = P_output_2[56, :]
    p19c0 = P_output_2[57, :]
    p19c1 = P_output_2[58, :]
    p19c2 = P_output_2[59, :]
    p19c3 = P_output_2[60, :]
    p20c0 = P_output_2[61, :]
    p20c1 = P_output_2[62, :]
    p20c2 = P_output_2[63, :]
    p20c3 = P_output_2[64, :]
    p21c0 = P_output_2[65, :]
    p21c1 = P_output_2[66, :]
    p21c2 = P_output_2[67, :]
    p21c3 = P_output_2[68, :]
    p21c4 = P_output_2[69, :]
    p22c0 = P_output_2[70, :]
    p22c1 = P_output_2[71, :]
    p22c2 = P_output_2[72, :]
    p22c3 = P_output_2[73, :]
    p22c4 = P_output_2[74, :]
    p23c0 = P_output_2[75, :]
    p23c1 = P_output_2[76, :]
    p23c2 = P_output_2[77, :]
    p23c3 = P_output_2[78, :]
    p23c4 = P_output_2[79, :]
    p24c0 = P_output_2[80, :]
    p24c1 = P_output_2[81, :]
    p24c2 = P_output_2[82, :]
    p24c3 = P_output_2[83, :]
    p24c4 = P_output_2[84, :]
    p25c0 = P_output_2[85, :]
    p25c1 = P_output_2[86, :]
    p25c2 = P_output_2[87, :]
    p25c3 = P_output_2[88, :]
    p25c4 = P_output_2[89, :]
    p26c0 = P_output_2[90, :]
    p26c1 = P_output_2[91, :]
    p26c2 = P_output_2[92, :]
    p26c3 = P_output_2[93, :]
    p26c4 = P_output_2[94, :]
    p27c0 = P_output_2[95, :]
    p27c1 = P_output_2[96, :]
    p27c2 = P_output_2[97, :]
    p27c3 = P_output_2[98, :]
    p27c4 = P_output_2[99, :]
    p28c0 = P_output_2[100, :]
    p28c1 = P_output_2[101, :]
    p28c2 = P_output_2[102, :]
    p28c3 = P_output_2[103, :]
    p28c4 = P_output_2[104, :]
    p29c0 = P_output_2[105, :]
    p29c1 = P_output_2[106, :]
    p29c2 = P_output_2[107, :]
    p29c3 = P_output_2[108, :]
    p29c4 = P_output_2[109, :]
    p30c0 = P_output_2[110, :]
    p30c1 = P_output_2[111, :]
    p30c2 = P_output_2[112, :]
    p30c3 = P_output_2[113, :]
    p30c4 = P_output_2[114, :]
    p31c0 = P_output_2[115, :]
    p31c1 = P_output_2[116, :]
    p31c2 = P_output_2[117, :]
    p31c3 = P_output_2[118, :]
    p31c4 = P_output_2[119, :]
    p32c0 = P_output_2[120, :]
    p32c1 = P_output_2[121, :]
    p32c2 = P_output_2[122, :]
    p32c3 = P_output_2[123, :]
    p32c4 = P_output_2[124, :]
    p33c0 = P_output_2[125, :]
    p33c1 = P_output_2[126, :]
    p33c2 = P_output_2[127, :]
    p33c3 = P_output_2[128, :]
    p33c4 = P_output_2[129, :]
    p34c0 = P_output_2[130, :]
    p34c1 = P_output_2[131, :]
    p34c2 = P_output_2[132, :]
    p34c3 = P_output_2[133, :]
    p34c4 = P_output_2[134, :]
    p35c0 = P_output_2[135, :]
    p35c1 = P_output_2[136, :]
    p35c2 = P_output_2[137, :]
    p35c3 = P_output_2[138, :]
    p35c4 = P_output_2[139, :]
    ##;
    ## Total Constant;
    #;
    a1 = (1/4)*((-2)*amuV**2*(sigmaMuV**2+sigmaV**2)**(-1)+(-2)*(sigmaMuT**2+ \
        sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+ \
        sigmaZ**2))**(-1))
    b00 = (-1)*amuV*bmuV*(sigmaMuV**2+sigmaV**2)**(-1)+(bmuT+amuT*(bmuZ+amuZ* \
        muY))*(sigmaMuT**2+sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*( \
        sigmaMuY**2+sigmaY**2)+sigmaZ**2))**(-1)
    b01 = amuV*(sigmaMuV**2+sigmaV**2)**(-1)
    n_sites  =  muY.shape[1]
    P_1 =  np.zeros((1, n_sites))
    P_2 =  np.zeros((2, n_sites))
    P_3 =  np.zeros((2, n_sites))
    P_4 =  np.zeros((2, n_sites))
    P_5 =  np.zeros((2, n_sites))
    P_6 =  np.zeros((3, n_sites))
    P_7 =  np.zeros((3, n_sites))
    P_8 =  np.zeros((3, n_sites))
    P_9 =  np.zeros((3, n_sites))
    P_10 =  np.zeros((3, n_sites))
    P_11 =  np.zeros((3, n_sites))
    P_12 =  np.zeros((3, n_sites))
    P_13 =  np.zeros((3, n_sites))
    P_14 =  np.zeros((3, n_sites))
    P_15 =  np.zeros((3, n_sites))
    P_16 =  np.zeros((4, n_sites))
    P_17 =  np.zeros((4, n_sites))
    P_18 =  np.zeros((4, n_sites))
    P_19 =  np.zeros((4, n_sites))
    P_20 =  np.zeros((4, n_sites))
    P_21 =  np.zeros((4, n_sites))
    P_22 =  np.zeros((4, n_sites))
    P_23 =  np.zeros((4, n_sites))
    P_24 =  np.zeros((4, n_sites))
    P_25 =  np.zeros((4, n_sites))
    P_26 =  np.zeros((4, n_sites))
    P_27 =  np.zeros((4, n_sites))
    P_28 =  np.zeros((4, n_sites))
    P_29 =  np.zeros((4, n_sites))
    P_30 =  np.zeros((4, n_sites))
    P_31 =  np.zeros((4, n_sites))
    P_32 =  np.zeros((4, n_sites))
    P_33 =  np.zeros((4, n_sites))
    P_34 =  np.zeros((4, n_sites))
    P_35 =  np.zeros((4, n_sites))
    P_36 =  np.zeros((5, n_sites))
    P_37 =  np.zeros((5, n_sites))
    P_38 =  np.zeros((5, n_sites))
    P_39 =  np.zeros((5, n_sites))
    P_40 =  np.zeros((5, n_sites))
    P_41 =  np.zeros((5, n_sites))
    P_42 =  np.zeros((5, n_sites))
    P_43 =  np.zeros((5, n_sites))
    P_44 =  np.zeros((5, n_sites))
    P_45 =  np.zeros((5, n_sites))
    P_46 =  np.zeros((5, n_sites))
    P_47 =  np.zeros((5, n_sites))
    P_48 =  np.zeros((5, n_sites))
    P_49 =  np.zeros((5, n_sites))
    P_50 =  np.zeros((5, n_sites))
    P_51 =  np.zeros((5, n_sites))
    P_52 =  np.zeros((5, n_sites))
    P_53 =  np.zeros((5, n_sites))
    P_54 =  np.zeros((5, n_sites))
    P_55 =  np.zeros((5, n_sites))
    P_56 =  np.zeros((5, n_sites))
    P_57 =  np.zeros((5, n_sites))
    P_58 =  np.zeros((5, n_sites))
    P_59 =  np.zeros((5, n_sites))
    P_60 =  np.zeros((5, n_sites))
    P_61 =  np.zeros((5, n_sites))
    P_62 =  np.zeros((5, n_sites))
    P_63 =  np.zeros((5, n_sites))
    P_64 =  np.zeros((5, n_sites))
    P_65 =  np.zeros((5, n_sites))
    P_66 =  np.zeros((5, n_sites))
    P_67 =  np.zeros((5, n_sites))
    P_68 =  np.zeros((5, n_sites))
    P_69 =  np.zeros((5, n_sites))
    P_70 =  np.zeros((5, n_sites))
    #;
    ## Polynomial Constants;
    #;
    constantTerm3 = 2**(-1/2)*((-1)*a1)**(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2)* \
        sigmaV**(-1)
    P_C_1 = constantTerm3
    P_C_2 = (-1)*constantTerm3*(2*a1*sigmaMuV**2+2*a1*sigmaV**2)**(-1)
    P_C_3 = (-1/2)*a1**(-1)*constantTerm3
    P_C_4 = (-1/2)*a1**(-1)*constantTerm3
    P_C_5 = (-1/2)*a1**(-1)*constantTerm3
    P_C_6 = (1/8)*a1**(-2)*constantTerm3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV*(sigmaMuV**2+sigmaV**2)**(-3)
    P_C_7 = (1/4)*a1**(-2)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_8 = (1/4)*a1**(-2)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_9 = (1/4)*a1**(-2)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_10 = (1/4)*a1**(-2)*constantTerm3
    P_C_11 = (1/4)*a1**(-2)*constantTerm3
    P_C_12 = (1/4)*a1**(-2)*constantTerm3
    P_C_13 = (1/4)*a1**(-2)*constantTerm3
    P_C_14 = (1/4)*a1**(-2)*constantTerm3
    P_C_15 = (1/4)*a1**(-2)*constantTerm3
    P_C_16 = (-1/48)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-3)
    P_C_17 = (-1/16)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_18 = (-1/16)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_19 = (-1/16)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_20 = (-1/8)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_21 = (-1/8)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_22 = (-1/8)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_23 = (-1/8)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_24 = (-1/8)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_25 = (-1/8)*a1**(-3)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_26 = (-1/8)*a1**(-3)*constantTerm3
    P_C_27 = (-1/8)*a1**(-3)*constantTerm3
    P_C_28 = (-1/8)*a1**(-3)*constantTerm3
    P_C_29 = (-1/8)*a1**(-3)*constantTerm3
    P_C_30 = (-1/8)*a1**(-3)*constantTerm3
    P_C_31 = (-1/8)*a1**(-3)*constantTerm3
    P_C_32 = (-1/8)*a1**(-3)*constantTerm3
    P_C_33 = (-1/8)*a1**(-3)*constantTerm3
    P_C_34 = (-1/8)*a1**(-3)*constantTerm3
    P_C_35 = (-1/8)*a1**(-3)*constantTerm3
    P_C_36 = (1/384)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-4)
    P_C_37 = (1/96)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-3)
    P_C_38 = (1/96)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-3)
    P_C_39 = (1/96)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-3)
    P_C_40 = (1/32)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_41 = (1/32)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_42 = (1/32)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_43 = (1/32)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_44 = (1/32)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_45 = (1/32)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-2)
    P_C_46 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_47 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_48 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_49 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_50 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_51 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_52 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_53 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_54 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_55 = (1/16)*a1**(-4)*constantTerm3*(sigmaMuV**2+sigmaV**2)**(-1)
    P_C_56 = (1/16)*a1**(-4)*constantTerm3
    P_C_57 = (1/16)*a1**(-4)*constantTerm3
    P_C_58 = (1/16)*a1**(-4)*constantTerm3
    P_C_59 = (1/16)*a1**(-4)*constantTerm3
    P_C_60 = (1/16)*a1**(-4)*constantTerm3
    P_C_61 = (1/16)*a1**(-4)*constantTerm3
    P_C_62 = (1/16)*a1**(-4)*constantTerm3
    P_C_63 = (1/16)*a1**(-4)*constantTerm3
    P_C_64 = (1/16)*a1**(-4)*constantTerm3
    P_C_65 = (1/16)*a1**(-4)*constantTerm3
    P_C_66 = (1/16)*a1**(-4)*constantTerm3
    P_C_67 = (1/16)*a1**(-4)*constantTerm3
    P_C_68 = (1/16)*a1**(-4)*constantTerm3
    P_C_69 = (1/16)*a1**(-4)*constantTerm3
    P_C_70 = (1/16)*a1**(-4)*constantTerm3
    #;
    ## Polynomial Coefficients;
    #;
    P_1[0,:] = (P_C_1)*p1c0
    P_2[0,:] = (P_C_2)*((-1)*amuV*b00+2*a1*bmuV)*p1c0*sigmaMuV
    P_2[1,:] = (-1)*(P_C_2)*(2*a1+amuV*b01)*p1c0*sigmaMuV
    P_3[0,:] = (P_C_3)*((-2)*a1*p2c0+b00*p2c1)
    P_3[1,:] = (P_C_3)*b01*p2c1
    P_4[0,:] = (P_C_4)*((-2)*a1*p3c0+b00*p3c1)
    P_4[1,:] = (P_C_4)*b01*p3c1
    P_5[0,:] = (P_C_5)*((-2)*a1*p4c0+b00*p4c1)
    P_5[1,:] = (P_C_5)*b01*p4c1
    P_6[0,:] = (P_C_6)*p1c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV*(amuV**2*b00**2+(-2)*a1*amuV*(amuV+2*b00*bmuV)+4*a1**2*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))
    P_6[1,:] = (-2)*(P_C_6)*(2*a1+amuV*b01)*((-1)*amuV*b00+2*a1*bmuV)*p1c0* \
        sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_6[2,:] = (P_C_6)*(2*a1+amuV*b01)**2*p1c0*sigmaMuV**2*(1+sigmaMuV**2* \
        sigmaV**(-2))**(1/2)*sigmaV
    P_7[0,:] = (P_C_7)*((-4)*a1**2*bmuV*p2c0+(-1)*amuV*b00**2*p2c1+2*a1*(b00* \
        bmuV*p2c1+amuV*(b00*p2c0+p2c1)))*sigmaMuV
    P_7[1,:] = 2*(P_C_7)*(2*a1**2*p2c0+(-1)*amuV*b00*b01*p2c1+a1*(amuV*b01* \
        p2c0+(-1)*b00*p2c1+b01*bmuV*p2c1))*sigmaMuV
    P_7[2,:] = (-1)*(P_C_7)*b01*(2*a1+amuV*b01)*p2c1*sigmaMuV
    P_8[0,:] = (P_C_8)*((-4)*a1**2*bmuV*p3c0+(-1)*amuV*b00**2*p3c1+2*a1*(b00* \
        bmuV*p3c1+amuV*(b00*p3c0+p3c1)))*sigmaMuV
    P_8[1,:] = 2*(P_C_8)*(2*a1**2*p3c0+(-1)*amuV*b00*b01*p3c1+a1*(amuV*b01* \
        p3c0+(-1)*b00*p3c1+b01*bmuV*p3c1))*sigmaMuV
    P_8[2,:] = (-1)*(P_C_8)*b01*(2*a1+amuV*b01)*p3c1*sigmaMuV
    P_9[0,:] = (P_C_9)*((-4)*a1**2*bmuV*p4c0+(-1)*amuV*b00**2*p4c1+2*a1*(b00* \
        bmuV*p4c1+amuV*(b00*p4c0+p4c1)))*sigmaMuV
    P_9[1,:] = 2*(P_C_9)*(2*a1**2*p4c0+(-1)*amuV*b00*b01*p4c1+a1*(amuV*b01* \
        p4c0+(-1)*b00*p4c1+b01*bmuV*p4c1))*sigmaMuV
    P_9[2,:] = (-1)*(P_C_9)*b01*(2*a1+amuV*b01)*p4c1*sigmaMuV
    P_10[0,:] = (P_C_10)*(4*a1**2*p5c0+b00**2*p5c2+(-2)*a1*(b00*p5c1+p5c2))
    P_10[1,:] = (P_C_10)*((-2)*a1*b01*p5c1+2*b00*b01*p5c2)
    P_10[2,:] = (P_C_10)*b01**2*p5c2
    P_11[0,:] = (P_C_11)*(4*a1**2*p6c0+b00**2*p6c2+(-2)*a1*(b00*p6c1+p6c2))
    P_11[1,:] = (P_C_11)*((-2)*a1*b01*p6c1+2*b00*b01*p6c2)
    P_11[2,:] = (P_C_11)*b01**2*p6c2
    P_12[0,:] = (P_C_12)*(4*a1**2*p7c0+b00**2*p7c2+(-2)*a1*(b00*p7c1+p7c2))
    P_12[1,:] = (P_C_12)*((-2)*a1*b01*p7c1+2*b00*b01*p7c2)
    P_12[2,:] = (P_C_12)*b01**2*p7c2
    P_13[0,:] = (P_C_13)*(4*a1**2*p8c0+b00**2*p8c2+(-2)*a1*(b00*p8c1+p8c2))
    P_13[1,:] = (P_C_13)*((-2)*a1*b01*p8c1+2*b00*b01*p8c2)
    P_13[2,:] = (P_C_13)*b01**2*p8c2
    P_14[0,:] = (P_C_14)*(4*a1**2*p9c0+b00**2*p9c2+(-2)*a1*(b00*p9c1+p9c2))
    P_14[1,:] = (P_C_14)*((-2)*a1*b01*p9c1+2*b00*b01*p9c2)
    P_14[2,:] = (P_C_14)*b01**2*p9c2
    P_15[0,:] = (P_C_15)*(4*a1**2*p10c0+b00**2*p10c2+(-2)*a1*(b00*p10c1+p10c2)) \
        
    P_15[1,:] = (P_C_15)*((-2)*a1*b01*p10c1+2*b00*b01*p10c2)
    P_15[2,:] = (P_C_15)*b01**2*p10c2
    P_16[0,:] = (P_C_16)*((-1)*amuV*b00+2*a1*bmuV)*p1c0*sigmaMuV**3*(amuV**2* \
        b00**2+(-2)*a1*amuV*(3*amuV+2*b00*bmuV)+4*a1**2*(bmuV**2+(-3)*( \
        sigmaMuV**2+sigmaV**2)))
    P_16[1,:] = (-3)*(P_C_16)*(2*a1+amuV*b01)*p1c0*sigmaMuV**3*(amuV**2*b00**2+( \
        -2)*a1*amuV*(amuV+2*b00*bmuV)+4*a1**2*(bmuV**2+(-1)*sigmaMuV**2+ \
        (-1)*sigmaV**2))
    P_16[2,:] = 3*(P_C_16)*(2*a1+amuV*b01)**2*((-1)*amuV*b00+2*a1*bmuV)*p1c0* \
        sigmaMuV**3
    P_16[3,:] = (-1)*(P_C_16)*(2*a1+amuV*b01)**3*p1c0*sigmaMuV**3
    P_17[0,:] = (P_C_17)*sigmaMuV**2*(amuV**2*b00**3*p2c1+(-2)*a1*amuV*b00*( \
        amuV*b00*p2c0+3*amuV*p2c1+2*b00*bmuV*p2c1)+8*a1**3*p2c0*((-1) \
        *bmuV**2+sigmaMuV**2+sigmaV**2)+4*a1**2*(amuV**2*p2c0+2*amuV* \
        bmuV*(b00*p2c0+p2c1)+b00*p2c1*(bmuV**2+(-1)*sigmaMuV**2+(-1)* \
        sigmaV**2)))
    P_17[1,:] = (-1)*(P_C_17)*sigmaMuV**2*((-16)*a1**3*bmuV*p2c0+(-3)*amuV**2* \
        b00**2*b01*p2c1+2*a1*amuV*((-2)*b00*(b00+(-2)*b01*bmuV)*p2c1+ \
        amuV*b01*(2*b00*p2c0+3*p2c1))+4*a1**2*(2*b00*bmuV*p2c1+2* \
        amuV*(b00*p2c0+(-1)*b01*bmuV*p2c0+p2c1)+b01*p2c1*((-1)*bmuV**2+ \
        sigmaMuV**2+sigmaV**2)))
    P_17[2,:] = (-1)*(P_C_17)*(2*a1+amuV*b01)*(4*a1**2*p2c0+(-3)*amuV*b00* \
        b01*p2c1+2*a1*(amuV*b01*p2c0+(-1)*b00*p2c1+2*b01*bmuV*p2c1))* \
        sigmaMuV**2
    P_17[3,:] = (P_C_17)*b01*(2*a1+amuV*b01)**2*p2c1*sigmaMuV**2
    P_18[0,:] = (P_C_18)*sigmaMuV**2*(amuV**2*b00**3*p3c1+(-2)*a1*amuV*b00*( \
        amuV*b00*p3c0+3*amuV*p3c1+2*b00*bmuV*p3c1)+8*a1**3*p3c0*((-1) \
        *bmuV**2+sigmaMuV**2+sigmaV**2)+4*a1**2*(amuV**2*p3c0+2*amuV* \
        bmuV*(b00*p3c0+p3c1)+b00*p3c1*(bmuV**2+(-1)*sigmaMuV**2+(-1)* \
        sigmaV**2)))
    P_18[1,:] = (-1)*(P_C_18)*sigmaMuV**2*((-16)*a1**3*bmuV*p3c0+(-3)*amuV**2* \
        b00**2*b01*p3c1+2*a1*amuV*((-2)*b00*(b00+(-2)*b01*bmuV)*p3c1+ \
        amuV*b01*(2*b00*p3c0+3*p3c1))+4*a1**2*(2*b00*bmuV*p3c1+2* \
        amuV*(b00*p3c0+(-1)*b01*bmuV*p3c0+p3c1)+b01*p3c1*((-1)*bmuV**2+ \
        sigmaMuV**2+sigmaV**2)))
    P_18[2,:] = (-1)*(P_C_18)*(2*a1+amuV*b01)*(4*a1**2*p3c0+(-3)*amuV*b00* \
        b01*p3c1+2*a1*(amuV*b01*p3c0+(-1)*b00*p3c1+2*b01*bmuV*p3c1))* \
        sigmaMuV**2
    P_18[3,:] = (P_C_18)*b01*(2*a1+amuV*b01)**2*p3c1*sigmaMuV**2
    P_19[0,:] = (P_C_19)*sigmaMuV**2*(amuV**2*b00**3*p4c1+(-2)*a1*amuV*b00*( \
        amuV*b00*p4c0+3*amuV*p4c1+2*b00*bmuV*p4c1)+8*a1**3*p4c0*((-1) \
        *bmuV**2+sigmaMuV**2+sigmaV**2)+4*a1**2*(amuV**2*p4c0+2*amuV* \
        bmuV*(b00*p4c0+p4c1)+b00*p4c1*(bmuV**2+(-1)*sigmaMuV**2+(-1)* \
        sigmaV**2)))
    P_19[1,:] = (-1)*(P_C_19)*sigmaMuV**2*((-16)*a1**3*bmuV*p4c0+(-3)*amuV**2* \
        b00**2*b01*p4c1+2*a1*amuV*((-2)*b00*(b00+(-2)*b01*bmuV)*p4c1+ \
        amuV*b01*(2*b00*p4c0+3*p4c1))+4*a1**2*(2*b00*bmuV*p4c1+2* \
        amuV*(b00*p4c0+(-1)*b01*bmuV*p4c0+p4c1)+b01*p4c1*((-1)*bmuV**2+ \
        sigmaMuV**2+sigmaV**2)))
    P_19[2,:] = (-1)*(P_C_19)*(2*a1+amuV*b01)*(4*a1**2*p4c0+(-3)*amuV*b00* \
        b01*p4c1+2*a1*(amuV*b01*p4c0+(-1)*b00*p4c1+2*b01*bmuV*p4c1))* \
        sigmaMuV**2
    P_19[3,:] = (P_C_19)*b01*(2*a1+amuV*b01)**2*p4c1*sigmaMuV**2
    P_20[0,:] = (P_C_20)*(8*a1**3*bmuV*p5c0+(-1)*amuV*b00**3*p5c2+2*a1*b00*( \
        amuV*b00*p5c1+3*amuV*p5c2+b00*bmuV*p5c2)+(-4)*a1**2*(amuV*( \
        b00*p5c0+p5c1)+bmuV*(b00*p5c1+p5c2)))*sigmaMuV
    P_20[1,:] = (P_C_20)*((-8)*a1**3*p5c0+(-4)*a1**2*(amuV*b01*p5c0+(-1)*b00* \
        p5c1+b01*bmuV*p5c1+(-1)*p5c2)+(-3)*amuV*b00**2*b01*p5c2+(-2)* \
        a1*b00*(b00+(-2)*b01*bmuV)*p5c2+a1*amuV*(4*b00*b01*p5c1+6* \
        b01*p5c2))*sigmaMuV
    P_20[2,:] = (P_C_20)*b01*(4*a1**2*p5c1+(-3)*amuV*b00*b01*p5c2+2*a1*(amuV* \
        b01*p5c1+(-2)*b00*p5c2+b01*bmuV*p5c2))*sigmaMuV
    P_20[3,:] = (-1)*(P_C_20)*b01**2*(2*a1+amuV*b01)*p5c2*sigmaMuV
    P_21[0,:] = (P_C_21)*(8*a1**3*bmuV*p6c0+(-1)*amuV*b00**3*p6c2+2*a1*b00*( \
        amuV*b00*p6c1+3*amuV*p6c2+b00*bmuV*p6c2)+(-4)*a1**2*(amuV*( \
        b00*p6c0+p6c1)+bmuV*(b00*p6c1+p6c2)))*sigmaMuV
    P_21[1,:] = (P_C_21)*((-8)*a1**3*p6c0+(-4)*a1**2*(amuV*b01*p6c0+(-1)*b00* \
        p6c1+b01*bmuV*p6c1+(-1)*p6c2)+(-3)*amuV*b00**2*b01*p6c2+(-2)* \
        a1*b00*(b00+(-2)*b01*bmuV)*p6c2+a1*amuV*(4*b00*b01*p6c1+6* \
        b01*p6c2))*sigmaMuV
    P_21[2,:] = (P_C_21)*b01*(4*a1**2*p6c1+(-3)*amuV*b00*b01*p6c2+2*a1*(amuV* \
        b01*p6c1+(-2)*b00*p6c2+b01*bmuV*p6c2))*sigmaMuV
    P_21[3,:] = (-1)*(P_C_21)*b01**2*(2*a1+amuV*b01)*p6c2*sigmaMuV
    P_22[0,:] = (P_C_22)*(8*a1**3*bmuV*p7c0+(-1)*amuV*b00**3*p7c2+2*a1*b00*( \
        amuV*b00*p7c1+3*amuV*p7c2+b00*bmuV*p7c2)+(-4)*a1**2*(amuV*( \
        b00*p7c0+p7c1)+bmuV*(b00*p7c1+p7c2)))*sigmaMuV
    P_22[1,:] = (P_C_22)*((-8)*a1**3*p7c0+(-4)*a1**2*(amuV*b01*p7c0+(-1)*b00* \
        p7c1+b01*bmuV*p7c1+(-1)*p7c2)+(-3)*amuV*b00**2*b01*p7c2+(-2)* \
        a1*b00*(b00+(-2)*b01*bmuV)*p7c2+a1*amuV*(4*b00*b01*p7c1+6* \
        b01*p7c2))*sigmaMuV
    P_22[2,:] = (P_C_22)*b01*(4*a1**2*p7c1+(-3)*amuV*b00*b01*p7c2+2*a1*(amuV* \
        b01*p7c1+(-2)*b00*p7c2+b01*bmuV*p7c2))*sigmaMuV
    P_22[3,:] = (-1)*(P_C_22)*b01**2*(2*a1+amuV*b01)*p7c2*sigmaMuV
    P_23[0,:] = (P_C_23)*(8*a1**3*bmuV*p8c0+(-1)*amuV*b00**3*p8c2+2*a1*b00*( \
        amuV*b00*p8c1+3*amuV*p8c2+b00*bmuV*p8c2)+(-4)*a1**2*(amuV*( \
        b00*p8c0+p8c1)+bmuV*(b00*p8c1+p8c2)))*sigmaMuV
    P_23[1,:] = (P_C_23)*((-8)*a1**3*p8c0+(-4)*a1**2*(amuV*b01*p8c0+(-1)*b00* \
        p8c1+b01*bmuV*p8c1+(-1)*p8c2)+(-3)*amuV*b00**2*b01*p8c2+(-2)* \
        a1*b00*(b00+(-2)*b01*bmuV)*p8c2+a1*amuV*(4*b00*b01*p8c1+6* \
        b01*p8c2))*sigmaMuV
    P_23[2,:] = (P_C_23)*b01*(4*a1**2*p8c1+(-3)*amuV*b00*b01*p8c2+2*a1*(amuV* \
        b01*p8c1+(-2)*b00*p8c2+b01*bmuV*p8c2))*sigmaMuV
    P_23[3,:] = (-1)*(P_C_23)*b01**2*(2*a1+amuV*b01)*p8c2*sigmaMuV
    P_24[0,:] = (P_C_24)*(8*a1**3*bmuV*p9c0+(-1)*amuV*b00**3*p9c2+2*a1*b00*( \
        amuV*b00*p9c1+3*amuV*p9c2+b00*bmuV*p9c2)+(-4)*a1**2*(amuV*( \
        b00*p9c0+p9c1)+bmuV*(b00*p9c1+p9c2)))*sigmaMuV
    P_24[1,:] = (P_C_24)*((-8)*a1**3*p9c0+(-4)*a1**2*(amuV*b01*p9c0+(-1)*b00* \
        p9c1+b01*bmuV*p9c1+(-1)*p9c2)+(-3)*amuV*b00**2*b01*p9c2+(-2)* \
        a1*b00*(b00+(-2)*b01*bmuV)*p9c2+a1*amuV*(4*b00*b01*p9c1+6* \
        b01*p9c2))*sigmaMuV
    P_24[2,:] = (P_C_24)*b01*(4*a1**2*p9c1+(-3)*amuV*b00*b01*p9c2+2*a1*(amuV* \
        b01*p9c1+(-2)*b00*p9c2+b01*bmuV*p9c2))*sigmaMuV
    P_24[3,:] = (-1)*(P_C_24)*b01**2*(2*a1+amuV*b01)*p9c2*sigmaMuV
    P_25[0,:] = (P_C_25)*(8*a1**3*bmuV*p10c0+(-1)*amuV*b00**3*p10c2+2*a1*b00*( \
        amuV*b00*p10c1+3*amuV*p10c2+b00*bmuV*p10c2)+(-4)*a1**2*(amuV*( \
        b00*p10c0+p10c1)+bmuV*(b00*p10c1+p10c2)))*sigmaMuV
    P_25[1,:] = (P_C_25)*((-8)*a1**3*p10c0+(-4)*a1**2*(amuV*b01*p10c0+(-1)*b00* \
        p10c1+b01*bmuV*p10c1+(-1)*p10c2)+(-3)*amuV*b00**2*b01*p10c2+(-2) \
        *a1*b00*(b00+(-2)*b01*bmuV)*p10c2+a1*amuV*(4*b00*b01*p10c1+ \
        6*b01*p10c2))*sigmaMuV
    P_25[2,:] = (P_C_25)*b01*(4*a1**2*p10c1+(-3)*amuV*b00*b01*p10c2+2*a1*( \
        amuV*b01*p10c1+(-2)*b00*p10c2+b01*bmuV*p10c2))*sigmaMuV
    P_25[3,:] = (-1)*(P_C_25)*b01**2*(2*a1+amuV*b01)*p10c2*sigmaMuV
    P_26[0,:] = (P_C_26)*((-8)*a1**3*p11c0+4*a1**2*(b00*p11c1+p11c2)+b00**3* \
        p11c3+(-2)*a1*b00*(b00*p11c2+3*p11c3))
    P_26[1,:] = (P_C_26)*b01*(4*a1**2*p11c1+(-4)*a1*b00*p11c2+(-6)*a1*p11c3+3* \
        b00**2*p11c3)
    P_26[2,:] = (P_C_26)*b01**2*((-2)*a1*p11c2+3*b00*p11c3)
    P_26[3,:] = (P_C_26)*b01**3*p11c3
    P_27[0,:] = (P_C_27)*((-8)*a1**3*p12c0+4*a1**2*(b00*p12c1+p12c2)+b00**3* \
        p12c3+(-2)*a1*b00*(b00*p12c2+3*p12c3))
    P_27[1,:] = (P_C_27)*b01*(4*a1**2*p12c1+(-4)*a1*b00*p12c2+(-6)*a1*p12c3+3* \
        b00**2*p12c3)
    P_27[2,:] = (P_C_27)*b01**2*((-2)*a1*p12c2+3*b00*p12c3)
    P_27[3,:] = (P_C_27)*b01**3*p12c3
    P_28[0,:] = (P_C_28)*((-8)*a1**3*p13c0+4*a1**2*(b00*p13c1+p13c2)+b00**3* \
        p13c3+(-2)*a1*b00*(b00*p13c2+3*p13c3))
    P_28[1,:] = (P_C_28)*b01*(4*a1**2*p13c1+(-4)*a1*b00*p13c2+(-6)*a1*p13c3+3* \
        b00**2*p13c3)
    P_28[2,:] = (P_C_28)*b01**2*((-2)*a1*p13c2+3*b00*p13c3)
    P_28[3,:] = (P_C_28)*b01**3*p13c3
    P_29[0,:] = (P_C_29)*((-8)*a1**3*p14c0+4*a1**2*(b00*p14c1+p14c2)+b00**3* \
        p14c3+(-2)*a1*b00*(b00*p14c2+3*p14c3))
    P_29[1,:] = (P_C_29)*b01*(4*a1**2*p14c1+(-4)*a1*b00*p14c2+(-6)*a1*p14c3+3* \
        b00**2*p14c3)
    P_29[2,:] = (P_C_29)*b01**2*((-2)*a1*p14c2+3*b00*p14c3)
    P_29[3,:] = (P_C_29)*b01**3*p14c3
    P_30[0,:] = (P_C_30)*((-8)*a1**3*p15c0+4*a1**2*(b00*p15c1+p15c2)+b00**3* \
        p15c3+(-2)*a1*b00*(b00*p15c2+3*p15c3))
    P_30[1,:] = (P_C_30)*b01*(4*a1**2*p15c1+(-4)*a1*b00*p15c2+(-6)*a1*p15c3+3* \
        b00**2*p15c3)
    P_30[2,:] = (P_C_30)*b01**2*((-2)*a1*p15c2+3*b00*p15c3)
    P_30[3,:] = (P_C_30)*b01**3*p15c3
    P_31[0,:] = (P_C_31)*((-8)*a1**3*p16c0+4*a1**2*(b00*p16c1+p16c2)+b00**3* \
        p16c3+(-2)*a1*b00*(b00*p16c2+3*p16c3))
    P_31[1,:] = (P_C_31)*b01*(4*a1**2*p16c1+(-4)*a1*b00*p16c2+(-6)*a1*p16c3+3* \
        b00**2*p16c3)
    P_31[2,:] = (P_C_31)*b01**2*((-2)*a1*p16c2+3*b00*p16c3)
    P_31[3,:] = (P_C_31)*b01**3*p16c3
    P_32[0,:] = (P_C_32)*((-8)*a1**3*p17c0+4*a1**2*(b00*p17c1+p17c2)+b00**3* \
        p17c3+(-2)*a1*b00*(b00*p17c2+3*p17c3))
    P_32[1,:] = (P_C_32)*b01*(4*a1**2*p17c1+(-4)*a1*b00*p17c2+(-6)*a1*p17c3+3* \
        b00**2*p17c3)
    P_32[2,:] = (P_C_32)*b01**2*((-2)*a1*p17c2+3*b00*p17c3)
    P_32[3,:] = (P_C_32)*b01**3*p17c3
    P_33[0,:] = (P_C_33)*((-8)*a1**3*p18c0+4*a1**2*(b00*p18c1+p18c2)+b00**3* \
        p18c3+(-2)*a1*b00*(b00*p18c2+3*p18c3))
    P_33[1,:] = (P_C_33)*b01*(4*a1**2*p18c1+(-4)*a1*b00*p18c2+(-6)*a1*p18c3+3* \
        b00**2*p18c3)
    P_33[2,:] = (P_C_33)*b01**2*((-2)*a1*p18c2+3*b00*p18c3)
    P_33[3,:] = (P_C_33)*b01**3*p18c3
    P_34[0,:] = (P_C_34)*((-8)*a1**3*p19c0+4*a1**2*(b00*p19c1+p19c2)+b00**3* \
        p19c3+(-2)*a1*b00*(b00*p19c2+3*p19c3))
    P_34[1,:] = (P_C_34)*b01*(4*a1**2*p19c1+(-4)*a1*b00*p19c2+(-6)*a1*p19c3+3* \
        b00**2*p19c3)
    P_34[2,:] = (P_C_34)*b01**2*((-2)*a1*p19c2+3*b00*p19c3)
    P_34[3,:] = (P_C_34)*b01**3*p19c3
    P_35[0,:] = (P_C_35)*((-8)*a1**3*p20c0+4*a1**2*(b00*p20c1+p20c2)+b00**3* \
        p20c3+(-2)*a1*b00*(b00*p20c2+3*p20c3))
    P_35[1,:] = (P_C_35)*b01*(4*a1**2*p20c1+(-4)*a1*b00*p20c2+(-6)*a1*p20c3+3* \
        b00**2*p20c3)
    P_35[2,:] = (P_C_35)*b01**2*((-2)*a1*p20c2+3*b00*p20c3)
    P_35[3,:] = (P_C_35)*b01**3*p20c3
    P_36[0,:] = (P_C_36)*p1c0*sigmaMuV**4*(amuV**4*b00**4+(-4)*a1*amuV**3* \
        b00**2*(3*amuV+2*b00*bmuV)+12*a1**2*amuV**2*(amuV**2+4*amuV* \
        b00*bmuV+2*b00**2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+16* \
        a1**4*(bmuV**4+(-6)*bmuV**2*(sigmaMuV**2+sigmaV**2)+3*(sigmaMuV**2+ \
        sigmaV**2)**2)+(-16)*a1**3*amuV*(3*amuV*(bmuV**2+(-1)*sigmaMuV**2+ \
        (-1)*sigmaV**2)+2*b00*bmuV*(bmuV**2+(-3)*(sigmaMuV**2+sigmaV**2)))) \
        
    P_36[1,:] = (-4)*(P_C_36)*(2*a1+amuV*b01)*((-1)*amuV*b00+2*a1*bmuV)*p1c0* \
        sigmaMuV**4*(amuV**2*b00**2+(-2)*a1*amuV*(3*amuV+2*b00*bmuV)+4* \
        a1**2*(bmuV**2+(-3)*(sigmaMuV**2+sigmaV**2)))
    P_36[2,:] = 6*(P_C_36)*(2*a1+amuV*b01)**2*p1c0*sigmaMuV**4*(amuV**2*b00**2+( \
        -2)*a1*amuV*(amuV+2*b00*bmuV)+4*a1**2*(bmuV**2+(-1)*sigmaMuV**2+ \
        (-1)*sigmaV**2))
    P_36[3,:] = (-4)*(P_C_36)*(2*a1+amuV*b01)**3*((-1)*amuV*b00+2*a1*bmuV)* \
        p1c0*sigmaMuV**4
    P_36[4,:] = (P_C_36)*(2*a1+amuV*b01)**4*p1c0*sigmaMuV**4
    P_37[0,:] = (P_C_37)*sigmaMuV**3*((-1)*amuV**3*b00**4*p2c1+2*a1*amuV**2* \
        b00**2*(amuV*b00*p2c0+6*amuV*p2c1+3*b00*bmuV*p2c1)+(-12)* \
        a1**2*amuV*(amuV**2*(b00*p2c0+p2c1)+amuV*b00*bmuV*(b00*p2c0+3* \
        p2c1)+b00**2*p2c1*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-16)* \
        a1**4*bmuV*p2c0*(bmuV**2+(-3)*(sigmaMuV**2+sigmaV**2))+8*a1**3*( \
        3*amuV**2*bmuV*p2c0+3*amuV*(b00*p2c0+p2c1)*(bmuV**2+(-1)* \
        sigmaMuV**2+(-1)*sigmaV**2)+b00*bmuV*p2c1*(bmuV**2+(-3)*( \
        sigmaMuV**2+sigmaV**2))))
    P_37[1,:] = 2*(P_C_37)*sigmaMuV**3*((-2)*amuV**3*b00**3*b01*p2c1+3*a1* \
        amuV**2*b00*((-1)*b00*(b00+(-3)*b01*bmuV)*p2c1+amuV*b01*(b00* \
        p2c0+4*p2c1))+24*a1**4*p2c0*(bmuV**2+(-1)*sigmaMuV**2+(-1)* \
        sigmaV**2)+(-6)*a1**2*amuV*(amuV**2*b01*p2c0+amuV*((-1)*b00**2* \
        p2c0+2*b00*b01*bmuV*p2c0+(-3)*b00*p2c1+3*b01*bmuV*p2c1)+(-2)* \
        b00*p2c1*(b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+sigmaV**2)))+(-4) \
        *a1**3*(3*amuV**2*p2c0+p2c1*((-1)*b01*bmuV**3+3*b00*(bmuV**2+( \
        -1)*sigmaMuV**2+(-1)*sigmaV**2)+3*b01*bmuV*(sigmaMuV**2+sigmaV**2)) \
        +3*amuV*(2*b00*bmuV*p2c0+2*bmuV*p2c1+b01*p2c0*((-1)*bmuV**2+ \
        sigmaMuV**2+sigmaV**2))))
    P_37[2,:] = (-6)*(P_C_37)*(2*a1+amuV*b01)*sigmaMuV**3*(4*a1**3*bmuV*p2c0+ \
        amuV**2*b00**2*b01*p2c1+(-1)*a1*amuV*((-1)*b00*(b00+(-3)*b01* \
        bmuV)*p2c1+amuV*b01*(b00*p2c0+2*p2c1))+(-2)*a1**2*(b00*bmuV* \
        p2c1+amuV*(b00*p2c0+(-1)*b01*bmuV*p2c0+p2c1)+b01*p2c1*((-1)* \
        bmuV**2+sigmaMuV**2+sigmaV**2)))
    P_37[3,:] = 2*(P_C_37)*(2*a1+amuV*b01)**2*(2*a1**2*p2c0+(-2)*amuV*b00* \
        b01*p2c1+a1*(amuV*b01*p2c0+(-1)*b00*p2c1+3*b01*bmuV*p2c1))* \
        sigmaMuV**3
    P_37[4,:] = (-1)*(P_C_37)*b01*(2*a1+amuV*b01)**3*p2c1*sigmaMuV**3
    P_38[0,:] = (P_C_38)*sigmaMuV**3*((-1)*amuV**3*b00**4*p3c1+2*a1*amuV**2* \
        b00**2*(amuV*b00*p3c0+6*amuV*p3c1+3*b00*bmuV*p3c1)+(-12)* \
        a1**2*amuV*(amuV**2*(b00*p3c0+p3c1)+amuV*b00*bmuV*(b00*p3c0+3* \
        p3c1)+b00**2*p3c1*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-16)* \
        a1**4*bmuV*p3c0*(bmuV**2+(-3)*(sigmaMuV**2+sigmaV**2))+8*a1**3*( \
        3*amuV**2*bmuV*p3c0+3*amuV*(b00*p3c0+p3c1)*(bmuV**2+(-1)* \
        sigmaMuV**2+(-1)*sigmaV**2)+b00*bmuV*p3c1*(bmuV**2+(-3)*( \
        sigmaMuV**2+sigmaV**2))))
    P_38[1,:] = 2*(P_C_38)*sigmaMuV**3*((-2)*amuV**3*b00**3*b01*p3c1+3*a1* \
        amuV**2*b00*((-1)*b00*(b00+(-3)*b01*bmuV)*p3c1+amuV*b01*(b00* \
        p3c0+4*p3c1))+24*a1**4*p3c0*(bmuV**2+(-1)*sigmaMuV**2+(-1)* \
        sigmaV**2)+(-6)*a1**2*amuV*(amuV**2*b01*p3c0+amuV*((-1)*b00**2* \
        p3c0+2*b00*b01*bmuV*p3c0+(-3)*b00*p3c1+3*b01*bmuV*p3c1)+(-2)* \
        b00*p3c1*(b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+sigmaV**2)))+(-4) \
        *a1**3*(3*amuV**2*p3c0+p3c1*((-1)*b01*bmuV**3+3*b00*(bmuV**2+( \
        -1)*sigmaMuV**2+(-1)*sigmaV**2)+3*b01*bmuV*(sigmaMuV**2+sigmaV**2)) \
        +3*amuV*(2*b00*bmuV*p3c0+2*bmuV*p3c1+b01*p3c0*((-1)*bmuV**2+ \
        sigmaMuV**2+sigmaV**2))))
    P_38[2,:] = (-6)*(P_C_38)*(2*a1+amuV*b01)*sigmaMuV**3*(4*a1**3*bmuV*p3c0+ \
        amuV**2*b00**2*b01*p3c1+(-1)*a1*amuV*((-1)*b00*(b00+(-3)*b01* \
        bmuV)*p3c1+amuV*b01*(b00*p3c0+2*p3c1))+(-2)*a1**2*(b00*bmuV* \
        p3c1+amuV*(b00*p3c0+(-1)*b01*bmuV*p3c0+p3c1)+b01*p3c1*((-1)* \
        bmuV**2+sigmaMuV**2+sigmaV**2)))
    P_38[3,:] = 2*(P_C_38)*(2*a1+amuV*b01)**2*(2*a1**2*p3c0+(-2)*amuV*b00* \
        b01*p3c1+a1*(amuV*b01*p3c0+(-1)*b00*p3c1+3*b01*bmuV*p3c1))* \
        sigmaMuV**3
    P_38[4,:] = (-1)*(P_C_38)*b01*(2*a1+amuV*b01)**3*p3c1*sigmaMuV**3
    P_39[0,:] = (P_C_39)*sigmaMuV**3*((-1)*amuV**3*b00**4*p4c1+2*a1*amuV**2* \
        b00**2*(amuV*b00*p4c0+6*amuV*p4c1+3*b00*bmuV*p4c1)+(-12)* \
        a1**2*amuV*(amuV**2*(b00*p4c0+p4c1)+amuV*b00*bmuV*(b00*p4c0+3* \
        p4c1)+b00**2*p4c1*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-16)* \
        a1**4*bmuV*p4c0*(bmuV**2+(-3)*(sigmaMuV**2+sigmaV**2))+8*a1**3*( \
        3*amuV**2*bmuV*p4c0+3*amuV*(b00*p4c0+p4c1)*(bmuV**2+(-1)* \
        sigmaMuV**2+(-1)*sigmaV**2)+b00*bmuV*p4c1*(bmuV**2+(-3)*( \
        sigmaMuV**2+sigmaV**2))))
    P_39[1,:] = 2*(P_C_39)*sigmaMuV**3*((-2)*amuV**3*b00**3*b01*p4c1+3*a1* \
        amuV**2*b00*((-1)*b00*(b00+(-3)*b01*bmuV)*p4c1+amuV*b01*(b00* \
        p4c0+4*p4c1))+24*a1**4*p4c0*(bmuV**2+(-1)*sigmaMuV**2+(-1)* \
        sigmaV**2)+(-6)*a1**2*amuV*(amuV**2*b01*p4c0+amuV*((-1)*b00**2* \
        p4c0+2*b00*b01*bmuV*p4c0+(-3)*b00*p4c1+3*b01*bmuV*p4c1)+(-2)* \
        b00*p4c1*(b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+sigmaV**2)))+(-4) \
        *a1**3*(3*amuV**2*p4c0+p4c1*((-1)*b01*bmuV**3+3*b00*(bmuV**2+( \
        -1)*sigmaMuV**2+(-1)*sigmaV**2)+3*b01*bmuV*(sigmaMuV**2+sigmaV**2)) \
        +3*amuV*(2*b00*bmuV*p4c0+2*bmuV*p4c1+b01*p4c0*((-1)*bmuV**2+ \
        sigmaMuV**2+sigmaV**2))))
    P_39[2,:] = (-6)*(P_C_39)*(2*a1+amuV*b01)*sigmaMuV**3*(4*a1**3*bmuV*p4c0+ \
        amuV**2*b00**2*b01*p4c1+(-1)*a1*amuV*((-1)*b00*(b00+(-3)*b01* \
        bmuV)*p4c1+amuV*b01*(b00*p4c0+2*p4c1))+(-2)*a1**2*(b00*bmuV* \
        p4c1+amuV*(b00*p4c0+(-1)*b01*bmuV*p4c0+p4c1)+b01*p4c1*((-1)* \
        bmuV**2+sigmaMuV**2+sigmaV**2)))
    P_39[3,:] = 2*(P_C_39)*(2*a1+amuV*b01)**2*(2*a1**2*p4c0+(-2)*amuV*b00* \
        b01*p4c1+a1*(amuV*b01*p4c0+(-1)*b00*p4c1+3*b01*bmuV*p4c1))* \
        sigmaMuV**3
    P_39[4,:] = (-1)*(P_C_39)*b01*(2*a1+amuV*b01)**3*p4c1*sigmaMuV**3
    P_40[0,:] = (P_C_40)*sigmaMuV**2*(amuV**2*b00**4*p5c2+(-2)*a1*amuV*b00**2*( \
        amuV*b00*p5c1+6*amuV*p5c2+2*b00*bmuV*p5c2)+16*a1**4*p5c0*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)+4*a1**2*(2*amuV*b00* \
        bmuV*(b00*p5c1+3*p5c2)+amuV**2*(b00**2*p5c0+3*b00*p5c1+3*p5c2)+ \
        b00**2*p5c2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-8)*a1**3* \
        (amuV**2*p5c0+2*amuV*bmuV*(b00*p5c0+p5c1)+(b00*p5c1+p5c2)*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)))
    P_40[1,:] = 2*(P_C_40)*sigmaMuV**2*((-16)*a1**4*bmuV*p5c0+2*amuV**2*b00**3* \
        b01*p5c2+(-1)*a1*amuV*b00*((-2)*b00*(b00+(-3)*b01*bmuV)*p5c2+ \
        3*amuV*b01*(b00*p5c1+4*p5c2))+4*a1**3*(2*b00*bmuV*p5c1+(-1)* \
        b01*bmuV**2*p5c1+2*amuV*(b00*p5c0+(-1)*b01*bmuV*p5c0+p5c1)+2* \
        bmuV*p5c2+b01*p5c1*sigmaMuV**2+b01*p5c1*sigmaV**2)+2*a1**2*( \
        amuV**2*b01*(2*b00*p5c0+3*p5c1)+amuV*((-2)*b00**2*p5c1+4*b00* \
        b01*bmuV*p5c1+(-6)*b00*p5c2+6*b01*bmuV*p5c2)+(-2)*b00*p5c2*( \
        b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+sigmaV**2))))
    P_40[2,:] = 2*(P_C_40)*sigmaMuV**2*(8*a1**4*p5c0+3*amuV**2*b00**2*b01**2* \
        p5c2+a1**3*(8*amuV*b01*p5c0+(-4)*(b00*p5c1+(-2)*b01*bmuV*p5c1+ \
        p5c2))+(-3)*a1*amuV*b01*((-2)*b00*(b00+(-1)*b01*bmuV)*p5c2+ \
        amuV*b01*(b00*p5c1+2*p5c2))+2*a1**2*(amuV**2*b01**2*p5c0+2* \
        amuV*b01*((-2)*b00*p5c1+b01*bmuV*p5c1+(-3)*p5c2)+p5c2*(b00**2+( \
        -4)*b00*b01*bmuV+b01**2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)) \
        ))
    P_40[3,:] = (-2)*(P_C_40)*b01*(2*a1+amuV*b01)*(2*a1**2*p5c1+(-2)*amuV* \
        b00*b01*p5c2+a1*(amuV*b01*p5c1+(-2)*b00*p5c2+2*b01*bmuV*p5c2)) \
        *sigmaMuV**2
    P_40[4,:] = (P_C_40)*b01**2*(2*a1+amuV*b01)**2*p5c2*sigmaMuV**2
    P_41[0,:] = (P_C_41)*sigmaMuV**2*(amuV**2*b00**4*p6c2+(-2)*a1*amuV*b00**2*( \
        amuV*b00*p6c1+6*amuV*p6c2+2*b00*bmuV*p6c2)+16*a1**4*p6c0*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)+4*a1**2*(2*amuV*b00* \
        bmuV*(b00*p6c1+3*p6c2)+amuV**2*(b00**2*p6c0+3*b00*p6c1+3*p6c2)+ \
        b00**2*p6c2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-8)*a1**3* \
        (amuV**2*p6c0+2*amuV*bmuV*(b00*p6c0+p6c1)+(b00*p6c1+p6c2)*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)))
    P_41[1,:] = 2*(P_C_41)*sigmaMuV**2*((-16)*a1**4*bmuV*p6c0+2*amuV**2*b00**3* \
        b01*p6c2+(-1)*a1*amuV*b00*((-2)*b00*(b00+(-3)*b01*bmuV)*p6c2+ \
        3*amuV*b01*(b00*p6c1+4*p6c2))+4*a1**3*(2*b00*bmuV*p6c1+(-1)* \
        b01*bmuV**2*p6c1+2*amuV*(b00*p6c0+(-1)*b01*bmuV*p6c0+p6c1)+2* \
        bmuV*p6c2+b01*p6c1*sigmaMuV**2+b01*p6c1*sigmaV**2)+2*a1**2*( \
        amuV**2*b01*(2*b00*p6c0+3*p6c1)+amuV*((-2)*b00**2*p6c1+4*b00* \
        b01*bmuV*p6c1+(-6)*b00*p6c2+6*b01*bmuV*p6c2)+(-2)*b00*p6c2*( \
        b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+sigmaV**2))))
    P_41[2,:] = 2*(P_C_41)*sigmaMuV**2*(8*a1**4*p6c0+3*amuV**2*b00**2*b01**2* \
        p6c2+a1**3*(8*amuV*b01*p6c0+(-4)*(b00*p6c1+(-2)*b01*bmuV*p6c1+ \
        p6c2))+(-3)*a1*amuV*b01*((-2)*b00*(b00+(-1)*b01*bmuV)*p6c2+ \
        amuV*b01*(b00*p6c1+2*p6c2))+2*a1**2*(amuV**2*b01**2*p6c0+2* \
        amuV*b01*((-2)*b00*p6c1+b01*bmuV*p6c1+(-3)*p6c2)+p6c2*(b00**2+( \
        -4)*b00*b01*bmuV+b01**2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)) \
        ))
    P_41[3,:] = (-2)*(P_C_41)*b01*(2*a1+amuV*b01)*(2*a1**2*p6c1+(-2)*amuV* \
        b00*b01*p6c2+a1*(amuV*b01*p6c1+(-2)*b00*p6c2+2*b01*bmuV*p6c2)) \
        *sigmaMuV**2
    P_41[4,:] = (P_C_41)*b01**2*(2*a1+amuV*b01)**2*p6c2*sigmaMuV**2
    P_42[0,:] = (P_C_42)*sigmaMuV**2*(amuV**2*b00**4*p7c2+(-2)*a1*amuV*b00**2*( \
        amuV*b00*p7c1+6*amuV*p7c2+2*b00*bmuV*p7c2)+16*a1**4*p7c0*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)+4*a1**2*(2*amuV*b00* \
        bmuV*(b00*p7c1+3*p7c2)+amuV**2*(b00**2*p7c0+3*b00*p7c1+3*p7c2)+ \
        b00**2*p7c2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-8)*a1**3* \
        (amuV**2*p7c0+2*amuV*bmuV*(b00*p7c0+p7c1)+(b00*p7c1+p7c2)*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)))
    P_42[1,:] = 2*(P_C_42)*sigmaMuV**2*((-16)*a1**4*bmuV*p7c0+2*amuV**2*b00**3* \
        b01*p7c2+(-1)*a1*amuV*b00*((-2)*b00*(b00+(-3)*b01*bmuV)*p7c2+ \
        3*amuV*b01*(b00*p7c1+4*p7c2))+4*a1**3*(2*b00*bmuV*p7c1+(-1)* \
        b01*bmuV**2*p7c1+2*amuV*(b00*p7c0+(-1)*b01*bmuV*p7c0+p7c1)+2* \
        bmuV*p7c2+b01*p7c1*sigmaMuV**2+b01*p7c1*sigmaV**2)+2*a1**2*( \
        amuV**2*b01*(2*b00*p7c0+3*p7c1)+amuV*((-2)*b00**2*p7c1+4*b00* \
        b01*bmuV*p7c1+(-6)*b00*p7c2+6*b01*bmuV*p7c2)+(-2)*b00*p7c2*( \
        b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+sigmaV**2))))
    P_42[2,:] = 2*(P_C_42)*sigmaMuV**2*(8*a1**4*p7c0+3*amuV**2*b00**2*b01**2* \
        p7c2+a1**3*(8*amuV*b01*p7c0+(-4)*(b00*p7c1+(-2)*b01*bmuV*p7c1+ \
        p7c2))+(-3)*a1*amuV*b01*((-2)*b00*(b00+(-1)*b01*bmuV)*p7c2+ \
        amuV*b01*(b00*p7c1+2*p7c2))+2*a1**2*(amuV**2*b01**2*p7c0+2* \
        amuV*b01*((-2)*b00*p7c1+b01*bmuV*p7c1+(-3)*p7c2)+p7c2*(b00**2+( \
        -4)*b00*b01*bmuV+b01**2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)) \
        ))
    P_42[3,:] = (-2)*(P_C_42)*b01*(2*a1+amuV*b01)*(2*a1**2*p7c1+(-2)*amuV* \
        b00*b01*p7c2+a1*(amuV*b01*p7c1+(-2)*b00*p7c2+2*b01*bmuV*p7c2)) \
        *sigmaMuV**2
    P_42[4,:] = (P_C_42)*b01**2*(2*a1+amuV*b01)**2*p7c2*sigmaMuV**2
    P_43[0,:] = (P_C_43)*sigmaMuV**2*(amuV**2*b00**4*p8c2+(-2)*a1*amuV*b00**2*( \
        amuV*b00*p8c1+6*amuV*p8c2+2*b00*bmuV*p8c2)+16*a1**4*p8c0*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)+4*a1**2*(2*amuV*b00* \
        bmuV*(b00*p8c1+3*p8c2)+amuV**2*(b00**2*p8c0+3*b00*p8c1+3*p8c2)+ \
        b00**2*p8c2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-8)*a1**3* \
        (amuV**2*p8c0+2*amuV*bmuV*(b00*p8c0+p8c1)+(b00*p8c1+p8c2)*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)))
    P_43[1,:] = 2*(P_C_43)*sigmaMuV**2*((-16)*a1**4*bmuV*p8c0+2*amuV**2*b00**3* \
        b01*p8c2+(-1)*a1*amuV*b00*((-2)*b00*(b00+(-3)*b01*bmuV)*p8c2+ \
        3*amuV*b01*(b00*p8c1+4*p8c2))+4*a1**3*(2*b00*bmuV*p8c1+(-1)* \
        b01*bmuV**2*p8c1+2*amuV*(b00*p8c0+(-1)*b01*bmuV*p8c0+p8c1)+2* \
        bmuV*p8c2+b01*p8c1*sigmaMuV**2+b01*p8c1*sigmaV**2)+2*a1**2*( \
        amuV**2*b01*(2*b00*p8c0+3*p8c1)+amuV*((-2)*b00**2*p8c1+4*b00* \
        b01*bmuV*p8c1+(-6)*b00*p8c2+6*b01*bmuV*p8c2)+(-2)*b00*p8c2*( \
        b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+sigmaV**2))))
    P_43[2,:] = 2*(P_C_43)*sigmaMuV**2*(8*a1**4*p8c0+3*amuV**2*b00**2*b01**2* \
        p8c2+a1**3*(8*amuV*b01*p8c0+(-4)*(b00*p8c1+(-2)*b01*bmuV*p8c1+ \
        p8c2))+(-3)*a1*amuV*b01*((-2)*b00*(b00+(-1)*b01*bmuV)*p8c2+ \
        amuV*b01*(b00*p8c1+2*p8c2))+2*a1**2*(amuV**2*b01**2*p8c0+2* \
        amuV*b01*((-2)*b00*p8c1+b01*bmuV*p8c1+(-3)*p8c2)+p8c2*(b00**2+( \
        -4)*b00*b01*bmuV+b01**2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)) \
        ))
    P_43[3,:] = (-2)*(P_C_43)*b01*(2*a1+amuV*b01)*(2*a1**2*p8c1+(-2)*amuV* \
        b00*b01*p8c2+a1*(amuV*b01*p8c1+(-2)*b00*p8c2+2*b01*bmuV*p8c2)) \
        *sigmaMuV**2
    P_43[4,:] = (P_C_43)*b01**2*(2*a1+amuV*b01)**2*p8c2*sigmaMuV**2
    P_44[0,:] = (P_C_44)*sigmaMuV**2*(amuV**2*b00**4*p9c2+(-2)*a1*amuV*b00**2*( \
        amuV*b00*p9c1+6*amuV*p9c2+2*b00*bmuV*p9c2)+16*a1**4*p9c0*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)+4*a1**2*(2*amuV*b00* \
        bmuV*(b00*p9c1+3*p9c2)+amuV**2*(b00**2*p9c0+3*b00*p9c1+3*p9c2)+ \
        b00**2*p9c2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-8)*a1**3* \
        (amuV**2*p9c0+2*amuV*bmuV*(b00*p9c0+p9c1)+(b00*p9c1+p9c2)*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)))
    P_44[1,:] = 2*(P_C_44)*sigmaMuV**2*((-16)*a1**4*bmuV*p9c0+2*amuV**2*b00**3* \
        b01*p9c2+(-1)*a1*amuV*b00*((-2)*b00*(b00+(-3)*b01*bmuV)*p9c2+ \
        3*amuV*b01*(b00*p9c1+4*p9c2))+4*a1**3*(2*b00*bmuV*p9c1+(-1)* \
        b01*bmuV**2*p9c1+2*amuV*(b00*p9c0+(-1)*b01*bmuV*p9c0+p9c1)+2* \
        bmuV*p9c2+b01*p9c1*sigmaMuV**2+b01*p9c1*sigmaV**2)+2*a1**2*( \
        amuV**2*b01*(2*b00*p9c0+3*p9c1)+amuV*((-2)*b00**2*p9c1+4*b00* \
        b01*bmuV*p9c1+(-6)*b00*p9c2+6*b01*bmuV*p9c2)+(-2)*b00*p9c2*( \
        b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+sigmaV**2))))
    P_44[2,:] = 2*(P_C_44)*sigmaMuV**2*(8*a1**4*p9c0+3*amuV**2*b00**2*b01**2* \
        p9c2+a1**3*(8*amuV*b01*p9c0+(-4)*(b00*p9c1+(-2)*b01*bmuV*p9c1+ \
        p9c2))+(-3)*a1*amuV*b01*((-2)*b00*(b00+(-1)*b01*bmuV)*p9c2+ \
        amuV*b01*(b00*p9c1+2*p9c2))+2*a1**2*(amuV**2*b01**2*p9c0+2* \
        amuV*b01*((-2)*b00*p9c1+b01*bmuV*p9c1+(-3)*p9c2)+p9c2*(b00**2+( \
        -4)*b00*b01*bmuV+b01**2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)) \
        ))
    P_44[3,:] = (-2)*(P_C_44)*b01*(2*a1+amuV*b01)*(2*a1**2*p9c1+(-2)*amuV* \
        b00*b01*p9c2+a1*(amuV*b01*p9c1+(-2)*b00*p9c2+2*b01*bmuV*p9c2)) \
        *sigmaMuV**2
    P_44[4,:] = (P_C_44)*b01**2*(2*a1+amuV*b01)**2*p9c2*sigmaMuV**2
    P_45[0,:] = (P_C_45)*sigmaMuV**2*(amuV**2*b00**4*p10c2+(-2)*a1*amuV*b00**2*( \
        amuV*b00*p10c1+6*amuV*p10c2+2*b00*bmuV*p10c2)+16*a1**4*p10c0*( \
        bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)+4*a1**2*(2*amuV*b00* \
        bmuV*(b00*p10c1+3*p10c2)+amuV**2*(b00**2*p10c0+3*b00*p10c1+3* \
        p10c2)+b00**2*p10c2*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2))+(-8) \
        *a1**3*(amuV**2*p10c0+2*amuV*bmuV*(b00*p10c0+p10c1)+(b00*p10c1+ \
        p10c2)*(bmuV**2+(-1)*sigmaMuV**2+(-1)*sigmaV**2)))
    P_45[1,:] = 2*(P_C_45)*sigmaMuV**2*((-16)*a1**4*bmuV*p10c0+2*amuV**2* \
        b00**3*b01*p10c2+(-1)*a1*amuV*b00*((-2)*b00*(b00+(-3)*b01* \
        bmuV)*p10c2+3*amuV*b01*(b00*p10c1+4*p10c2))+4*a1**3*(2*b00* \
        bmuV*p10c1+(-1)*b01*bmuV**2*p10c1+2*amuV*(b00*p10c0+(-1)*b01* \
        bmuV*p10c0+p10c1)+2*bmuV*p10c2+b01*p10c1*sigmaMuV**2+b01*p10c1* \
        sigmaV**2)+2*a1**2*(amuV**2*b01*(2*b00*p10c0+3*p10c1)+amuV*((-2) \
        *b00**2*p10c1+4*b00*b01*bmuV*p10c1+(-6)*b00*p10c2+6*b01*bmuV* \
        p10c2)+(-2)*b00*p10c2*(b00*bmuV+b01*((-1)*bmuV**2+sigmaMuV**2+ \
        sigmaV**2))))
    P_45[2,:] = 2*(P_C_45)*sigmaMuV**2*(8*a1**4*p10c0+3*amuV**2*b00**2*b01**2* \
        p10c2+a1**3*(8*amuV*b01*p10c0+(-4)*(b00*p10c1+(-2)*b01*bmuV* \
        p10c1+p10c2))+(-3)*a1*amuV*b01*((-2)*b00*(b00+(-1)*b01*bmuV)* \
        p10c2+amuV*b01*(b00*p10c1+2*p10c2))+2*a1**2*(amuV**2*b01**2* \
        p10c0+2*amuV*b01*((-2)*b00*p10c1+b01*bmuV*p10c1+(-3)*p10c2)+ \
        p10c2*(b00**2+(-4)*b00*b01*bmuV+b01**2*(bmuV**2+(-1)*sigmaMuV**2+( \
        -1)*sigmaV**2))))
    P_45[3,:] = (-2)*(P_C_45)*b01*(2*a1+amuV*b01)*(2*a1**2*p10c1+(-2)*amuV* \
        b00*b01*p10c2+a1*(amuV*b01*p10c1+(-2)*b00*p10c2+2*b01*bmuV* \
        p10c2))*sigmaMuV**2
    P_45[4,:] = (P_C_45)*b01**2*(2*a1+amuV*b01)**2*p10c2*sigmaMuV**2
    P_46[0,:] = (P_C_46)*((-16)*a1**4*bmuV*p11c0+8*a1**3*(amuV*(b00*p11c0+p11c1) \
        +bmuV*(b00*p11c1+p11c2))+(-1)*amuV*b00**4*p11c3+2*a1*b00**2*( \
        amuV*b00*p11c2+6*amuV*p11c3+b00*bmuV*p11c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p11c2+3*p11c3)+amuV*(b00**2*p11c1+3*b00*p11c2+3*p11c3) \
        ))*sigmaMuV
    P_46[1,:] = 2*(P_C_46)*(8*a1**4*p11c0+4*a1**3*(amuV*b01*p11c0+(-1)*b00* \
        p11c1+b01*bmuV*p11c1+(-1)*p11c2)+(-2)*amuV*b00**3*b01*p11c3+2* \
        a1**2*(b00**2*p11c2+(-2)*b00*b01*bmuV*p11c2+(-1)*amuV*b01*(2* \
        b00*p11c1+3*p11c2)+3*b00*p11c3+(-3)*b01*bmuV*p11c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p11c3+3*amuV*b01*(b00*p11c2+4* \
        p11c3)))*sigmaMuV
    P_46[2,:] = 2*(P_C_46)*b01*((-4)*a1**3*p11c1+(-2)*a1**2*(amuV*b01*p11c1+( \
        -2)*b00*p11c2+b01*bmuV*p11c2+(-3)*p11c3)+(-3)*amuV*b00**2*b01* \
        p11c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p11c3+amuV*b01*(b00*p11c2+ \
        2*p11c3)))*sigmaMuV
    P_46[3,:] = 2*(P_C_46)*b01**2*(2*a1**2*p11c2+(-2)*amuV*b00*b01*p11c3+a1*( \
        amuV*b01*p11c2+(-3)*b00*p11c3+b01*bmuV*p11c3))*sigmaMuV
    P_46[4,:] = (-1)*(P_C_46)*b01**3*(2*a1+amuV*b01)*p11c3*sigmaMuV
    P_47[0,:] = (P_C_47)*((-16)*a1**4*bmuV*p12c0+8*a1**3*(amuV*(b00*p12c0+p12c1) \
        +bmuV*(b00*p12c1+p12c2))+(-1)*amuV*b00**4*p12c3+2*a1*b00**2*( \
        amuV*b00*p12c2+6*amuV*p12c3+b00*bmuV*p12c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p12c2+3*p12c3)+amuV*(b00**2*p12c1+3*b00*p12c2+3*p12c3) \
        ))*sigmaMuV
    P_47[1,:] = 2*(P_C_47)*(8*a1**4*p12c0+4*a1**3*(amuV*b01*p12c0+(-1)*b00* \
        p12c1+b01*bmuV*p12c1+(-1)*p12c2)+(-2)*amuV*b00**3*b01*p12c3+2* \
        a1**2*(b00**2*p12c2+(-2)*b00*b01*bmuV*p12c2+(-1)*amuV*b01*(2* \
        b00*p12c1+3*p12c2)+3*b00*p12c3+(-3)*b01*bmuV*p12c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p12c3+3*amuV*b01*(b00*p12c2+4* \
        p12c3)))*sigmaMuV
    P_47[2,:] = 2*(P_C_47)*b01*((-4)*a1**3*p12c1+(-2)*a1**2*(amuV*b01*p12c1+( \
        -2)*b00*p12c2+b01*bmuV*p12c2+(-3)*p12c3)+(-3)*amuV*b00**2*b01* \
        p12c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p12c3+amuV*b01*(b00*p12c2+ \
        2*p12c3)))*sigmaMuV
    P_47[3,:] = 2*(P_C_47)*b01**2*(2*a1**2*p12c2+(-2)*amuV*b00*b01*p12c3+a1*( \
        amuV*b01*p12c2+(-3)*b00*p12c3+b01*bmuV*p12c3))*sigmaMuV
    P_47[4,:] = (-1)*(P_C_47)*b01**3*(2*a1+amuV*b01)*p12c3*sigmaMuV
    P_48[0,:] = (P_C_48)*((-16)*a1**4*bmuV*p13c0+8*a1**3*(amuV*(b00*p13c0+p13c1) \
        +bmuV*(b00*p13c1+p13c2))+(-1)*amuV*b00**4*p13c3+2*a1*b00**2*( \
        amuV*b00*p13c2+6*amuV*p13c3+b00*bmuV*p13c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p13c2+3*p13c3)+amuV*(b00**2*p13c1+3*b00*p13c2+3*p13c3) \
        ))*sigmaMuV
    P_48[1,:] = 2*(P_C_48)*(8*a1**4*p13c0+4*a1**3*(amuV*b01*p13c0+(-1)*b00* \
        p13c1+b01*bmuV*p13c1+(-1)*p13c2)+(-2)*amuV*b00**3*b01*p13c3+2* \
        a1**2*(b00**2*p13c2+(-2)*b00*b01*bmuV*p13c2+(-1)*amuV*b01*(2* \
        b00*p13c1+3*p13c2)+3*b00*p13c3+(-3)*b01*bmuV*p13c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p13c3+3*amuV*b01*(b00*p13c2+4* \
        p13c3)))*sigmaMuV
    P_48[2,:] = 2*(P_C_48)*b01*((-4)*a1**3*p13c1+(-2)*a1**2*(amuV*b01*p13c1+( \
        -2)*b00*p13c2+b01*bmuV*p13c2+(-3)*p13c3)+(-3)*amuV*b00**2*b01* \
        p13c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p13c3+amuV*b01*(b00*p13c2+ \
        2*p13c3)))*sigmaMuV
    P_48[3,:] = 2*(P_C_48)*b01**2*(2*a1**2*p13c2+(-2)*amuV*b00*b01*p13c3+a1*( \
        amuV*b01*p13c2+(-3)*b00*p13c3+b01*bmuV*p13c3))*sigmaMuV
    P_48[4,:] = (-1)*(P_C_48)*b01**3*(2*a1+amuV*b01)*p13c3*sigmaMuV
    P_49[0,:] = (P_C_49)*((-16)*a1**4*bmuV*p14c0+8*a1**3*(amuV*(b00*p14c0+p14c1) \
        +bmuV*(b00*p14c1+p14c2))+(-1)*amuV*b00**4*p14c3+2*a1*b00**2*( \
        amuV*b00*p14c2+6*amuV*p14c3+b00*bmuV*p14c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p14c2+3*p14c3)+amuV*(b00**2*p14c1+3*b00*p14c2+3*p14c3) \
        ))*sigmaMuV
    P_49[1,:] = 2*(P_C_49)*(8*a1**4*p14c0+4*a1**3*(amuV*b01*p14c0+(-1)*b00* \
        p14c1+b01*bmuV*p14c1+(-1)*p14c2)+(-2)*amuV*b00**3*b01*p14c3+2* \
        a1**2*(b00**2*p14c2+(-2)*b00*b01*bmuV*p14c2+(-1)*amuV*b01*(2* \
        b00*p14c1+3*p14c2)+3*b00*p14c3+(-3)*b01*bmuV*p14c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p14c3+3*amuV*b01*(b00*p14c2+4* \
        p14c3)))*sigmaMuV
    P_49[2,:] = 2*(P_C_49)*b01*((-4)*a1**3*p14c1+(-2)*a1**2*(amuV*b01*p14c1+( \
        -2)*b00*p14c2+b01*bmuV*p14c2+(-3)*p14c3)+(-3)*amuV*b00**2*b01* \
        p14c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p14c3+amuV*b01*(b00*p14c2+ \
        2*p14c3)))*sigmaMuV
    P_49[3,:] = 2*(P_C_49)*b01**2*(2*a1**2*p14c2+(-2)*amuV*b00*b01*p14c3+a1*( \
        amuV*b01*p14c2+(-3)*b00*p14c3+b01*bmuV*p14c3))*sigmaMuV
    P_49[4,:] = (-1)*(P_C_49)*b01**3*(2*a1+amuV*b01)*p14c3*sigmaMuV
    P_50[0,:] = (P_C_50)*((-16)*a1**4*bmuV*p15c0+8*a1**3*(amuV*(b00*p15c0+p15c1) \
        +bmuV*(b00*p15c1+p15c2))+(-1)*amuV*b00**4*p15c3+2*a1*b00**2*( \
        amuV*b00*p15c2+6*amuV*p15c3+b00*bmuV*p15c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p15c2+3*p15c3)+amuV*(b00**2*p15c1+3*b00*p15c2+3*p15c3) \
        ))*sigmaMuV
    P_50[1,:] = 2*(P_C_50)*(8*a1**4*p15c0+4*a1**3*(amuV*b01*p15c0+(-1)*b00* \
        p15c1+b01*bmuV*p15c1+(-1)*p15c2)+(-2)*amuV*b00**3*b01*p15c3+2* \
        a1**2*(b00**2*p15c2+(-2)*b00*b01*bmuV*p15c2+(-1)*amuV*b01*(2* \
        b00*p15c1+3*p15c2)+3*b00*p15c3+(-3)*b01*bmuV*p15c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p15c3+3*amuV*b01*(b00*p15c2+4* \
        p15c3)))*sigmaMuV
    P_50[2,:] = 2*(P_C_50)*b01*((-4)*a1**3*p15c1+(-2)*a1**2*(amuV*b01*p15c1+( \
        -2)*b00*p15c2+b01*bmuV*p15c2+(-3)*p15c3)+(-3)*amuV*b00**2*b01* \
        p15c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p15c3+amuV*b01*(b00*p15c2+ \
        2*p15c3)))*sigmaMuV
    P_50[3,:] = 2*(P_C_50)*b01**2*(2*a1**2*p15c2+(-2)*amuV*b00*b01*p15c3+a1*( \
        amuV*b01*p15c2+(-3)*b00*p15c3+b01*bmuV*p15c3))*sigmaMuV
    P_50[4,:] = (-1)*(P_C_50)*b01**3*(2*a1+amuV*b01)*p15c3*sigmaMuV
    P_51[0,:] = (P_C_51)*((-16)*a1**4*bmuV*p16c0+8*a1**3*(amuV*(b00*p16c0+p16c1) \
        +bmuV*(b00*p16c1+p16c2))+(-1)*amuV*b00**4*p16c3+2*a1*b00**2*( \
        amuV*b00*p16c2+6*amuV*p16c3+b00*bmuV*p16c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p16c2+3*p16c3)+amuV*(b00**2*p16c1+3*b00*p16c2+3*p16c3) \
        ))*sigmaMuV
    P_51[1,:] = 2*(P_C_51)*(8*a1**4*p16c0+4*a1**3*(amuV*b01*p16c0+(-1)*b00* \
        p16c1+b01*bmuV*p16c1+(-1)*p16c2)+(-2)*amuV*b00**3*b01*p16c3+2* \
        a1**2*(b00**2*p16c2+(-2)*b00*b01*bmuV*p16c2+(-1)*amuV*b01*(2* \
        b00*p16c1+3*p16c2)+3*b00*p16c3+(-3)*b01*bmuV*p16c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p16c3+3*amuV*b01*(b00*p16c2+4* \
        p16c3)))*sigmaMuV
    P_51[2,:] = 2*(P_C_51)*b01*((-4)*a1**3*p16c1+(-2)*a1**2*(amuV*b01*p16c1+( \
        -2)*b00*p16c2+b01*bmuV*p16c2+(-3)*p16c3)+(-3)*amuV*b00**2*b01* \
        p16c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p16c3+amuV*b01*(b00*p16c2+ \
        2*p16c3)))*sigmaMuV
    P_51[3,:] = 2*(P_C_51)*b01**2*(2*a1**2*p16c2+(-2)*amuV*b00*b01*p16c3+a1*( \
        amuV*b01*p16c2+(-3)*b00*p16c3+b01*bmuV*p16c3))*sigmaMuV
    P_51[4,:] = (-1)*(P_C_51)*b01**3*(2*a1+amuV*b01)*p16c3*sigmaMuV
    P_52[0,:] = (P_C_52)*((-16)*a1**4*bmuV*p17c0+8*a1**3*(amuV*(b00*p17c0+p17c1) \
        +bmuV*(b00*p17c1+p17c2))+(-1)*amuV*b00**4*p17c3+2*a1*b00**2*( \
        amuV*b00*p17c2+6*amuV*p17c3+b00*bmuV*p17c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p17c2+3*p17c3)+amuV*(b00**2*p17c1+3*b00*p17c2+3*p17c3) \
        ))*sigmaMuV
    P_52[1,:] = 2*(P_C_52)*(8*a1**4*p17c0+4*a1**3*(amuV*b01*p17c0+(-1)*b00* \
        p17c1+b01*bmuV*p17c1+(-1)*p17c2)+(-2)*amuV*b00**3*b01*p17c3+2* \
        a1**2*(b00**2*p17c2+(-2)*b00*b01*bmuV*p17c2+(-1)*amuV*b01*(2* \
        b00*p17c1+3*p17c2)+3*b00*p17c3+(-3)*b01*bmuV*p17c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p17c3+3*amuV*b01*(b00*p17c2+4* \
        p17c3)))*sigmaMuV
    P_52[2,:] = 2*(P_C_52)*b01*((-4)*a1**3*p17c1+(-2)*a1**2*(amuV*b01*p17c1+( \
        -2)*b00*p17c2+b01*bmuV*p17c2+(-3)*p17c3)+(-3)*amuV*b00**2*b01* \
        p17c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p17c3+amuV*b01*(b00*p17c2+ \
        2*p17c3)))*sigmaMuV
    P_52[3,:] = 2*(P_C_52)*b01**2*(2*a1**2*p17c2+(-2)*amuV*b00*b01*p17c3+a1*( \
        amuV*b01*p17c2+(-3)*b00*p17c3+b01*bmuV*p17c3))*sigmaMuV
    P_52[4,:] = (-1)*(P_C_52)*b01**3*(2*a1+amuV*b01)*p17c3*sigmaMuV
    P_53[0,:] = (P_C_53)*((-16)*a1**4*bmuV*p18c0+8*a1**3*(amuV*(b00*p18c0+p18c1) \
        +bmuV*(b00*p18c1+p18c2))+(-1)*amuV*b00**4*p18c3+2*a1*b00**2*( \
        amuV*b00*p18c2+6*amuV*p18c3+b00*bmuV*p18c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p18c2+3*p18c3)+amuV*(b00**2*p18c1+3*b00*p18c2+3*p18c3) \
        ))*sigmaMuV
    P_53[1,:] = 2*(P_C_53)*(8*a1**4*p18c0+4*a1**3*(amuV*b01*p18c0+(-1)*b00* \
        p18c1+b01*bmuV*p18c1+(-1)*p18c2)+(-2)*amuV*b00**3*b01*p18c3+2* \
        a1**2*(b00**2*p18c2+(-2)*b00*b01*bmuV*p18c2+(-1)*amuV*b01*(2* \
        b00*p18c1+3*p18c2)+3*b00*p18c3+(-3)*b01*bmuV*p18c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p18c3+3*amuV*b01*(b00*p18c2+4* \
        p18c3)))*sigmaMuV
    P_53[2,:] = 2*(P_C_53)*b01*((-4)*a1**3*p18c1+(-2)*a1**2*(amuV*b01*p18c1+( \
        -2)*b00*p18c2+b01*bmuV*p18c2+(-3)*p18c3)+(-3)*amuV*b00**2*b01* \
        p18c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p18c3+amuV*b01*(b00*p18c2+ \
        2*p18c3)))*sigmaMuV
    P_53[3,:] = 2*(P_C_53)*b01**2*(2*a1**2*p18c2+(-2)*amuV*b00*b01*p18c3+a1*( \
        amuV*b01*p18c2+(-3)*b00*p18c3+b01*bmuV*p18c3))*sigmaMuV
    P_53[4,:] = (-1)*(P_C_53)*b01**3*(2*a1+amuV*b01)*p18c3*sigmaMuV
    P_54[0,:] = (P_C_54)*((-16)*a1**4*bmuV*p19c0+8*a1**3*(amuV*(b00*p19c0+p19c1) \
        +bmuV*(b00*p19c1+p19c2))+(-1)*amuV*b00**4*p19c3+2*a1*b00**2*( \
        amuV*b00*p19c2+6*amuV*p19c3+b00*bmuV*p19c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p19c2+3*p19c3)+amuV*(b00**2*p19c1+3*b00*p19c2+3*p19c3) \
        ))*sigmaMuV
    P_54[1,:] = 2*(P_C_54)*(8*a1**4*p19c0+4*a1**3*(amuV*b01*p19c0+(-1)*b00* \
        p19c1+b01*bmuV*p19c1+(-1)*p19c2)+(-2)*amuV*b00**3*b01*p19c3+2* \
        a1**2*(b00**2*p19c2+(-2)*b00*b01*bmuV*p19c2+(-1)*amuV*b01*(2* \
        b00*p19c1+3*p19c2)+3*b00*p19c3+(-3)*b01*bmuV*p19c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p19c3+3*amuV*b01*(b00*p19c2+4* \
        p19c3)))*sigmaMuV
    P_54[2,:] = 2*(P_C_54)*b01*((-4)*a1**3*p19c1+(-2)*a1**2*(amuV*b01*p19c1+( \
        -2)*b00*p19c2+b01*bmuV*p19c2+(-3)*p19c3)+(-3)*amuV*b00**2*b01* \
        p19c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p19c3+amuV*b01*(b00*p19c2+ \
        2*p19c3)))*sigmaMuV
    P_54[3,:] = 2*(P_C_54)*b01**2*(2*a1**2*p19c2+(-2)*amuV*b00*b01*p19c3+a1*( \
        amuV*b01*p19c2+(-3)*b00*p19c3+b01*bmuV*p19c3))*sigmaMuV
    P_54[4,:] = (-1)*(P_C_54)*b01**3*(2*a1+amuV*b01)*p19c3*sigmaMuV
    P_55[0,:] = (P_C_55)*((-16)*a1**4*bmuV*p20c0+8*a1**3*(amuV*(b00*p20c0+p20c1) \
        +bmuV*(b00*p20c1+p20c2))+(-1)*amuV*b00**4*p20c3+2*a1*b00**2*( \
        amuV*b00*p20c2+6*amuV*p20c3+b00*bmuV*p20c3)+(-4)*a1**2*(b00* \
        bmuV*(b00*p20c2+3*p20c3)+amuV*(b00**2*p20c1+3*b00*p20c2+3*p20c3) \
        ))*sigmaMuV
    P_55[1,:] = 2*(P_C_55)*(8*a1**4*p20c0+4*a1**3*(amuV*b01*p20c0+(-1)*b00* \
        p20c1+b01*bmuV*p20c1+(-1)*p20c2)+(-2)*amuV*b00**3*b01*p20c3+2* \
        a1**2*(b00**2*p20c2+(-2)*b00*b01*bmuV*p20c2+(-1)*amuV*b01*(2* \
        b00*p20c1+3*p20c2)+3*b00*p20c3+(-3)*b01*bmuV*p20c3)+a1*b00*(( \
        -1)*b00*(b00+(-3)*b01*bmuV)*p20c3+3*amuV*b01*(b00*p20c2+4* \
        p20c3)))*sigmaMuV
    P_55[2,:] = 2*(P_C_55)*b01*((-4)*a1**3*p20c1+(-2)*a1**2*(amuV*b01*p20c1+( \
        -2)*b00*p20c2+b01*bmuV*p20c2+(-3)*p20c3)+(-3)*amuV*b00**2*b01* \
        p20c3+3*a1*(b00*((-1)*b00+b01*bmuV)*p20c3+amuV*b01*(b00*p20c2+ \
        2*p20c3)))*sigmaMuV
    P_55[3,:] = 2*(P_C_55)*b01**2*(2*a1**2*p20c2+(-2)*amuV*b00*b01*p20c3+a1*( \
        amuV*b01*p20c2+(-3)*b00*p20c3+b01*bmuV*p20c3))*sigmaMuV
    P_55[4,:] = (-1)*(P_C_55)*b01**3*(2*a1+amuV*b01)*p20c3*sigmaMuV
    P_56[0,:] = (P_C_56)*(16*a1**4*p21c0+(-8)*a1**3*(b00*p21c1+p21c2)+b00**4* \
        p21c4+4*a1**2*(b00**2*p21c2+3*b00*p21c3+3*p21c4)+(-2)*a1* \
        b00**2*(b00*p21c3+6*p21c4))
    P_56[1,:] = 2*(P_C_56)*b01*((-4)*a1**3*p21c1+a1**2*(4*b00*p21c2+6*p21c3)+ \
        2*b00**3*p21c4+(-3)*a1*b00*(b00*p21c3+4*p21c4))
    P_56[2,:] = 2*(P_C_56)*b01**2*(2*a1**2*p21c2+3*b00**2*p21c4+(-3)*a1*(b00* \
        p21c3+2*p21c4))
    P_56[3,:] = (-2)*(P_C_56)*b01**3*(a1*p21c3+(-2)*b00*p21c4)
    P_56[4,:] = (P_C_56)*b01**4*p21c4
    P_57[0,:] = (P_C_57)*(16*a1**4*p22c0+(-8)*a1**3*(b00*p22c1+p22c2)+b00**4* \
        p22c4+4*a1**2*(b00**2*p22c2+3*b00*p22c3+3*p22c4)+(-2)*a1* \
        b00**2*(b00*p22c3+6*p22c4))
    P_57[1,:] = 2*(P_C_57)*b01*((-4)*a1**3*p22c1+a1**2*(4*b00*p22c2+6*p22c3)+ \
        2*b00**3*p22c4+(-3)*a1*b00*(b00*p22c3+4*p22c4))
    P_57[2,:] = 2*(P_C_57)*b01**2*(2*a1**2*p22c2+3*b00**2*p22c4+(-3)*a1*(b00* \
        p22c3+2*p22c4))
    P_57[3,:] = (-2)*(P_C_57)*b01**3*(a1*p22c3+(-2)*b00*p22c4)
    P_57[4,:] = (P_C_57)*b01**4*p22c4
    P_58[0,:] = (P_C_58)*(16*a1**4*p23c0+(-8)*a1**3*(b00*p23c1+p23c2)+b00**4* \
        p23c4+4*a1**2*(b00**2*p23c2+3*b00*p23c3+3*p23c4)+(-2)*a1* \
        b00**2*(b00*p23c3+6*p23c4))
    P_58[1,:] = 2*(P_C_58)*b01*((-4)*a1**3*p23c1+a1**2*(4*b00*p23c2+6*p23c3)+ \
        2*b00**3*p23c4+(-3)*a1*b00*(b00*p23c3+4*p23c4))
    P_58[2,:] = 2*(P_C_58)*b01**2*(2*a1**2*p23c2+3*b00**2*p23c4+(-3)*a1*(b00* \
        p23c3+2*p23c4))
    P_58[3,:] = (-2)*(P_C_58)*b01**3*(a1*p23c3+(-2)*b00*p23c4)
    P_58[4,:] = (P_C_58)*b01**4*p23c4
    P_59[0,:] = (P_C_59)*(16*a1**4*p24c0+(-8)*a1**3*(b00*p24c1+p24c2)+b00**4* \
        p24c4+4*a1**2*(b00**2*p24c2+3*b00*p24c3+3*p24c4)+(-2)*a1* \
        b00**2*(b00*p24c3+6*p24c4))
    P_59[1,:] = 2*(P_C_59)*b01*((-4)*a1**3*p24c1+a1**2*(4*b00*p24c2+6*p24c3)+ \
        2*b00**3*p24c4+(-3)*a1*b00*(b00*p24c3+4*p24c4))
    P_59[2,:] = 2*(P_C_59)*b01**2*(2*a1**2*p24c2+3*b00**2*p24c4+(-3)*a1*(b00* \
        p24c3+2*p24c4))
    P_59[3,:] = (-2)*(P_C_59)*b01**3*(a1*p24c3+(-2)*b00*p24c4)
    P_59[4,:] = (P_C_59)*b01**4*p24c4
    P_60[0,:] = (P_C_60)*(16*a1**4*p25c0+(-8)*a1**3*(b00*p25c1+p25c2)+b00**4* \
        p25c4+4*a1**2*(b00**2*p25c2+3*b00*p25c3+3*p25c4)+(-2)*a1* \
        b00**2*(b00*p25c3+6*p25c4))
    P_60[1,:] = 2*(P_C_60)*b01*((-4)*a1**3*p25c1+a1**2*(4*b00*p25c2+6*p25c3)+ \
        2*b00**3*p25c4+(-3)*a1*b00*(b00*p25c3+4*p25c4))
    P_60[2,:] = 2*(P_C_60)*b01**2*(2*a1**2*p25c2+3*b00**2*p25c4+(-3)*a1*(b00* \
        p25c3+2*p25c4))
    P_60[3,:] = (-2)*(P_C_60)*b01**3*(a1*p25c3+(-2)*b00*p25c4)
    P_60[4,:] = (P_C_60)*b01**4*p25c4
    P_61[0,:] = (P_C_61)*(16*a1**4*p26c0+(-8)*a1**3*(b00*p26c1+p26c2)+b00**4* \
        p26c4+4*a1**2*(b00**2*p26c2+3*b00*p26c3+3*p26c4)+(-2)*a1* \
        b00**2*(b00*p26c3+6*p26c4))
    P_61[1,:] = 2*(P_C_61)*b01*((-4)*a1**3*p26c1+a1**2*(4*b00*p26c2+6*p26c3)+ \
        2*b00**3*p26c4+(-3)*a1*b00*(b00*p26c3+4*p26c4))
    P_61[2,:] = 2*(P_C_61)*b01**2*(2*a1**2*p26c2+3*b00**2*p26c4+(-3)*a1*(b00* \
        p26c3+2*p26c4))
    P_61[3,:] = (-2)*(P_C_61)*b01**3*(a1*p26c3+(-2)*b00*p26c4)
    P_61[4,:] = (P_C_61)*b01**4*p26c4
    P_62[0,:] = (P_C_62)*(16*a1**4*p27c0+(-8)*a1**3*(b00*p27c1+p27c2)+b00**4* \
        p27c4+4*a1**2*(b00**2*p27c2+3*b00*p27c3+3*p27c4)+(-2)*a1* \
        b00**2*(b00*p27c3+6*p27c4))
    P_62[1,:] = 2*(P_C_62)*b01*((-4)*a1**3*p27c1+a1**2*(4*b00*p27c2+6*p27c3)+ \
        2*b00**3*p27c4+(-3)*a1*b00*(b00*p27c3+4*p27c4))
    P_62[2,:] = 2*(P_C_62)*b01**2*(2*a1**2*p27c2+3*b00**2*p27c4+(-3)*a1*(b00* \
        p27c3+2*p27c4))
    P_62[3,:] = (-2)*(P_C_62)*b01**3*(a1*p27c3+(-2)*b00*p27c4)
    P_62[4,:] = (P_C_62)*b01**4*p27c4
    P_63[0,:] = (P_C_63)*(16*a1**4*p28c0+(-8)*a1**3*(b00*p28c1+p28c2)+b00**4* \
        p28c4+4*a1**2*(b00**2*p28c2+3*b00*p28c3+3*p28c4)+(-2)*a1* \
        b00**2*(b00*p28c3+6*p28c4))
    P_63[1,:] = 2*(P_C_63)*b01*((-4)*a1**3*p28c1+a1**2*(4*b00*p28c2+6*p28c3)+ \
        2*b00**3*p28c4+(-3)*a1*b00*(b00*p28c3+4*p28c4))
    P_63[2,:] = 2*(P_C_63)*b01**2*(2*a1**2*p28c2+3*b00**2*p28c4+(-3)*a1*(b00* \
        p28c3+2*p28c4))
    P_63[3,:] = (-2)*(P_C_63)*b01**3*(a1*p28c3+(-2)*b00*p28c4)
    P_63[4,:] = (P_C_63)*b01**4*p28c4
    P_64[0,:] = (P_C_64)*(16*a1**4*p29c0+(-8)*a1**3*(b00*p29c1+p29c2)+b00**4* \
        p29c4+4*a1**2*(b00**2*p29c2+3*b00*p29c3+3*p29c4)+(-2)*a1* \
        b00**2*(b00*p29c3+6*p29c4))
    P_64[1,:] = 2*(P_C_64)*b01*((-4)*a1**3*p29c1+a1**2*(4*b00*p29c2+6*p29c3)+ \
        2*b00**3*p29c4+(-3)*a1*b00*(b00*p29c3+4*p29c4))
    P_64[2,:] = 2*(P_C_64)*b01**2*(2*a1**2*p29c2+3*b00**2*p29c4+(-3)*a1*(b00* \
        p29c3+2*p29c4))
    P_64[3,:] = (-2)*(P_C_64)*b01**3*(a1*p29c3+(-2)*b00*p29c4)
    P_64[4,:] = (P_C_64)*b01**4*p29c4
    P_65[0,:] = (P_C_65)*(16*a1**4*p30c0+(-8)*a1**3*(b00*p30c1+p30c2)+b00**4* \
        p30c4+4*a1**2*(b00**2*p30c2+3*b00*p30c3+3*p30c4)+(-2)*a1* \
        b00**2*(b00*p30c3+6*p30c4))
    P_65[1,:] = 2*(P_C_65)*b01*((-4)*a1**3*p30c1+a1**2*(4*b00*p30c2+6*p30c3)+ \
        2*b00**3*p30c4+(-3)*a1*b00*(b00*p30c3+4*p30c4))
    P_65[2,:] = 2*(P_C_65)*b01**2*(2*a1**2*p30c2+3*b00**2*p30c4+(-3)*a1*(b00* \
        p30c3+2*p30c4))
    P_65[3,:] = (-2)*(P_C_65)*b01**3*(a1*p30c3+(-2)*b00*p30c4)
    P_65[4,:] = (P_C_65)*b01**4*p30c4
    P_66[0,:] = (P_C_66)*(16*a1**4*p31c0+(-8)*a1**3*(b00*p31c1+p31c2)+b00**4* \
        p31c4+4*a1**2*(b00**2*p31c2+3*b00*p31c3+3*p31c4)+(-2)*a1* \
        b00**2*(b00*p31c3+6*p31c4))
    P_66[1,:] = 2*(P_C_66)*b01*((-4)*a1**3*p31c1+a1**2*(4*b00*p31c2+6*p31c3)+ \
        2*b00**3*p31c4+(-3)*a1*b00*(b00*p31c3+4*p31c4))
    P_66[2,:] = 2*(P_C_66)*b01**2*(2*a1**2*p31c2+3*b00**2*p31c4+(-3)*a1*(b00* \
        p31c3+2*p31c4))
    P_66[3,:] = (-2)*(P_C_66)*b01**3*(a1*p31c3+(-2)*b00*p31c4)
    P_66[4,:] = (P_C_66)*b01**4*p31c4
    P_67[0,:] = (P_C_67)*(16*a1**4*p32c0+(-8)*a1**3*(b00*p32c1+p32c2)+b00**4* \
        p32c4+4*a1**2*(b00**2*p32c2+3*b00*p32c3+3*p32c4)+(-2)*a1* \
        b00**2*(b00*p32c3+6*p32c4))
    P_67[1,:] = 2*(P_C_67)*b01*((-4)*a1**3*p32c1+a1**2*(4*b00*p32c2+6*p32c3)+ \
        2*b00**3*p32c4+(-3)*a1*b00*(b00*p32c3+4*p32c4))
    P_67[2,:] = 2*(P_C_67)*b01**2*(2*a1**2*p32c2+3*b00**2*p32c4+(-3)*a1*(b00* \
        p32c3+2*p32c4))
    P_67[3,:] = (-2)*(P_C_67)*b01**3*(a1*p32c3+(-2)*b00*p32c4)
    P_67[4,:] = (P_C_67)*b01**4*p32c4
    P_68[0,:] = (P_C_68)*(16*a1**4*p33c0+(-8)*a1**3*(b00*p33c1+p33c2)+b00**4* \
        p33c4+4*a1**2*(b00**2*p33c2+3*b00*p33c3+3*p33c4)+(-2)*a1* \
        b00**2*(b00*p33c3+6*p33c4))
    P_68[1,:] = 2*(P_C_68)*b01*((-4)*a1**3*p33c1+a1**2*(4*b00*p33c2+6*p33c3)+ \
        2*b00**3*p33c4+(-3)*a1*b00*(b00*p33c3+4*p33c4))
    P_68[2,:] = 2*(P_C_68)*b01**2*(2*a1**2*p33c2+3*b00**2*p33c4+(-3)*a1*(b00* \
        p33c3+2*p33c4))
    P_68[3,:] = (-2)*(P_C_68)*b01**3*(a1*p33c3+(-2)*b00*p33c4)
    P_68[4,:] = (P_C_68)*b01**4*p33c4
    P_69[0,:] = (P_C_69)*(16*a1**4*p34c0+(-8)*a1**3*(b00*p34c1+p34c2)+b00**4* \
        p34c4+4*a1**2*(b00**2*p34c2+3*b00*p34c3+3*p34c4)+(-2)*a1* \
        b00**2*(b00*p34c3+6*p34c4))
    P_69[1,:] = 2*(P_C_69)*b01*((-4)*a1**3*p34c1+a1**2*(4*b00*p34c2+6*p34c3)+ \
        2*b00**3*p34c4+(-3)*a1*b00*(b00*p34c3+4*p34c4))
    P_69[2,:] = 2*(P_C_69)*b01**2*(2*a1**2*p34c2+3*b00**2*p34c4+(-3)*a1*(b00* \
        p34c3+2*p34c4))
    P_69[3,:] = (-2)*(P_C_69)*b01**3*(a1*p34c3+(-2)*b00*p34c4)
    P_69[4,:] = (P_C_69)*b01**4*p34c4
    P_70[0,:] = (P_C_70)*(16*a1**4*p35c0+(-8)*a1**3*(b00*p35c1+p35c2)+b00**4* \
        p35c4+4*a1**2*(b00**2*p35c2+3*b00*p35c3+3*p35c4)+(-2)*a1* \
        b00**2*(b00*p35c3+6*p35c4))
    P_70[1,:] = 2*(P_C_70)*b01*((-4)*a1**3*p35c1+a1**2*(4*b00*p35c2+6*p35c3)+ \
        2*b00**3*p35c4+(-3)*a1*b00*(b00*p35c3+4*p35c4))
    P_70[2,:] = 2*(P_C_70)*b01**2*(2*a1**2*p35c2+3*b00**2*p35c4+(-3)*a1*(b00* \
        p35c3+2*p35c4))
    P_70[3,:] = (-2)*(P_C_70)*b01**3*(a1*p35c3+(-2)*b00*p35c4)
    P_70[4,:] = (P_C_70)*b01**4*p35c4
    P_output_3 = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13, \
        P_14, P_15, P_16, P_17, P_18, P_19, P_20, P_21, P_22, P_23, P_24, P_25, \
        P_26, P_27, P_28, P_29, P_30, P_31, P_32, P_33, P_34, P_35, P_36, P_37, \
        P_38, P_39, P_40, P_41, P_42, P_43, P_44, P_45, P_46, P_47, P_48, P_49, \
        P_50, P_51, P_52, P_53, P_54, P_55, P_56, P_57, P_58, P_59, P_60, P_61, \
        P_62, P_63, P_64, P_65, P_66, P_67, P_68, P_69, P_70))
    
    ##
    ###########################################################################
    ########################## CDF FOURTH INTEGRATION #########################
    ###########################################################################
    
    # PC Coefficients from Third Integration;
    # P_output_3 = PC_Coeffs_Simpler_Third_Integration_new_function(muY, sigmaMuY, ...
    #    sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT,  ...
    #   amuV, bmuV, sigmaMuV, sigmaV);
    
    p1c0 = P_output_3[0, :]
    p2c0 = P_output_3[1, :]
    p2c1 = P_output_3[2, :]
    p3c0 = P_output_3[3, :]
    p3c1 = P_output_3[4, :]
    p4c0 = P_output_3[5, :]
    p4c1 = P_output_3[6, :]
    p5c0 = P_output_3[7, :]
    p5c1 = P_output_3[8, :]
    p6c0 = P_output_3[9, :]
    p6c1 = P_output_3[10, :]
    p6c2 = P_output_3[11, :]
    p7c0 = P_output_3[12, :]
    p7c1 = P_output_3[13, :]
    p7c2 = P_output_3[14, :]
    p8c0 = P_output_3[15, :]
    p8c1 = P_output_3[16, :]
    p8c2 = P_output_3[17, :]
    p9c0 = P_output_3[18, :]
    p9c1 = P_output_3[19, :]
    p9c2 = P_output_3[20, :]
    p10c0 = P_output_3[21, :]
    p10c1 = P_output_3[22, :]
    p10c2 = P_output_3[23, :]
    p11c0 = P_output_3[24, :]
    p11c1 = P_output_3[25, :]
    p11c2 = P_output_3[26, :]
    p12c0 = P_output_3[27, :]
    p12c1 = P_output_3[28, :]
    p12c2 = P_output_3[29, :]
    p13c0 = P_output_3[30, :]
    p13c1 = P_output_3[31, :]
    p13c2 = P_output_3[32, :]
    p14c0 = P_output_3[33, :]
    p14c1 = P_output_3[34, :]
    p14c2 = P_output_3[35, :]
    p15c0 = P_output_3[36, :]
    p15c1 = P_output_3[37, :]
    p15c2 = P_output_3[38, :]
    p16c0 = P_output_3[39, :]
    p16c1 = P_output_3[40, :]
    p16c2 = P_output_3[41, :]
    p16c3 = P_output_3[42, :]
    p17c0 = P_output_3[43, :]
    p17c1 = P_output_3[44, :]
    p17c2 = P_output_3[45, :]
    p17c3 = P_output_3[46, :]
    p18c0 = P_output_3[47, :]
    p18c1 = P_output_3[48, :]
    p18c2 = P_output_3[49, :]
    p18c3 = P_output_3[50, :]
    p19c0 = P_output_3[51, :]
    p19c1 = P_output_3[52, :]
    p19c2 = P_output_3[53, :]
    p19c3 = P_output_3[54, :]
    p20c0 = P_output_3[55, :]
    p20c1 = P_output_3[56, :]
    p20c2 = P_output_3[57, :]
    p20c3 = P_output_3[58, :]
    p21c0 = P_output_3[59, :]
    p21c1 = P_output_3[60, :]
    p21c2 = P_output_3[61, :]
    p21c3 = P_output_3[62, :]
    p22c0 = P_output_3[63, :]
    p22c1 = P_output_3[64, :]
    p22c2 = P_output_3[65, :]
    p22c3 = P_output_3[66, :]
    p23c0 = P_output_3[67, :]
    p23c1 = P_output_3[68, :]
    p23c2 = P_output_3[69, :]
    p23c3 = P_output_3[70, :]
    p24c0 = P_output_3[71, :]
    p24c1 = P_output_3[72, :]
    p24c2 = P_output_3[73, :]
    p24c3 = P_output_3[74, :]
    p25c0 = P_output_3[75, :]
    p25c1 = P_output_3[76, :]
    p25c2 = P_output_3[77, :]
    p25c3 = P_output_3[78, :]
    p26c0 = P_output_3[79, :]
    p26c1 = P_output_3[80, :]
    p26c2 = P_output_3[81, :]
    p26c3 = P_output_3[82, :]
    p27c0 = P_output_3[83, :]
    p27c1 = P_output_3[84, :]
    p27c2 = P_output_3[85, :]
    p27c3 = P_output_3[86, :]
    p28c0 = P_output_3[87, :]
    p28c1 = P_output_3[88, :]
    p28c2 = P_output_3[89, :]
    p28c3 = P_output_3[90, :]
    p29c0 = P_output_3[91, :]
    p29c1 = P_output_3[92, :]
    p29c2 = P_output_3[93, :]
    p29c3 = P_output_3[94, :]
    p30c0 = P_output_3[95, :]
    p30c1 = P_output_3[96, :]
    p30c2 = P_output_3[97, :]
    p30c3 = P_output_3[98, :]
    p31c0 = P_output_3[99, :]
    p31c1 = P_output_3[100, :]
    p31c2 = P_output_3[101, :]
    p31c3 = P_output_3[102, :]
    p32c0 = P_output_3[103, :]
    p32c1 = P_output_3[104, :]
    p32c2 = P_output_3[105, :]
    p32c3 = P_output_3[106, :]
    p33c0 = P_output_3[107, :]
    p33c1 = P_output_3[108, :]
    p33c2 = P_output_3[109, :]
    p33c3 = P_output_3[110, :]
    p34c0 = P_output_3[111, :]
    p34c1 = P_output_3[112, :]
    p34c2 = P_output_3[113, :]
    p34c3 = P_output_3[114, :]
    p35c0 = P_output_3[115, :]
    p35c1 = P_output_3[116, :]
    p35c2 = P_output_3[117, :]
    p35c3 = P_output_3[118, :]
    p36c0 = P_output_3[119, :]
    p36c1 = P_output_3[120, :]
    p36c2 = P_output_3[121, :]
    p36c3 = P_output_3[122, :]
    p36c4 = P_output_3[123, :]
    p37c0 = P_output_3[124, :]
    p37c1 = P_output_3[125, :]
    p37c2 = P_output_3[126, :]
    p37c3 = P_output_3[127, :]
    p37c4 = P_output_3[128, :]
    p38c0 = P_output_3[129, :]
    p38c1 = P_output_3[130, :]
    p38c2 = P_output_3[131, :]
    p38c3 = P_output_3[132, :]
    p38c4 = P_output_3[133, :]
    p39c0 = P_output_3[134, :]
    p39c1 = P_output_3[135, :]
    p39c2 = P_output_3[136, :]
    p39c3 = P_output_3[137, :]
    p39c4 = P_output_3[138, :]
    p40c0 = P_output_3[139, :]
    p40c1 = P_output_3[140, :]
    p40c2 = P_output_3[141, :]
    p40c3 = P_output_3[142, :]
    p40c4 = P_output_3[143, :]
    p41c0 = P_output_3[144, :]
    p41c1 = P_output_3[145, :]
    p41c2 = P_output_3[146, :]
    p41c3 = P_output_3[147, :]
    p41c4 = P_output_3[148, :]
    p42c0 = P_output_3[149, :]
    p42c1 = P_output_3[150, :]
    p42c2 = P_output_3[151, :]
    p42c3 = P_output_3[152, :]
    p42c4 = P_output_3[153, :]
    p43c0 = P_output_3[154, :]
    p43c1 = P_output_3[155, :]
    p43c2 = P_output_3[156, :]
    p43c3 = P_output_3[157, :]
    p43c4 = P_output_3[158, :]
    p44c0 = P_output_3[159, :]
    p44c1 = P_output_3[160, :]
    p44c2 = P_output_3[161, :]
    p44c3 = P_output_3[162, :]
    p44c4 = P_output_3[163, :]
    p45c0 = P_output_3[164, :]
    p45c1 = P_output_3[165, :]
    p45c2 = P_output_3[166, :]
    p45c3 = P_output_3[167, :]
    p45c4 = P_output_3[168, :]
    p46c0 = P_output_3[169, :]
    p46c1 = P_output_3[170, :]
    p46c2 = P_output_3[171, :]
    p46c3 = P_output_3[172, :]
    p46c4 = P_output_3[173, :]
    p47c0 = P_output_3[174, :]
    p47c1 = P_output_3[175, :]
    p47c2 = P_output_3[176, :]
    p47c3 = P_output_3[177, :]
    p47c4 = P_output_3[178, :]
    p48c0 = P_output_3[179, :]
    p48c1 = P_output_3[180, :]
    p48c2 = P_output_3[181, :]
    p48c3 = P_output_3[182, :]
    p48c4 = P_output_3[183, :]
    p49c0 = P_output_3[184, :]
    p49c1 = P_output_3[185, :]
    p49c2 = P_output_3[186, :]
    p49c3 = P_output_3[187, :]
    p49c4 = P_output_3[188, :]
    p50c0 = P_output_3[189, :]
    p50c1 = P_output_3[190, :]
    p50c2 = P_output_3[191, :]
    p50c3 = P_output_3[192, :]
    p50c4 = P_output_3[193, :]
    p51c0 = P_output_3[194, :]
    p51c1 = P_output_3[195, :]
    p51c2 = P_output_3[196, :]
    p51c3 = P_output_3[197, :]
    p51c4 = P_output_3[198, :]
    p52c0 = P_output_3[199, :]
    p52c1 = P_output_3[200, :]
    p52c2 = P_output_3[201, :]
    p52c3 = P_output_3[202, :]
    p52c4 = P_output_3[203, :]
    p53c0 = P_output_3[204, :]
    p53c1 = P_output_3[205, :]
    p53c2 = P_output_3[206, :]
    p53c3 = P_output_3[207, :]
    p53c4 = P_output_3[208, :]
    p54c0 = P_output_3[209, :]
    p54c1 = P_output_3[210, :]
    p54c2 = P_output_3[211, :]
    p54c3 = P_output_3[212, :]
    p54c4 = P_output_3[213, :]
    p55c0 = P_output_3[214, :]
    p55c1 = P_output_3[215, :]
    p55c2 = P_output_3[216, :]
    p55c3 = P_output_3[217, :]
    p55c4 = P_output_3[218, :]
    p56c0 = P_output_3[219, :]
    p56c1 = P_output_3[220, :]
    p56c2 = P_output_3[221, :]
    p56c3 = P_output_3[222, :]
    p56c4 = P_output_3[223, :]
    p57c0 = P_output_3[224, :]
    p57c1 = P_output_3[225, :]
    p57c2 = P_output_3[226, :]
    p57c3 = P_output_3[227, :]
    p57c4 = P_output_3[228, :]
    p58c0 = P_output_3[229, :]
    p58c1 = P_output_3[230, :]
    p58c2 = P_output_3[231, :]
    p58c3 = P_output_3[232, :]
    p58c4 = P_output_3[233, :]
    p59c0 = P_output_3[234, :]
    p59c1 = P_output_3[235, :]
    p59c2 = P_output_3[236, :]
    p59c3 = P_output_3[237, :]
    p59c4 = P_output_3[238, :]
    p60c0 = P_output_3[239, :]
    p60c1 = P_output_3[240, :]
    p60c2 = P_output_3[241, :]
    p60c3 = P_output_3[242, :]
    p60c4 = P_output_3[243, :]
    p61c0 = P_output_3[244, :]
    p61c1 = P_output_3[245, :]
    p61c2 = P_output_3[246, :]
    p61c3 = P_output_3[247, :]
    p61c4 = P_output_3[248, :]
    p62c0 = P_output_3[249, :]
    p62c1 = P_output_3[250, :]
    p62c2 = P_output_3[251, :]
    p62c3 = P_output_3[252, :]
    p62c4 = P_output_3[253, :]
    p63c0 = P_output_3[254, :]
    p63c1 = P_output_3[255, :]
    p63c2 = P_output_3[256, :]
    p63c3 = P_output_3[257, :]
    p63c4 = P_output_3[258, :]
    p64c0 = P_output_3[259, :]
    p64c1 = P_output_3[260, :]
    p64c2 = P_output_3[261, :]
    p64c3 = P_output_3[262, :]
    p64c4 = P_output_3[263, :]
    p65c0 = P_output_3[264, :]
    p65c1 = P_output_3[265, :]
    p65c2 = P_output_3[266, :]
    p65c3 = P_output_3[267, :]
    p65c4 = P_output_3[268, :]
    p66c0 = P_output_3[269, :]
    p66c1 = P_output_3[270, :]
    p66c2 = P_output_3[271, :]
    p66c3 = P_output_3[272, :]
    p66c4 = P_output_3[273, :]
    p67c0 = P_output_3[274, :]
    p67c1 = P_output_3[275, :]
    p67c2 = P_output_3[276, :]
    p67c3 = P_output_3[277, :]
    p67c4 = P_output_3[278, :]
    p68c0 = P_output_3[279, :]
    p68c1 = P_output_3[280, :]
    p68c2 = P_output_3[281, :]
    p68c3 = P_output_3[282, :]
    p68c4 = P_output_3[283, :]
    p69c0 = P_output_3[284, :]
    p69c1 = P_output_3[285, :]
    p69c2 = P_output_3[286, :]
    p69c3 = P_output_3[287, :]
    p69c4 = P_output_3[288, :]
    p70c0 = P_output_3[289, :]
    p70c1 = P_output_3[290, :]
    p70c2 = P_output_3[291, :]
    p70c3 = P_output_3[292, :]
    p70c4 = P_output_3[293, :]
    v  =  v.T
    a = (-1/2)*(sigmaMuV**2+sigmaV**2+amuV**2*(sigmaMuT**2+sigmaT**2+amuT**2* \
        (sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)))**(-1)
    b = (bmuV+amuV*(bmuT+amuT*(bmuZ+amuZ*muY)))*(sigmaMuV**2+sigmaV**2+ \
        amuV**2*(sigmaMuT**2+sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*( \
        sigmaMuY**2+sigmaY**2)+sigmaZ**2)))**(-1)
    
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
    P_16  =  np.zeros((n_pts_v, n_sites))
    P_17  =  np.zeros((n_pts_v, n_sites))
    P_18  =  np.zeros((n_pts_v, n_sites))
    P_19  =  np.zeros((n_pts_v, n_sites))
    P_20  =  np.zeros((n_pts_v, n_sites))
    P_21  =  np.zeros((n_pts_v, n_sites))
    P_22  =  np.zeros((n_pts_v, n_sites))
    P_23  =  np.zeros((n_pts_v, n_sites))
    P_24  =  np.zeros((n_pts_v, n_sites))
    P_25  =  np.zeros((n_pts_v, n_sites))
    P_26  =  np.zeros((n_pts_v, n_sites))
    P_27  =  np.zeros((n_pts_v, n_sites))
    P_28  =  np.zeros((n_pts_v, n_sites))
    P_29  =  np.zeros((n_pts_v, n_sites))
    P_30  =  np.zeros((n_pts_v, n_sites))
    P_31  =  np.zeros((n_pts_v, n_sites))
    P_32  =  np.zeros((n_pts_v, n_sites))
    P_33  =  np.zeros((n_pts_v, n_sites))
    P_34  =  np.zeros((n_pts_v, n_sites))
    P_35  =  np.zeros((n_pts_v, n_sites))
    P_36  =  np.zeros((n_pts_v, n_sites))
    P_37  =  np.zeros((n_pts_v, n_sites))
    P_38  =  np.zeros((n_pts_v, n_sites))
    P_39  =  np.zeros((n_pts_v, n_sites))
    P_40  =  np.zeros((n_pts_v, n_sites))
    P_41  =  np.zeros((n_pts_v, n_sites))
    P_42  =  np.zeros((n_pts_v, n_sites))
    P_43  =  np.zeros((n_pts_v, n_sites))
    P_44  =  np.zeros((n_pts_v, n_sites))
    P_45  =  np.zeros((n_pts_v, n_sites))
    P_46  =  np.zeros((n_pts_v, n_sites))
    P_47  =  np.zeros((n_pts_v, n_sites))
    P_48  =  np.zeros((n_pts_v, n_sites))
    P_49  =  np.zeros((n_pts_v, n_sites))
    P_50  =  np.zeros((n_pts_v, n_sites))
    P_51  =  np.zeros((n_pts_v, n_sites))
    P_52  =  np.zeros((n_pts_v, n_sites))
    P_53  =  np.zeros((n_pts_v, n_sites))
    P_54  =  np.zeros((n_pts_v, n_sites))
    P_55  =  np.zeros((n_pts_v, n_sites))
    P_56  =  np.zeros((n_pts_v, n_sites))
    P_57  =  np.zeros((n_pts_v, n_sites))
    P_58  =  np.zeros((n_pts_v, n_sites))
    P_59  =  np.zeros((n_pts_v, n_sites))
    P_60  =  np.zeros((n_pts_v, n_sites))
    P_61  =  np.zeros((n_pts_v, n_sites))
    P_62  =  np.zeros((n_pts_v, n_sites))
    P_63  =  np.zeros((n_pts_v, n_sites))
    P_64  =  np.zeros((n_pts_v, n_sites))
    P_65  =  np.zeros((n_pts_v, n_sites))
    P_66  =  np.zeros((n_pts_v, n_sites))
    P_67  =  np.zeros((n_pts_v, n_sites))
    P_68  =  np.zeros((n_pts_v, n_sites))
    P_69  =  np.zeros((n_pts_v, n_sites))
    P_70  =  np.zeros((n_pts_v, n_sites))
    
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
        P_4[i, :] = (1/4)*a**(-1)*(2*expCst*p4c1+((-1)*a)**(-3/2)*a*((-1)+ErfCst)*( \
            2*a*p4c0+(-1)*b*p4c1)*np.pi**(1/2))
        P_5[i, :] = (1/4)*a**(-1)*(2*expCst*p5c1+((-1)*a)**(-3/2)*a*((-1)+ErfCst)*( \
            2*a*p5c0+(-1)*b*p5c1)*np.pi**(1/2))
        P_6[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p6c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
            *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p6c0*np.pi**( \
            1/2)+2*a*(((-1)+ErfCst)*(b*p6c1+p6c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p6c1+p6c2*v[i])))
        P_7[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p7c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
            *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p7c0*np.pi**( \
            1/2)+2*a*(((-1)+ErfCst)*(b*p7c1+p7c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p7c1+p7c2*v[i])))
        P_8[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p8c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
            *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p8c0*np.pi**( \
            1/2)+2*a*(((-1)+ErfCst)*(b*p8c1+p8c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p8c1+p8c2*v[i])))
        P_9[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p9c2*((-2)*((-1)*a)**(1/2)*expCst+(-1) \
            *b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p9c0*np.pi**( \
            1/2)+2*a*(((-1)+ErfCst)*(b*p9c1+p9c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p9c1+p9c2*v[i])))
        P_10[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p10c2*((-2)*((-1)*a)**(1/2)*expCst+( \
            -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p10c0* \
            np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p10c1+p10c2)*np.pi**(1/2)+2*((-1)*a) \
            **(1/2)*expCst*(p10c1+p10c2*v[i])))
        P_11[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p11c2*((-2)*((-1)*a)**(1/2)*expCst+( \
            -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p11c0* \
            np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p11c1+p11c2)*np.pi**(1/2)+2*((-1)*a) \
            **(1/2)*expCst*(p11c1+p11c2*v[i])))
        P_12[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p12c2*((-2)*((-1)*a)**(1/2)*expCst+( \
            -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p12c0* \
            np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p12c1+p12c2)*np.pi**(1/2)+2*((-1)*a) \
            **(1/2)*expCst*(p12c1+p12c2*v[i])))
        P_13[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p13c2*((-2)*((-1)*a)**(1/2)*expCst+( \
            -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p13c0* \
            np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p13c1+p13c2)*np.pi**(1/2)+2*((-1)*a) \
            **(1/2)*expCst*(p13c1+p13c2*v[i])))
        P_14[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p14c2*((-2)*((-1)*a)**(1/2)*expCst+( \
            -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p14c0* \
            np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p14c1+p14c2)*np.pi**(1/2)+2*((-1)*a) \
            **(1/2)*expCst*(p14c1+p14c2*v[i])))
        P_15[i, :] = (1/8)*((-1)*a)**(-5/2)*(b*p15c2*((-2)*((-1)*a)**(1/2)*expCst+( \
            -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p15c0* \
            np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p15c1+p15c2)*np.pi**(1/2)+2*((-1)*a) \
            **(1/2)*expCst*(p15c1+p15c2*v[i])))
        P_16[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p16c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p16c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p16c3+b**2*((-1)+ErfCst)* \
            p16c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p16c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p16c2+p16c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p16c1+ \
            p16c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p16c1+v[i]*(p16c2+p16c3* \
            v[i]))))
        P_17[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p17c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p17c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p17c3+b**2*((-1)+ErfCst)* \
            p17c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p17c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p17c2+p17c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p17c1+ \
            p17c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p17c1+v[i]*(p17c2+p17c3* \
            v[i]))))
        P_18[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p18c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p18c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p18c3+b**2*((-1)+ErfCst)* \
            p18c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p18c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p18c2+p18c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p18c1+ \
            p18c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p18c1+v[i]*(p18c2+p18c3* \
            v[i]))))
        P_19[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p19c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p19c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p19c3+b**2*((-1)+ErfCst)* \
            p19c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p19c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p19c2+p19c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p19c1+ \
            p19c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p19c1+v[i]*(p19c2+p19c3* \
            v[i]))))
        P_20[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p20c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p20c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p20c3+b**2*((-1)+ErfCst)* \
            p20c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p20c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p20c2+p20c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p20c1+ \
            p20c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p20c1+v[i]*(p20c2+p20c3* \
            v[i]))))
        P_21[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p21c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p21c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p21c3+b**2*((-1)+ErfCst)* \
            p21c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p21c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p21c2+p21c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p21c1+ \
            p21c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p21c1+v[i]*(p21c2+p21c3* \
            v[i]))))
        P_22[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p22c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p22c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p22c3+b**2*((-1)+ErfCst)* \
            p22c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p22c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p22c2+p22c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p22c1+ \
            p22c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p22c1+v[i]*(p22c2+p22c3* \
            v[i]))))
        P_23[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p23c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p23c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p23c3+b**2*((-1)+ErfCst)* \
            p23c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p23c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p23c2+p23c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p23c1+ \
            p23c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p23c1+v[i]*(p23c2+p23c3* \
            v[i]))))
        P_24[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p24c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p24c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p24c3+b**2*((-1)+ErfCst)* \
            p24c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p24c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p24c2+p24c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p24c1+ \
            p24c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p24c1+v[i]*(p24c2+p24c3* \
            v[i]))))
        P_25[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p25c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p25c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p25c3+b**2*((-1)+ErfCst)* \
            p25c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p25c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p25c2+p25c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p25c1+ \
            p25c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p25c1+v[i]*(p25c2+p25c3* \
            v[i]))))
        P_26[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p26c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p26c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p26c3+b**2*((-1)+ErfCst)* \
            p26c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p26c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p26c2+p26c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p26c1+ \
            p26c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p26c1+v[i]*(p26c2+p26c3* \
            v[i]))))
        P_27[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p27c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p27c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p27c3+b**2*((-1)+ErfCst)* \
            p27c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p27c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p27c2+p27c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p27c1+ \
            p27c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p27c1+v[i]*(p27c2+p27c3* \
            v[i]))))
        P_28[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p28c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p28c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p28c3+b**2*((-1)+ErfCst)* \
            p28c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p28c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p28c2+p28c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p28c1+ \
            p28c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p28c1+v[i]*(p28c2+p28c3* \
            v[i]))))
        P_29[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p29c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p29c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p29c3+b**2*((-1)+ErfCst)* \
            p29c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p29c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p29c2+p29c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p29c1+ \
            p29c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p29c1+v[i]*(p29c2+p29c3* \
            v[i]))))
        P_30[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p30c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p30c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p30c3+b**2*((-1)+ErfCst)* \
            p30c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p30c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p30c2+p30c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p30c1+ \
            p30c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p30c1+v[i]*(p30c2+p30c3* \
            v[i]))))
        P_31[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p31c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p31c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p31c3+b**2*((-1)+ErfCst)* \
            p31c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p31c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p31c2+p31c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p31c1+ \
            p31c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p31c1+v[i]*(p31c2+p31c3* \
            v[i]))))
        P_32[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p32c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p32c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p32c3+b**2*((-1)+ErfCst)* \
            p32c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p32c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p32c2+p32c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p32c1+ \
            p32c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p32c1+v[i]*(p32c2+p32c3* \
            v[i]))))
        P_33[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p33c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p33c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p33c3+b**2*((-1)+ErfCst)* \
            p33c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p33c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p33c2+p33c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p33c1+ \
            p33c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p33c1+v[i]*(p33c2+p33c3* \
            v[i]))))
        P_34[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p34c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p34c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p34c3+b**2*((-1)+ErfCst)* \
            p34c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p34c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p34c2+p34c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p34c1+ \
            p34c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p34c1+v[i]*(p34c2+p34c3* \
            v[i]))))
        P_35[i, :] = (-1/16)*((-1)*a)**(-7/2)*(b**2*p35c3*(2*((-1)*a)**(1/2)*expCst+ \
            b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p35c0*np.pi**(1/2) \
            +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p35c3+b**2*((-1)+ErfCst)* \
            p35c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p35c3*np.pi**(1/2)+2*((-1)*a)**( \
            1/2)*b*expCst*(p35c2+p35c3*v[i]))+4*a**2*(((-1)+ErfCst)*(b*p35c1+ \
            p35c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p35c1+v[i]*(p35c2+p35c3* \
            v[i]))))
        P_36[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p36c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p36c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p36c4+b**2*(( \
            -1)+ErfCst)*p36c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p36c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p36c3+p36c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p36c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p36c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p36c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p36c3+3* \
            p36c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p36c2+v[i]*(p36c3+p36c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p36c1+p36c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p36c1+v[i]*(p36c2+v[i]*(p36c3+p36c4*v[i])))))
        P_37[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p37c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p37c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p37c4+b**2*(( \
            -1)+ErfCst)*p37c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p37c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p37c3+p37c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p37c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p37c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p37c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p37c3+3* \
            p37c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p37c2+v[i]*(p37c3+p37c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p37c1+p37c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p37c1+v[i]*(p37c2+v[i]*(p37c3+p37c4*v[i])))))
        P_38[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p38c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p38c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p38c4+b**2*(( \
            -1)+ErfCst)*p38c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p38c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p38c3+p38c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p38c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p38c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p38c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p38c3+3* \
            p38c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p38c2+v[i]*(p38c3+p38c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p38c1+p38c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p38c1+v[i]*(p38c2+v[i]*(p38c3+p38c4*v[i])))))
        P_39[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p39c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p39c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p39c4+b**2*(( \
            -1)+ErfCst)*p39c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p39c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p39c3+p39c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p39c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p39c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p39c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p39c3+3* \
            p39c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p39c2+v[i]*(p39c3+p39c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p39c1+p39c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p39c1+v[i]*(p39c2+v[i]*(p39c3+p39c4*v[i])))))
        P_40[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p40c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p40c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p40c4+b**2*(( \
            -1)+ErfCst)*p40c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p40c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p40c3+p40c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p40c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p40c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p40c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p40c3+3* \
            p40c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p40c2+v[i]*(p40c3+p40c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p40c1+p40c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p40c1+v[i]*(p40c2+v[i]*(p40c3+p40c4*v[i])))))
        P_41[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p41c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p41c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p41c4+b**2*(( \
            -1)+ErfCst)*p41c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p41c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p41c3+p41c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p41c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p41c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p41c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p41c3+3* \
            p41c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p41c2+v[i]*(p41c3+p41c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p41c1+p41c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p41c1+v[i]*(p41c2+v[i]*(p41c3+p41c4*v[i])))))
        P_42[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p42c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p42c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p42c4+b**2*(( \
            -1)+ErfCst)*p42c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p42c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p42c3+p42c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p42c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p42c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p42c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p42c3+3* \
            p42c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p42c2+v[i]*(p42c3+p42c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p42c1+p42c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p42c1+v[i]*(p42c2+v[i]*(p42c3+p42c4*v[i])))))
        P_43[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p43c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p43c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p43c4+b**2*(( \
            -1)+ErfCst)*p43c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p43c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p43c3+p43c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p43c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p43c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p43c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p43c3+3* \
            p43c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p43c2+v[i]*(p43c3+p43c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p43c1+p43c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p43c1+v[i]*(p43c2+v[i]*(p43c3+p43c4*v[i])))))
        P_44[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p44c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p44c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p44c4+b**2*(( \
            -1)+ErfCst)*p44c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p44c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p44c3+p44c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p44c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p44c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p44c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p44c3+3* \
            p44c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p44c2+v[i]*(p44c3+p44c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p44c1+p44c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p44c1+v[i]*(p44c2+v[i]*(p44c3+p44c4*v[i])))))
        P_45[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p45c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p45c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p45c4+b**2*(( \
            -1)+ErfCst)*p45c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p45c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p45c3+p45c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p45c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p45c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p45c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p45c3+3* \
            p45c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p45c2+v[i]*(p45c3+p45c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p45c1+p45c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p45c1+v[i]*(p45c2+v[i]*(p45c3+p45c4*v[i])))))
        P_46[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p46c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p46c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p46c4+b**2*(( \
            -1)+ErfCst)*p46c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p46c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p46c3+p46c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p46c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p46c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p46c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p46c3+3* \
            p46c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p46c2+v[i]*(p46c3+p46c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p46c1+p46c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p46c1+v[i]*(p46c2+v[i]*(p46c3+p46c4*v[i])))))
        P_47[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p47c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p47c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p47c4+b**2*(( \
            -1)+ErfCst)*p47c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p47c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p47c3+p47c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p47c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p47c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p47c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p47c3+3* \
            p47c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p47c2+v[i]*(p47c3+p47c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p47c1+p47c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p47c1+v[i]*(p47c2+v[i]*(p47c3+p47c4*v[i])))))
        P_48[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p48c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p48c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p48c4+b**2*(( \
            -1)+ErfCst)*p48c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p48c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p48c3+p48c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p48c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p48c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p48c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p48c3+3* \
            p48c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p48c2+v[i]*(p48c3+p48c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p48c1+p48c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p48c1+v[i]*(p48c2+v[i]*(p48c3+p48c4*v[i])))))
        P_49[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p49c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p49c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p49c4+b**2*(( \
            -1)+ErfCst)*p49c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p49c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p49c3+p49c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p49c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p49c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p49c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p49c3+3* \
            p49c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p49c2+v[i]*(p49c3+p49c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p49c1+p49c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p49c1+v[i]*(p49c2+v[i]*(p49c3+p49c4*v[i])))))
        P_50[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p50c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p50c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p50c4+b**2*(( \
            -1)+ErfCst)*p50c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p50c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p50c3+p50c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p50c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p50c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p50c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p50c3+3* \
            p50c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p50c2+v[i]*(p50c3+p50c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p50c1+p50c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p50c1+v[i]*(p50c2+v[i]*(p50c3+p50c4*v[i])))))
        P_51[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p51c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p51c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p51c4+b**2*(( \
            -1)+ErfCst)*p51c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p51c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p51c3+p51c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p51c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p51c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p51c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p51c3+3* \
            p51c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p51c2+v[i]*(p51c3+p51c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p51c1+p51c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p51c1+v[i]*(p51c2+v[i]*(p51c3+p51c4*v[i])))))
        P_52[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p52c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p52c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p52c4+b**2*(( \
            -1)+ErfCst)*p52c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p52c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p52c3+p52c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p52c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p52c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p52c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p52c3+3* \
            p52c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p52c2+v[i]*(p52c3+p52c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p52c1+p52c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p52c1+v[i]*(p52c2+v[i]*(p52c3+p52c4*v[i])))))
        P_53[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p53c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p53c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p53c4+b**2*(( \
            -1)+ErfCst)*p53c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p53c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p53c3+p53c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p53c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p53c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p53c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p53c3+3* \
            p53c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p53c2+v[i]*(p53c3+p53c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p53c1+p53c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p53c1+v[i]*(p53c2+v[i]*(p53c3+p53c4*v[i])))))
        P_54[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p54c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p54c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p54c4+b**2*(( \
            -1)+ErfCst)*p54c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p54c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p54c3+p54c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p54c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p54c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p54c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p54c3+3* \
            p54c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p54c2+v[i]*(p54c3+p54c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p54c1+p54c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p54c1+v[i]*(p54c2+v[i]*(p54c3+p54c4*v[i])))))
        P_55[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p55c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p55c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p55c4+b**2*(( \
            -1)+ErfCst)*p55c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p55c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p55c3+p55c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p55c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p55c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p55c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p55c3+3* \
            p55c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p55c2+v[i]*(p55c3+p55c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p55c1+p55c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p55c1+v[i]*(p55c2+v[i]*(p55c3+p55c4*v[i])))))
        P_56[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p56c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p56c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p56c4+b**2*(( \
            -1)+ErfCst)*p56c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p56c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p56c3+p56c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p56c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p56c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p56c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p56c3+3* \
            p56c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p56c2+v[i]*(p56c3+p56c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p56c1+p56c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p56c1+v[i]*(p56c2+v[i]*(p56c3+p56c4*v[i])))))
        P_57[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p57c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p57c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p57c4+b**2*(( \
            -1)+ErfCst)*p57c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p57c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p57c3+p57c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p57c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p57c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p57c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p57c3+3* \
            p57c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p57c2+v[i]*(p57c3+p57c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p57c1+p57c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p57c1+v[i]*(p57c2+v[i]*(p57c3+p57c4*v[i])))))
        P_58[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p58c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p58c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p58c4+b**2*(( \
            -1)+ErfCst)*p58c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p58c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p58c3+p58c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p58c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p58c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p58c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p58c3+3* \
            p58c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p58c2+v[i]*(p58c3+p58c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p58c1+p58c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p58c1+v[i]*(p58c2+v[i]*(p58c3+p58c4*v[i])))))
        P_59[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p59c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p59c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p59c4+b**2*(( \
            -1)+ErfCst)*p59c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p59c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p59c3+p59c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p59c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p59c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p59c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p59c3+3* \
            p59c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p59c2+v[i]*(p59c3+p59c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p59c1+p59c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p59c1+v[i]*(p59c2+v[i]*(p59c3+p59c4*v[i])))))
        P_60[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p60c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p60c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p60c4+b**2*(( \
            -1)+ErfCst)*p60c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p60c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p60c3+p60c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p60c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p60c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p60c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p60c3+3* \
            p60c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p60c2+v[i]*(p60c3+p60c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p60c1+p60c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p60c1+v[i]*(p60c2+v[i]*(p60c3+p60c4*v[i])))))
        P_61[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p61c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p61c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p61c4+b**2*(( \
            -1)+ErfCst)*p61c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p61c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p61c3+p61c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p61c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p61c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p61c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p61c3+3* \
            p61c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p61c2+v[i]*(p61c3+p61c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p61c1+p61c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p61c1+v[i]*(p61c2+v[i]*(p61c3+p61c4*v[i])))))
        P_62[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p62c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p62c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p62c4+b**2*(( \
            -1)+ErfCst)*p62c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p62c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p62c3+p62c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p62c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p62c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p62c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p62c3+3* \
            p62c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p62c2+v[i]*(p62c3+p62c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p62c1+p62c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p62c1+v[i]*(p62c2+v[i]*(p62c3+p62c4*v[i])))))
        P_63[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p63c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p63c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p63c4+b**2*(( \
            -1)+ErfCst)*p63c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p63c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p63c3+p63c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p63c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p63c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p63c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p63c3+3* \
            p63c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p63c2+v[i]*(p63c3+p63c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p63c1+p63c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p63c1+v[i]*(p63c2+v[i]*(p63c3+p63c4*v[i])))))
        P_64[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p64c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p64c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p64c4+b**2*(( \
            -1)+ErfCst)*p64c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p64c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p64c3+p64c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p64c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p64c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p64c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p64c3+3* \
            p64c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p64c2+v[i]*(p64c3+p64c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p64c1+p64c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p64c1+v[i]*(p64c2+v[i]*(p64c3+p64c4*v[i])))))
        P_65[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p65c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p65c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p65c4+b**2*(( \
            -1)+ErfCst)*p65c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p65c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p65c3+p65c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p65c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p65c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p65c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p65c3+3* \
            p65c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p65c2+v[i]*(p65c3+p65c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p65c1+p65c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p65c1+v[i]*(p65c2+v[i]*(p65c3+p65c4*v[i])))))
        P_66[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p66c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p66c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p66c4+b**2*(( \
            -1)+ErfCst)*p66c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p66c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p66c3+p66c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p66c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p66c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p66c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p66c3+3* \
            p66c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p66c2+v[i]*(p66c3+p66c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p66c1+p66c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p66c1+v[i]*(p66c2+v[i]*(p66c3+p66c4*v[i])))))
        P_67[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p67c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p67c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p67c4+b**2*(( \
            -1)+ErfCst)*p67c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p67c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p67c3+p67c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p67c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p67c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p67c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p67c3+3* \
            p67c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p67c2+v[i]*(p67c3+p67c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p67c1+p67c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p67c1+v[i]*(p67c2+v[i]*(p67c3+p67c4*v[i])))))
        P_68[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p68c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p68c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p68c4+b**2*(( \
            -1)+ErfCst)*p68c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p68c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p68c3+p68c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p68c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p68c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p68c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p68c3+3* \
            p68c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p68c2+v[i]*(p68c3+p68c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p68c1+p68c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p68c1+v[i]*(p68c2+v[i]*(p68c3+p68c4*v[i])))))
        P_69[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p69c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p69c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p69c4+b**2*(( \
            -1)+ErfCst)*p69c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p69c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p69c3+p69c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p69c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p69c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p69c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p69c3+3* \
            p69c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p69c2+v[i]*(p69c3+p69c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p69c1+p69c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p69c1+v[i]*(p69c2+v[i]*(p69c3+p69c4*v[i])))))
        P_70[i, :] = (1/32)*((-1)*a)**(-9/2)*(b**3*p70c4*((-2)*((-1)*a)**(1/2)* \
            expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
            p70c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p70c4+b**2*(( \
            -1)+ErfCst)*p70c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p70c4*np.pi**(1/2)+2*( \
            (-1)*a)**(1/2)*b*expCst*(p70c3+p70c4*v[i]))+(-4)*a**2*(b**2*((-1)+ \
            ErfCst)*p70c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p70c3*np.pi**(1/2)+3*((-1) \
            +ErfCst)*p70c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p70c3+3* \
            p70c4*v[i])+2*((-1)*a)**(1/2)*b*expCst*(p70c2+v[i]*(p70c3+p70c4*v[i])))+ \
            8*a**3*(((-1)+ErfCst)*(b*p70c1+p70c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
            *expCst*(p70c1+v[i]*(p70c2+v[i]*(p70c3+p70c4*v[i])))))
        
        
        P_output = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13, \
            P_14, P_15, P_16, P_17, P_18, P_19, P_20, P_21, P_22, P_23, P_24, P_25, \
            P_26, P_27, P_28, P_29, P_30, P_31, P_32, P_33, P_34, P_35, P_36, P_37, \
            P_38, P_39, P_40, P_41, P_42, P_43, P_44, P_45, P_46, P_47, P_48, P_49, \
            P_50, P_51, P_52, P_53, P_54, P_55, P_56, P_57, P_58, P_59, P_60, P_61, \
            P_62, P_63, P_64, P_65, P_66, P_67, P_68, P_69, P_70))
        
         #
    #;
    #;
    return P_output
