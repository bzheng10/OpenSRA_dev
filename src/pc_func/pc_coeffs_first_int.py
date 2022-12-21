from numpy import zeros, pi, vstack
from numba import njit, float64

@njit(
    float64[:,:](float64[:],float64[:],float64[:],
                 float64[:],float64[:],float64[:],float64[:]),
    fastmath=True,
    cache=True
)
def pc_coeffs_first_int(
    muY, sigmaMuY, sigmaY,\
    amuZ, bmuZ, sigmaMuZ, sigmaZ
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
    
    ##;
    ## Total Constant;
    a1 = (-1/2)*(sigmaMuY**2+sigmaY**2)**(-1)*(sigmaMuZ**2+sigmaZ**2)**(-1)*( \
        sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)
    b00 = muY*(sigmaMuY**2+sigmaY**2)**(-1)+(-1)*amuZ*bmuZ*(sigmaMuZ**2+ \
        sigmaZ**2)**(-1)
    b01 = amuZ*(sigmaMuZ**2+sigmaZ**2)**(-1)
    P_1 =  zeros((1, n_sites))
    P_2 =  zeros((2, n_sites))
    P_3 =  zeros((2, n_sites))
    P_4 =  zeros((3, n_sites))
    P_5 =  zeros((3, n_sites))
    P_6 =  zeros((3, n_sites))
    P_7 =  zeros((4, n_sites))
    P_8 =  zeros((4, n_sites))
    P_9 =  zeros((4, n_sites))
    P_10 =  zeros((4, n_sites))
    P_11 =  zeros((5, n_sites))
    P_12 =  zeros((5, n_sites))
    P_13 =  zeros((5, n_sites))
    P_14 =  zeros((5, n_sites))
    P_15 =  zeros((5, n_sites))
    #;
    ## Polynomial Constants;
    #;
    constantTerm1 = (1/2)*((-1)*a1)**(-1/2)*pi**(-1/2)*(1+sigmaMuY**2*sigmaY**(-2))**( \
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
    P_output = vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13, P_14, P_15))
    
    #
    return P_output
