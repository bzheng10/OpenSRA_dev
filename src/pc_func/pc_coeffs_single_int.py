# import numpy as np
from numpy import exp, pi, zeros
from .pc_util import erf
from numba import njit, float64
# from .pc_coeffs_first_int import pc_coeffs_first_int

@njit(
    float64[:,:](
        float64[:],float64[:],float64[:],
        float64[:],float64[:],float64[:]
    ),
    fastmath = True,
    cache = True
)
def pc_coeffs_single_int(
    muY, sigmaMuY, sigmaY,
    muZ, sigmaMuZ, sigmaZ
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
    n_sites = len(muY)
    
    #
    ## Total Constant
    #
    a1 = (-1/2)*(sigmaMuY**2+sigmaY**2)**(-1)*(sigmaMuZ**2+sigmaZ**2)**(-1)*( \
        sigmaMuY**2+sigmaMuZ**2+sigmaY**2+sigmaZ**2)
    b1 = muY*(sigmaMuY**2+sigmaY**2)**(-1)+muZ*(sigmaMuZ**2+sigmaZ**2)**(-1)
    c1 = (-1/2)*muY**2*(sigmaMuY**2+sigmaY**2)**(-1)+(-1/2)*muZ**2*( \
        sigmaMuZ**2+sigmaZ**2)**(-1)
    a2 = (-1/2)*(sigmaMuY**2+sigmaY**2)**(-1)
    b2 = muY*(sigmaMuY**2+sigmaY**2)**(-1)
    c2 = (-1/2)*muY**2*(sigmaMuY**2+sigmaY**2)**(-1)
    muPhi = muZ
    sigmaPhi = (sigmaMuZ**2+sigmaZ**2)**(1/2)
    ConstantExp1 = exp(1)**((-1/4)*a1**(-1)*b1**2+c1)*pi**(1/2)
    ConstantExp2 = exp(1)**((2+(-4)*a2*sigmaPhi**2)**(-1)*(2*c2+2*b2*muPhi+2*a2* \
        muPhi**2+b2**2*sigmaPhi**2+(-4)*a2*c2*sigmaPhi**2))
    Constanterf = exp(1)**((-1/4)*a2**(-1)*b2**2+c2)*pi**(1/2)*((-1)+erf((1/2)*a2**( \
        -1)*(b2+2*a2*muPhi)*((-1)*a2**(-1)+2*sigmaPhi**2)**(-1/2)))

    #
    ## Polynomial Coefficients
    #################
    d0 = 1
    P_1 = (-1/2)*((-1)*a2)**(-1/2)*Constanterf*d0*(2*pi)**(-1/2)*(1+ \
        sigmaMuY**2*sigmaY**(-2))**(-1/2)*sigmaY**(-1)
    
    #################
    d0 = sigmaMuZ
    P_2 = (1/2)*((-1)*a1)**(-1/2)*ConstantExp1*d0*pi**(-1)*(1+sigmaMuY**2* \
        sigmaY**(-2))**(-1/2)*sigmaY**(-1)*(1+sigmaMuZ**2*sigmaZ**(-2))**( \
        -1/2)*sigmaZ**(-1)
        
    #################
    d0 = (-1)*muY*sigmaMuY
    d1 = sigmaMuY
    P_3 = (1/4)*(2*pi)**(-1/2)*(((-1)*a2)**(-3/2)*Constanterf*(2*a2*d0+( \
        -1)*b2*d1)+(-2)*a2**(-1)*ConstantExp2*d1*(1+(-2)*a2*sigmaPhi**2) \
        **(-1/2))*(1+sigmaMuY**2*sigmaY**(-2))**(-3/2)*sigmaY**(-3)
    
    #################
    d0 = (-1)*muZ*sigmaMuZ**2
    d1 = sigmaMuZ**2
    P_4 = (1/8)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*pi**(-1) \
        *(1+sigmaMuY**2*sigmaY**(-2))**(-1/2)*sigmaY**(-1)*(1+sigmaMuZ**2* \
        sigmaZ**(-2))**(-1/2)*sigmaZ**(-1)*(sigmaMuZ**2+sigmaZ**2)**(-1)
        
    #################
    d0 = (-1)*muY*sigmaMuY*sigmaMuZ
    d1 = sigmaMuY*sigmaMuZ
    P_5 = (1/4)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*pi**(-1) \
        *(1+sigmaMuY**2*sigmaY**(-2))**(-3/2)*sigmaY**(-3)*(1+sigmaMuZ**2* \
        sigmaZ**(-2))**(-1/2)*sigmaZ**(-1)
        
    #################
    d0 = muY**2*sigmaMuY**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY+(-1)* \
        sigmaMuY**4*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY+(-1)* \
        sigmaMuY**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY**3
    d1 = (-2)*muY*sigmaMuY**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY
    d2 = sigmaMuY**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY
    P_6 = (1/2)*(2*pi)**(-1/2)*((-1/8)*((-1)*a2)**(-5/2)*Constanterf*(4* \
        a2**2*d0+b2**2*d2+(-2)*a2*(b2*d1+d2))+(1/4)*a2**(-2)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1* \
        sigmaPhi**2+(-2)*a2*(d1+d2*(muPhi+2*b2*sigmaPhi**2))))*( \
        sigmaMuY**2+sigmaY**2)**(-3)
        
    #################
    d0 = muZ**2*sigmaMuZ**3+(-1)*sigmaMuZ**5+(-1)*sigmaMuZ**3*sigmaZ**2
    d1 = (-2)*muZ*sigmaMuZ**3
    d2 = sigmaMuZ**3
    P_7 = (1/48)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*pi**(-1)*(1+sigmaMuY**2*sigmaY**(-2))**(-1/2)* \
        sigmaY**(-1)*(1+sigmaMuZ**2*sigmaZ**(-2))**(-1/2)*sigmaZ**(-1)*( \
        sigmaMuZ**2+sigmaZ**2)**(-2)
        
    #################
    d0 = muY*muZ*sigmaMuY*sigmaMuZ**2
    d1 = (-1)*muY*sigmaMuY*sigmaMuZ**2+(-1)*muZ*sigmaMuY*sigmaMuZ**2
    d2 = sigmaMuY*sigmaMuZ**2
    P_8 = (1/16)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*pi**(-1)*(1+sigmaMuY**2*sigmaY**(-2))**(-3/2)* \
        sigmaY**(-3)*(1+sigmaMuZ**2*sigmaZ**(-2))**(-3/2)*sigmaZ**(-3)
        
    #################
    d0 = muY**2*sigmaMuY**2*sigmaMuZ+(-1)*sigmaMuY**4*sigmaMuZ+(-1)* \
        sigmaMuY**2*sigmaMuZ*sigmaY**2
    d1 = (-2)*muY*sigmaMuY**2*sigmaMuZ
    d2 = sigmaMuY**2*sigmaMuZ
    P_9 = (1/16)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*pi**(-1)*(1+sigmaMuY**2*sigmaY**(-2))**(-1/2)* \
        sigmaY**(-1)*(sigmaMuY**2+sigmaY**2)**(-2)*(1+sigmaMuZ**2*sigmaZ**( \
        -2))**(-1/2)*sigmaZ**(-1)
        
    #################
    d0 = (-1)*muY**3*sigmaMuY**3+3*muY*sigmaMuY**3*(sigmaMuY**2+sigmaY**2)
    d1 = 3*muY**2*sigmaMuY**3+(-3)*sigmaMuY**3*(sigmaMuY**2+sigmaY**2)
    d2 = (-3)*muY*sigmaMuY**3
    d3 = sigmaMuY**3
    P_10 = (1/96)*(2*pi)**(-1/2)*(((-1)*a2)**(-7/2)*Constanterf*(8*a2**3* \
        d0+(-4)*a2**2*(b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+( \
        -2)*a2**(-3)*ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2* \
        d3+16*a2**4*d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3* \
        b2**2*d3*sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3* \
        sigmaPhi**2+d2*(muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3* \
        b2*sigmaPhi**2)+d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+ \
        3*b2**2*sigmaPhi**4))))*(1+sigmaMuY**2*sigmaY**(-2))**(-1/2)* \
        sigmaY**(-1)*(sigmaMuY**2+sigmaY**2)**(-3)
        
    #################
    d0 = (-1)*muZ**3*sigmaMuZ**4+3*muZ*sigmaMuZ**4*(sigmaMuZ**2+sigmaZ**2)
    d1 = 3*muZ**2*sigmaMuZ**4+(-3)*sigmaMuZ**4*(sigmaMuZ**2+sigmaZ**2)
    d2 = (-3)*muZ*sigmaMuZ**4
    d3 = sigmaMuZ**4
    P_11 = (1/384)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1)*(1+ \
        sigmaMuY**2*sigmaY**(-2))**(-1/2)*sigmaY**(-1)*(1+sigmaMuZ**2* \
        sigmaZ**(-2))**(-1/2)*sigmaZ**(-1)*(sigmaMuZ**2+sigmaZ**2)**(-3)
        
    #################
    d0 = (-1)*muY*muZ**2*sigmaMuY*sigmaMuZ**3*(1+sigmaMuY**2*sigmaY**(-2)) \
        **(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+muY* \
        sigmaMuY*sigmaMuZ**5*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+ \
        sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+muY*sigmaMuY*sigmaMuZ**3*( \
        1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**( \
        -2))**(1/2)*sigmaZ**3
    d1 = 2*muY*muZ*sigmaMuY*sigmaMuZ**3*(1+sigmaMuY**2*sigmaY**(-2))**(1/2) \
        *sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+muZ**2* \
        sigmaMuY*sigmaMuZ**3*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+ \
        sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+(-1)*sigmaMuY*sigmaMuZ**5*( \
        1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**( \
        -2))**(1/2)*sigmaZ+(-1)*sigmaMuY*sigmaMuZ**3*(1+sigmaMuY**2* \
        sigmaY**(-2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)* \
        sigmaZ**3
    d2 = (-1)*muY*sigmaMuY*sigmaMuZ**3*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)* \
        sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+(-2)*muZ* \
        sigmaMuY*sigmaMuZ**3*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+ \
        sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ
    d3 = sigmaMuY*sigmaMuZ**3*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+ \
        sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ
    P_12 = (1/96)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1)*( \
        sigmaMuY**2+sigmaY**2)**(-2)*(sigmaMuZ**2+sigmaZ**2)**(-3)
        
    #################
    d0 = (-1)*muY**2*muZ*sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**( \
        -2))**(1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+muZ* \
        sigmaMuY**4*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY* \
        (1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+muZ*sigmaMuY**2* \
        sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY**3*(1+ \
        sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ
    d1 = muY**2*sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)* \
        sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+2*muY*muZ* \
        sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY* \
        (1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+(-1)*sigmaMuY**4* \
        sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY*(1+ \
        sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+(-1)*sigmaMuY**2* \
        sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY**3*(1+ \
        sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ
    d2 = (-2)*muY*sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**( \
        1/2)*sigmaY*(1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ+(-1)*muZ* \
        sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY* \
        (1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ
    d3 = sigmaMuY**2*sigmaMuZ**2*(1+sigmaMuY**2*sigmaY**(-2))**(1/2)*sigmaY* \
        (1+sigmaMuZ**2*sigmaZ**(-2))**(1/2)*sigmaZ
    P_13 = (1/64)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1)*( \
        sigmaMuY**2+sigmaY**2)**(-3)*(sigmaMuZ**2+sigmaZ**2)**(-2)
        
    #################
    d0 = (-1)*muY**3*sigmaMuY**3*sigmaMuZ+3*muY*sigmaMuY**3*sigmaMuZ*( \
        sigmaMuY**2+sigmaY**2)
    d1 = 3*muY**2*sigmaMuY**3*sigmaMuZ+(-3)*sigmaMuY**3*sigmaMuZ*( \
        sigmaMuY**2+sigmaY**2)
    d2 = (-3)*muY*sigmaMuY**3*sigmaMuZ
    d3 = sigmaMuY**3*sigmaMuZ
    P_14 = (1/96)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1)*(1+ \
        sigmaMuY**2*sigmaY**(-2))**(-1/2)*sigmaY**(-1)*(sigmaMuY**2+ \
        sigmaY**2)**(-3)*(1+sigmaMuZ**2*sigmaZ**(-2))**(-1/2)*sigmaZ**(-1)
        
    #################
    d0 = muY**4*sigmaMuY**4+(-6)*muY**2*sigmaMuY**4*(sigmaMuY**2+sigmaY**2)+ \
        3*sigmaMuY**4*(sigmaMuY**2+sigmaY**2)**2
    d1 = (-4)*sigmaMuY**4*(muY**3+(-3)*muY*(sigmaMuY**2+sigmaY**2))
    d2 = 6*sigmaMuY**4*(muY**2+(-1)*sigmaMuY**2+(-1)*sigmaY**2)
    d3 = (-4)*muY*sigmaMuY**4
    d4 = sigmaMuY**4
    P_15 = (1/768)*(2*pi)**(-1/2)*(((-1)*a2)**(-9/2)*Constanterf*((-16)* \
        a2**4*d0+8*a2**3*(b2*d1+d2)+(-1)*b2**4*d4+(-4)*a2**2*(b2**2*d2+ \
        3*b2*d3+3*d4)+2*a2*b2**2*(b2*d3+6*d4))+2*a2**(-4)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-7/2)*(8*a2**3*d1*((-1)+ \
        2*a2*sigmaPhi**2)**3+(-4)*a2**2*d2*(1+(-2)*a2*sigmaPhi**2)**2*( \
        2*a2*muPhi+b2*((-1)+4*a2*sigmaPhi**2))+(-2)*a2*d3*(1+(-2)*a2* \
        sigmaPhi**2)*(b2**2+(-24)*a2**3*sigmaPhi**4+(-2)*a2*(2+b2*muPhi+ \
        3*b2**2*sigmaPhi**2)+4*a2**2*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi* \
        sigmaPhi**2+3*b2**2*sigmaPhi**4))+d4*(b2**3+96*a2**4*sigmaPhi**4*( \
        muPhi+2*b2*sigmaPhi**2)+(-2)*a2*b2*(5+b2*muPhi+4*b2**2* \
        sigmaPhi**2)+4*a2**2*(3*muPhi+b2*muPhi**2+19*b2*sigmaPhi**2+4* \
        b2**2*muPhi*sigmaPhi**2+6*b2**3*sigmaPhi**4)+(-8)*a2**3*(muPhi**3+ \
        9*muPhi*sigmaPhi**2+4*b2*muPhi**2*sigmaPhi**2+26*b2*sigmaPhi**4+ \
        6*b2**2*muPhi*sigmaPhi**4+4*b2**3*sigmaPhi**6))))*(1+sigmaMuY**2* \
        sigmaY**(-2))**(-1/2)*sigmaY**(-1)*(sigmaMuY**2+sigmaY**2)**(-4)
        
    # store to output
    P_output = zeros((n_sites, 15))
    P_output[:,0] = P_1
    P_output[:,1] = P_2
    P_output[:,2] = P_3
    P_output[:,3] = P_4
    P_output[:,4] = P_5
    P_output[:,5] = P_6
    P_output[:,6] = P_7
    P_output[:,7] = P_8
    P_output[:,8] = P_9
    P_output[:,9] = P_10
    P_output[:,10] = P_11
    P_output[:,11] = P_12
    P_output[:,12] = P_13
    P_output[:,13] = P_14
    P_output[:,14] = P_15
    
    # return
    return P_output