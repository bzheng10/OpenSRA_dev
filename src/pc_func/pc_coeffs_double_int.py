# import numpy as np
from numpy import exp, pi, zeros
from .pc_util import erf
from numba import njit, float64
from .pc_coeffs_first_int import pc_coeffs_first_int

@njit(
    float64[:,:](
        float64[:],float64[:],float64[:],
        float64[:],float64[:],float64[:],float64[:],
        float64[:],float64[:],float64[:]
    ),
    fastmath = True,
    cache = True
)
def pc_coeffs_double_int(
    muY, sigmaMuY, sigmaY,
    amuZ, bmuZ, sigmaMuZ, sigmaZ,
    muT, sigmaMuT, sigmaT
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

    # P_output = [P_1 ...P_70] is a Matlab array containing 70 stacked arrays,
    # P_i, i = 1,...70
    #
    # Each array P_i is a (n_v * n_sites) array of type double, containing the 
    # Polynomial Chaos coefficients of the risk curve, such that P_i(j, k) 
    # represents the PC coefficient of DV value ln(v(j)) at site k.
    
    """
    # dimensions
    n_sites = len(muY)
    
    ## PC Coefficients from First Integration!!
    P_output_1 = pc_coeffs_first_int(
        muY, sigmaMuY, sigmaY,
        amuZ, bmuZ, sigmaMuZ, sigmaZ
    )
    
    # pull results
    p1c0 = zeros((1,n_sites))
    p2c0 = zeros((1,n_sites))
    p2c1 = zeros((1,n_sites))
    p3c0 = zeros((1,n_sites))
    p3c1 = zeros((1,n_sites))
    p4c0 = zeros((1,n_sites))
    p4c1 = zeros((1,n_sites))
    p4c2 = zeros((1,n_sites))
    p5c0 = zeros((1,n_sites))
    p5c1 = zeros((1,n_sites))
    p5c2 = zeros((1,n_sites))
    p6c0 = zeros((1,n_sites))
    p6c1 = zeros((1,n_sites))
    p6c2 = zeros((1,n_sites))
    p7c0 = zeros((1,n_sites))
    p7c1 = zeros((1,n_sites))
    p7c2 = zeros((1,n_sites))
    p7c3 = zeros((1,n_sites))
    p8c0 = zeros((1,n_sites))
    p8c1 = zeros((1,n_sites))
    p8c2 = zeros((1,n_sites))
    p8c3 = zeros((1,n_sites))
    p9c0 = zeros((1,n_sites))
    p9c1 = zeros((1,n_sites))
    p9c2 = zeros((1,n_sites))
    p9c3 = zeros((1,n_sites))
    p10c0 = zeros((1,n_sites))
    p10c1 = zeros((1,n_sites))
    p10c2 = zeros((1,n_sites))
    p10c3 = zeros((1,n_sites))
    p11c0 = zeros((1,n_sites))
    p11c1 = zeros((1,n_sites))
    p11c2 = zeros((1,n_sites))
    p11c3 = zeros((1,n_sites))
    p11c4 = zeros((1,n_sites))
    p12c0 = zeros((1,n_sites))
    p12c1 = zeros((1,n_sites))
    p12c2 = zeros((1,n_sites))
    p12c3 = zeros((1,n_sites))
    p12c4 = zeros((1,n_sites))
    p13c0 = zeros((1,n_sites))
    p13c1 = zeros((1,n_sites))
    p13c2 = zeros((1,n_sites))
    p13c3 = zeros((1,n_sites))
    p13c4 = zeros((1,n_sites))
    p14c0 = zeros((1,n_sites))
    p14c1 = zeros((1,n_sites))
    p14c2 = zeros((1,n_sites))
    p14c3 = zeros((1,n_sites))
    p14c4 = zeros((1,n_sites))
    p15c0 = zeros((1,n_sites))
    p15c1 = zeros((1,n_sites))
    p15c2 = zeros((1,n_sites))
    p15c3 = zeros((1,n_sites))
    p15c4 = zeros((1,n_sites))
    
    # pull results and reshape
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
    
    #
    ## Total Constant
    #
    a1 = (1/4)*((-2)*(sigmaMuT**2+sigmaT**2)**(-1)+(-2)*(sigmaMuZ**2+amuZ**2* \
        (sigmaMuY**2+sigmaY**2)+sigmaZ**2)**(-1))
    b1 = muT*(sigmaMuT**2+sigmaT**2)**(-1)+(bmuZ+amuZ*muY)*(sigmaMuZ**2+ \
        amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)**(-1)
    c1 = (1/4)*((-2)*muT**2*(sigmaMuT**2+sigmaT**2)**(-1)+(-2)*(bmuZ+amuZ* \
        muY)**2*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)**(-1))
    a2 = (-1/2)*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)**(-1)
    b2 = (bmuZ+amuZ*muY)*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+ \
        sigmaZ**2)**(-1)
    c2 = (-1/2)*(bmuZ+amuZ*muY)**2*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+ \
        sigmaY**2)+sigmaZ**2)**(-1)
    muPhi = muT
    sigmaPhi = (sigmaMuT**2+sigmaT**2)**(1/2)
    ConstantExp1 = exp(1)**((-1/4)*a1**(-1)*b1**2+c1)*pi**(1/2)
    ConstantExp2 = exp(1)**((2+(-4)*a2*sigmaPhi**2)**(-1)*(2*c2+2*b2*muPhi+2*a2* \
        muPhi**2+b2**2*sigmaPhi**2+(-4)*a2*c2*sigmaPhi**2))
    Constanterf = exp(1)**((-1/4)*a2**(-1)*b2**2+c2)*pi**(1/2)*((-1)+erf((1/2)*a2**( \
        -1)*(b2+2*a2*muPhi)*((-1)*a2**(-1)+2*sigmaPhi**2)**(-1/2)))
    
    #
    ## Polynomial Coefficients
    #################
    d0 = p1c0
    P_1 = (-1/2)*((-1)*a2)**(-1/2)*Constanterf*d0
    
    #################
    d0 = p1c0*sigmaMuT
    P_2 = ((-1)*a1)**(-1/2)*ConstantExp1*d0*(2*pi)**(-1/2)*(1+sigmaMuT**2* \
        sigmaT**(-2))**(-1/2)*sigmaT**(-1)
    
    #################
    d0 = p2c0
    d1 = p2c1
    P_3 = (1/4)*(((-1)*a2)**(-3/2)*Constanterf*(2*a2*d0+(-1)*b2*d1)+(-2)* \
        a2**(-1)*ConstantExp2*d1*(1+(-2)*a2*sigmaPhi**2)**(-1/2))
    
    #################
    d0 = p3c0
    d1 = p3c1
    P_4 = (1/4)*(((-1)*a2)**(-3/2)*Constanterf*(2*a2*d0+(-1)*b2*d1)+(-2)* \
        a2**(-1)*ConstantExp2*d1*(1+(-2)*a2*sigmaPhi**2)**(-1/2))
    
    #################
    d0 = (-1)*muT*p1c0*sigmaMuT**2
    d1 = p1c0*sigmaMuT**2
    P_5 = (1/4)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*(2*pi) \
        **(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-3/2)*sigmaT**(-3)
    
    #################
    d0 = p2c0*sigmaMuT
    d1 = p2c1*sigmaMuT
    P_6 = (1/2)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*(2*pi) \
        **(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-1/2)*sigmaT**(-1)
    
    #################
    d0 = p3c0*sigmaMuT
    d1 = p3c1*sigmaMuT
    P_7 = (1/2)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*(2*pi) \
        **(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-1/2)*sigmaT**(-1)
    
    #################
    d0 = p4c0
    d1 = p4c1
    d2 = p4c2
    P_8 = (-1/8)*((-1)*a2)**(-5/2)*Constanterf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = p5c0
    d1 = p5c1
    d2 = p5c2
    P_9 = (-1/8)*((-1)*a2)**(-5/2)*Constanterf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = p6c0
    d1 = p6c1
    d2 = p6c2
    P_10 = (-1/8)*((-1)*a2)**(-5/2)*Constanterf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = muT**2*p1c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT+ \
    (-1)*p1c0*sigmaMuT**5*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT+( \
    -1)*p1c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT**3
    d1 = (-2)*muT*p1c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT
    d2 = p1c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT
    P_11 = (1/24)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(sigmaMuT**2+sigmaT**2)**(-3)
    
    #################
    d0 = (-1)*muT*p2c0*sigmaMuT**2
    d1 = p2c0*sigmaMuT**2+(-1)*muT*p2c1*sigmaMuT**2
    d2 = p2c1*sigmaMuT**2
    P_12 = (1/8)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-3/2) \
        *sigmaT**(-3)
    
    #################
    d0 = (-1)*muT*p3c0*sigmaMuT**2
    d1 = p3c0*sigmaMuT**2+(-1)*muT*p3c1*sigmaMuT**2
    d2 = p3c1*sigmaMuT**2
    P_13 = (1/8)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-3/2) \
        *sigmaT**(-3)
    
    #################
    d0 = p4c0*sigmaMuT
    d1 = p4c1*sigmaMuT
    d2 = p4c2*sigmaMuT
    P_14 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-1/2) \
        *sigmaT**(-1)
    
    #################
    d0 = p5c0*sigmaMuT
    d1 = p5c1*sigmaMuT
    d2 = p5c2*sigmaMuT
    P_15 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-1/2) \
        *sigmaT**(-1)
    
    #################
    d0 = p6c0*sigmaMuT
    d1 = p6c1*sigmaMuT
    d2 = p6c2*sigmaMuT
    P_16 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuT**2*sigmaT**(-2))**(-1/2) \
        *sigmaT**(-1)
    
    #################
    d0 = p7c0
    d1 = p7c1
    d2 = p7c2
    d3 = p7c3
    P_17 = (1/16)*(((-1)*a2)**(-7/2)*Constanterf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p8c0
    d1 = p8c1
    d2 = p8c2
    d3 = p8c3
    P_18 = (1/16)*(((-1)*a2)**(-7/2)*Constanterf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p9c0
    d1 = p9c1
    d2 = p9c2
    d3 = p9c3
    P_19 = (1/16)*(((-1)*a2)**(-7/2)*Constanterf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p10c0
    d1 = p10c1
    d2 = p10c2
    d3 = p10c3
    P_20 = (1/16)*(((-1)*a2)**(-7/2)*Constanterf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = (-1)*muT**3*p1c0*sigmaMuT**4+3*muT*p1c0*sigmaMuT**4*(sigmaMuT**2+ \
        sigmaT**2)
    d1 = 3*muT**2*p1c0*sigmaMuT**4+(-3)*p1c0*sigmaMuT**4*(sigmaMuT**2+ \
        sigmaT**2)
    d2 = (-3)*muT*p1c0*sigmaMuT**4
    d3 = p1c0*sigmaMuT**4
    P_21 = (1/192)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuT**2*sigmaT**(-2))**(-1/2)*sigmaT**(-1)*(sigmaMuT**2+ \
        sigmaT**2)**(-3)
    
    #################
    d0 = muT**2*p2c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT+ \
        (-1)*p2c0*sigmaMuT**5*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT+( \
        -1)*p2c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT**3
    d1 = (-2)*muT*p2c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT+muT**2*p2c1*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT+(-1)*p2c1*sigmaMuT**5*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT+(-1)*p2c1*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT**3
    d2 = p2c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT+(-2)* \
        muT*p2c1*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT
    d3 = p2c1*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT
    P_22 = (1/48)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*( \
        sigmaMuT**2+sigmaT**2)**(-3)
    
    #################
    d0 = muT**2*p3c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT+ \
        (-1)*p3c0*sigmaMuT**5*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT+( \
        -1)*p3c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT**3
    d1 = (-2)*muT*p3c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT+muT**2*p3c1*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT+(-1)*p3c1*sigmaMuT**5*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT+(-1)*p3c1*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)* \
        sigmaT**3
    d2 = p3c0*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT+(-2)* \
        muT*p3c1*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT
    d3 = p3c1*sigmaMuT**3*(1+sigmaMuT**2*sigmaT**(-2))**(1/2)*sigmaT
    P_23 = (1/48)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*( \
        sigmaMuT**2+sigmaT**2)**(-3)
    
    #################
    d0 = (-1)*muT*p4c0*sigmaMuT**2
    d1 = p4c0*sigmaMuT**2+(-1)*muT*p4c1*sigmaMuT**2
    d2 = p4c1*sigmaMuT**2+(-1)*muT*p4c2*sigmaMuT**2
    d3 = p4c2*sigmaMuT**2
    P_24 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuT**2*sigmaT**(-2))**(-3/2)*sigmaT**(-3)
    
    #################
    d0 = (-1)*muT*p5c0*sigmaMuT**2
    d1 = p5c0*sigmaMuT**2+(-1)*muT*p5c1*sigmaMuT**2
    d2 = p5c1*sigmaMuT**2+(-1)*muT*p5c2*sigmaMuT**2
    d3 = p5c2*sigmaMuT**2
    P_25 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuT**2*sigmaT**(-2))**(-3/2)*sigmaT**(-3)
    
    #################
    d0 = (-1)*muT*p6c0*sigmaMuT**2
    d1 = p6c0*sigmaMuT**2+(-1)*muT*p6c1*sigmaMuT**2
    d2 = p6c1*sigmaMuT**2+(-1)*muT*p6c2*sigmaMuT**2
    d3 = p6c2*sigmaMuT**2
    P_26 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuT**2*sigmaT**(-2))**(-3/2)*sigmaT**(-3)
    
    #################
    d0 = p7c0*sigmaMuT
    d1 = p7c1*sigmaMuT
    d2 = p7c2*sigmaMuT
    d3 = p7c3*sigmaMuT
    P_27 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuT**2*sigmaT**(-2))**(-1/2)*sigmaT**(-1)
    
    #################
    d0 = p8c0*sigmaMuT
    d1 = p8c1*sigmaMuT
    d2 = p8c2*sigmaMuT
    d3 = p8c3*sigmaMuT
    P_28 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuT**2*sigmaT**(-2))**(-1/2)*sigmaT**(-1)
    
    #################
    d0 = p9c0*sigmaMuT
    d1 = p9c1*sigmaMuT
    d2 = p9c2*sigmaMuT
    d3 = p9c3*sigmaMuT
    P_29 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuT**2*sigmaT**(-2))**(-1/2)*sigmaT**(-1)
    
    #################
    d0 = p10c0*sigmaMuT
    d1 = p10c1*sigmaMuT
    d2 = p10c2*sigmaMuT
    d3 = p10c3*sigmaMuT
    P_30 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
    b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
    sigmaMuT**2*sigmaT**(-2))**(-1/2)*sigmaT**(-1)
    
    #################
    d0 = p11c0
    d1 = p11c1
    d2 = p11c2
    d3 = p11c3
    d4 = p11c4
    P_31 = (1/32)*(((-1)*a2)**(-9/2)*Constanterf*((-16)*a2**4*d0+8*a2**3*( \
        b2*d1+d2)+(-1)*b2**4*d4+(-4)*a2**2*(b2**2*d2+3*b2*d3+3*d4)+2* \
        a2*b2**2*(b2*d3+6*d4))+2*a2**(-4)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-7/2)*(8*a2**3*d1*((-1)+2*a2*sigmaPhi**2)**3+(-4)* \
        a2**2*d2*(1+(-2)*a2*sigmaPhi**2)**2*(2*a2*muPhi+b2*((-1)+4*a2* \
        sigmaPhi**2))+(-2)*a2*d3*(1+(-2)*a2*sigmaPhi**2)*(b2**2+(-24)* \
        a2**3*sigmaPhi**4+(-2)*a2*(2+b2*muPhi+3*b2**2*sigmaPhi**2)+4* \
        a2**2*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))+d4*(b2**3+96*a2**4*sigmaPhi**4*(muPhi+2*b2* \
        sigmaPhi**2)+(-2)*a2*b2*(5+b2*muPhi+4*b2**2*sigmaPhi**2)+4* \
        a2**2*(3*muPhi+b2*muPhi**2+19*b2*sigmaPhi**2+4*b2**2*muPhi* \
        sigmaPhi**2+6*b2**3*sigmaPhi**4)+(-8)*a2**3*(muPhi**3+9*muPhi* \
        sigmaPhi**2+4*b2*muPhi**2*sigmaPhi**2+26*b2*sigmaPhi**4+6*b2**2* \
        muPhi*sigmaPhi**4+4*b2**3*sigmaPhi**6))))
    
    #################
    d0 = p12c0
    d1 = p12c1
    d2 = p12c2
    d3 = p12c3
    d4 = p12c4
    P_32 = (1/32)*(((-1)*a2)**(-9/2)*Constanterf*((-16)*a2**4*d0+8*a2**3*( \
        b2*d1+d2)+(-1)*b2**4*d4+(-4)*a2**2*(b2**2*d2+3*b2*d3+3*d4)+2* \
        a2*b2**2*(b2*d3+6*d4))+2*a2**(-4)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-7/2)*(8*a2**3*d1*((-1)+2*a2*sigmaPhi**2)**3+(-4)* \
        a2**2*d2*(1+(-2)*a2*sigmaPhi**2)**2*(2*a2*muPhi+b2*((-1)+4*a2* \
        sigmaPhi**2))+(-2)*a2*d3*(1+(-2)*a2*sigmaPhi**2)*(b2**2+(-24)* \
        a2**3*sigmaPhi**4+(-2)*a2*(2+b2*muPhi+3*b2**2*sigmaPhi**2)+4* \
        a2**2*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))+d4*(b2**3+96*a2**4*sigmaPhi**4*(muPhi+2*b2* \
        sigmaPhi**2)+(-2)*a2*b2*(5+b2*muPhi+4*b2**2*sigmaPhi**2)+4* \
        a2**2*(3*muPhi+b2*muPhi**2+19*b2*sigmaPhi**2+4*b2**2*muPhi* \
        sigmaPhi**2+6*b2**3*sigmaPhi**4)+(-8)*a2**3*(muPhi**3+9*muPhi* \
        sigmaPhi**2+4*b2*muPhi**2*sigmaPhi**2+26*b2*sigmaPhi**4+6*b2**2* \
        muPhi*sigmaPhi**4+4*b2**3*sigmaPhi**6))))
    
    #################
    d0 = p13c0
    d1 = p13c1
    d2 = p13c2
    d3 = p13c3
    d4 = p13c4
    P_33 = (1/32)*(((-1)*a2)**(-9/2)*Constanterf*((-16)*a2**4*d0+8*a2**3*( \
        b2*d1+d2)+(-1)*b2**4*d4+(-4)*a2**2*(b2**2*d2+3*b2*d3+3*d4)+2* \
        a2*b2**2*(b2*d3+6*d4))+2*a2**(-4)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-7/2)*(8*a2**3*d1*((-1)+2*a2*sigmaPhi**2)**3+(-4)* \
        a2**2*d2*(1+(-2)*a2*sigmaPhi**2)**2*(2*a2*muPhi+b2*((-1)+4*a2* \
        sigmaPhi**2))+(-2)*a2*d3*(1+(-2)*a2*sigmaPhi**2)*(b2**2+(-24)* \
        a2**3*sigmaPhi**4+(-2)*a2*(2+b2*muPhi+3*b2**2*sigmaPhi**2)+4* \
        a2**2*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))+d4*(b2**3+96*a2**4*sigmaPhi**4*(muPhi+2*b2* \
        sigmaPhi**2)+(-2)*a2*b2*(5+b2*muPhi+4*b2**2*sigmaPhi**2)+4* \
        a2**2*(3*muPhi+b2*muPhi**2+19*b2*sigmaPhi**2+4*b2**2*muPhi* \
        sigmaPhi**2+6*b2**3*sigmaPhi**4)+(-8)*a2**3*(muPhi**3+9*muPhi* \
        sigmaPhi**2+4*b2*muPhi**2*sigmaPhi**2+26*b2*sigmaPhi**4+6*b2**2* \
        muPhi*sigmaPhi**4+4*b2**3*sigmaPhi**6))))
    
    #################
    d0 = p14c0
    d1 = p14c1
    d2 = p14c2
    d3 = p14c3
    d4 = p14c4
    P_34 = (1/32)*(((-1)*a2)**(-9/2)*Constanterf*((-16)*a2**4*d0+8*a2**3*( \
        b2*d1+d2)+(-1)*b2**4*d4+(-4)*a2**2*(b2**2*d2+3*b2*d3+3*d4)+2* \
        a2*b2**2*(b2*d3+6*d4))+2*a2**(-4)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-7/2)*(8*a2**3*d1*((-1)+2*a2*sigmaPhi**2)**3+(-4)* \
        a2**2*d2*(1+(-2)*a2*sigmaPhi**2)**2*(2*a2*muPhi+b2*((-1)+4*a2* \
        sigmaPhi**2))+(-2)*a2*d3*(1+(-2)*a2*sigmaPhi**2)*(b2**2+(-24)* \
        a2**3*sigmaPhi**4+(-2)*a2*(2+b2*muPhi+3*b2**2*sigmaPhi**2)+4* \
        a2**2*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))+d4*(b2**3+96*a2**4*sigmaPhi**4*(muPhi+2*b2* \
        sigmaPhi**2)+(-2)*a2*b2*(5+b2*muPhi+4*b2**2*sigmaPhi**2)+4* \
        a2**2*(3*muPhi+b2*muPhi**2+19*b2*sigmaPhi**2+4*b2**2*muPhi* \
        sigmaPhi**2+6*b2**3*sigmaPhi**4)+(-8)*a2**3*(muPhi**3+9*muPhi* \
        sigmaPhi**2+4*b2*muPhi**2*sigmaPhi**2+26*b2*sigmaPhi**4+6*b2**2* \
        muPhi*sigmaPhi**4+4*b2**3*sigmaPhi**6))))
    
    #################
    d0 = p15c0
    d1 = p15c1
    d2 = p15c2
    d3 = p15c3
    d4 = p15c4
    P_35 = (1/32)*(((-1)*a2)**(-9/2)*Constanterf*((-16)*a2**4*d0+8*a2**3*( \
        b2*d1+d2)+(-1)*b2**4*d4+(-4)*a2**2*(b2**2*d2+3*b2*d3+3*d4)+2* \
        a2*b2**2*(b2*d3+6*d4))+2*a2**(-4)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-7/2)*(8*a2**3*d1*((-1)+2*a2*sigmaPhi**2)**3+(-4)* \
        a2**2*d2*(1+(-2)*a2*sigmaPhi**2)**2*(2*a2*muPhi+b2*((-1)+4*a2* \
        sigmaPhi**2))+(-2)*a2*d3*(1+(-2)*a2*sigmaPhi**2)*(b2**2+(-24)* \
        a2**3*sigmaPhi**4+(-2)*a2*(2+b2*muPhi+3*b2**2*sigmaPhi**2)+4* \
        a2**2*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))+d4*(b2**3+96*a2**4*sigmaPhi**4*(muPhi+2*b2* \
        sigmaPhi**2)+(-2)*a2*b2*(5+b2*muPhi+4*b2**2*sigmaPhi**2)+4* \
        a2**2*(3*muPhi+b2*muPhi**2+19*b2*sigmaPhi**2+4*b2**2*muPhi* \
        sigmaPhi**2+6*b2**3*sigmaPhi**4)+(-8)*a2**3*(muPhi**3+9*muPhi* \
        sigmaPhi**2+4*b2*muPhi**2*sigmaPhi**2+26*b2*sigmaPhi**4+6*b2**2* \
        muPhi*sigmaPhi**4+4*b2**3*sigmaPhi**6))))
        
    # store to output
    P_output = zeros((n_sites, 35))
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
    P_output[:,15] = P_16
    P_output[:,16] = P_17
    P_output[:,17] = P_18
    P_output[:,18] = P_19
    P_output[:,19] = P_20
    P_output[:,20] = P_21
    P_output[:,21] = P_22
    P_output[:,22] = P_23
    P_output[:,23] = P_24
    P_output[:,24] = P_25
    P_output[:,25] = P_26
    P_output[:,26] = P_27
    P_output[:,27] = P_28
    P_output[:,28] = P_29
    P_output[:,29] = P_30
    P_output[:,30] = P_31
    P_output[:,31] = P_32
    P_output[:,32] = P_33
    P_output[:,33] = P_34
    P_output[:,34] = P_35
    
    # return
    return P_output
