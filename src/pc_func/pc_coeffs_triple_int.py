from numpy import exp, pi, zeros
from .pc_util import erf
from numba import njit, float64
from .pc_coeffs_second_int import pc_coeffs_second_int

@njit(
    float64[:,:](
        float64[:],float64[:],float64[:],
        float64[:],float64[:],float64[:],float64[:],
        float64[:],float64[:],float64[:],float64[:],
        float64[:],float64[:],float64[:]
    ),
    fastmath = True,
    cache = True
)
def pc_coeffs_triple_int(
    muY, sigmaMuY, sigmaY,
    amuZ, bmuZ, sigmaMuZ, sigmaZ,
    amuT, bmuT, sigmaMuT, sigmaT,
    muV, sigmaMuV, sigmaV
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
    
    ## PC Coefficients from Second Integration!!
    P_output_2 = pc_coeffs_second_int(
        muY, sigmaMuY, sigmaY, \
        amuZ, bmuZ, sigmaMuZ, sigmaZ, \
        amuT, bmuT, sigmaMuT, sigmaT
    )
    # pull results
    p1c0 = zeros((1,n_sites))
    p2c0 = zeros((1,n_sites))
    p2c1 = zeros((1,n_sites))
    p3c0 = zeros((1,n_sites))
    p3c1 = zeros((1,n_sites))
    p4c0 = zeros((1,n_sites))
    p4c1 = zeros((1,n_sites))
    p5c0 = zeros((1,n_sites))
    p5c1 = zeros((1,n_sites))
    p5c2 = zeros((1,n_sites))
    p6c0 = zeros((1,n_sites))
    p6c1 = zeros((1,n_sites))
    p6c2 = zeros((1,n_sites))
    p7c0 = zeros((1,n_sites))
    p7c1 = zeros((1,n_sites))
    p7c2 = zeros((1,n_sites))
    p8c0 = zeros((1,n_sites))
    p8c1 = zeros((1,n_sites))
    p8c2 = zeros((1,n_sites))
    p9c0 = zeros((1,n_sites))
    p9c1 = zeros((1,n_sites))
    p9c2 = zeros((1,n_sites))
    p10c0 = zeros((1,n_sites))
    p10c1 = zeros((1,n_sites))
    p10c2 = zeros((1,n_sites))
    p11c0 = zeros((1,n_sites))
    p11c1 = zeros((1,n_sites))
    p11c2 = zeros((1,n_sites))
    p11c3 = zeros((1,n_sites))
    p12c0 = zeros((1,n_sites))
    p12c1 = zeros((1,n_sites))
    p12c2 = zeros((1,n_sites))
    p12c3 = zeros((1,n_sites))
    p13c0 = zeros((1,n_sites))
    p13c1 = zeros((1,n_sites))
    p13c2 = zeros((1,n_sites))
    p13c3 = zeros((1,n_sites))
    p14c0 = zeros((1,n_sites))
    p14c1 = zeros((1,n_sites))
    p14c2 = zeros((1,n_sites))
    p14c3 = zeros((1,n_sites))
    p15c0 = zeros((1,n_sites))
    p15c1 = zeros((1,n_sites))
    p15c2 = zeros((1,n_sites))
    p15c3 = zeros((1,n_sites))
    p16c0 = zeros((1,n_sites))
    p16c1 = zeros((1,n_sites))
    p16c2 = zeros((1,n_sites))
    p16c3 = zeros((1,n_sites))
    p17c0 = zeros((1,n_sites))
    p17c1 = zeros((1,n_sites))
    p17c2 = zeros((1,n_sites))
    p17c3 = zeros((1,n_sites))
    p18c0 = zeros((1,n_sites))
    p18c1 = zeros((1,n_sites))
    p18c2 = zeros((1,n_sites))
    p18c3 = zeros((1,n_sites))
    p19c0 = zeros((1,n_sites))
    p19c1 = zeros((1,n_sites))
    p19c2 = zeros((1,n_sites))
    p19c3 = zeros((1,n_sites))
    p20c0 = zeros((1,n_sites))
    p20c1 = zeros((1,n_sites))
    p20c2 = zeros((1,n_sites))
    p20c3 = zeros((1,n_sites))
    p21c0 = zeros((1,n_sites))
    p21c1 = zeros((1,n_sites))
    p21c2 = zeros((1,n_sites))
    p21c3 = zeros((1,n_sites))
    p21c4 = zeros((1,n_sites))
    p22c0 = zeros((1,n_sites))
    p22c1 = zeros((1,n_sites))
    p22c2 = zeros((1,n_sites))
    p22c3 = zeros((1,n_sites))
    p22c4 = zeros((1,n_sites))
    p23c0 = zeros((1,n_sites))
    p23c1 = zeros((1,n_sites))
    p23c2 = zeros((1,n_sites))
    p23c3 = zeros((1,n_sites))
    p23c4 = zeros((1,n_sites))
    p24c0 = zeros((1,n_sites))
    p24c1 = zeros((1,n_sites))
    p24c2 = zeros((1,n_sites))
    p24c3 = zeros((1,n_sites))
    p24c4 = zeros((1,n_sites))
    p25c0 = zeros((1,n_sites))
    p25c1 = zeros((1,n_sites))
    p25c2 = zeros((1,n_sites))
    p25c3 = zeros((1,n_sites))
    p25c4 = zeros((1,n_sites))
    p26c0 = zeros((1,n_sites))
    p26c1 = zeros((1,n_sites))
    p26c2 = zeros((1,n_sites))
    p26c3 = zeros((1,n_sites))
    p26c4 = zeros((1,n_sites))
    p27c0 = zeros((1,n_sites))
    p27c1 = zeros((1,n_sites))
    p27c2 = zeros((1,n_sites))
    p27c3 = zeros((1,n_sites))
    p27c4 = zeros((1,n_sites))
    p28c0 = zeros((1,n_sites))
    p28c1 = zeros((1,n_sites))
    p28c2 = zeros((1,n_sites))
    p28c3 = zeros((1,n_sites))
    p28c4 = zeros((1,n_sites))
    p29c0 = zeros((1,n_sites))
    p29c1 = zeros((1,n_sites))
    p29c2 = zeros((1,n_sites))
    p29c3 = zeros((1,n_sites))
    p29c4 = zeros((1,n_sites))
    p30c0 = zeros((1,n_sites))
    p30c1 = zeros((1,n_sites))
    p30c2 = zeros((1,n_sites))
    p30c3 = zeros((1,n_sites))
    p30c4 = zeros((1,n_sites))
    p31c0 = zeros((1,n_sites))
    p31c1 = zeros((1,n_sites))
    p31c2 = zeros((1,n_sites))
    p31c3 = zeros((1,n_sites))
    p31c4 = zeros((1,n_sites))
    p32c0 = zeros((1,n_sites))
    p32c1 = zeros((1,n_sites))
    p32c2 = zeros((1,n_sites))
    p32c3 = zeros((1,n_sites))
    p32c4 = zeros((1,n_sites))
    p33c0 = zeros((1,n_sites))
    p33c1 = zeros((1,n_sites))
    p33c2 = zeros((1,n_sites))
    p33c3 = zeros((1,n_sites))
    p33c4 = zeros((1,n_sites))
    p34c0 = zeros((1,n_sites))
    p34c1 = zeros((1,n_sites))
    p34c2 = zeros((1,n_sites))
    p34c3 = zeros((1,n_sites))
    p34c4 = zeros((1,n_sites))
    p35c0 = zeros((1,n_sites))
    p35c1 = zeros((1,n_sites))
    p35c2 = zeros((1,n_sites))
    p35c3 = zeros((1,n_sites))
    p35c4 = zeros((1,n_sites))
    
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
    
    #
    ## Total Constant
    #
    a1 = (1/4)*((-2)*(sigmaMuV**2+sigmaV**2)**(-1)+(-2)*(sigmaMuT**2+ \
        sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+ \
        sigmaZ**2))**(-1))
    b1 = muV*(sigmaMuV**2+sigmaV**2)**(-1)+(bmuT+amuT*(bmuZ+amuZ*muY))*( \
        sigmaMuT**2+sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+ \
        sigmaY**2)+sigmaZ**2))**(-1)
    c1 = (1/4)*((-2)*muV**2*sigmaV**(-2)+2*muV**2*sigmaMuV**2*sigmaV**(-2) \
        *(sigmaMuV**2+sigmaV**2)**(-1)+(-2)*(bmuT+amuT*(bmuZ+amuZ*muY))**2* \
        (sigmaMuT**2+sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+ \
        sigmaY**2)+sigmaZ**2))**(-1))
    a2 = (-1/2)*(sigmaMuT**2+sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*( \
        sigmaMuY**2+sigmaY**2)+sigmaZ**2))**(-1)
    b2 = (bmuT+amuT*(bmuZ+amuZ*muY))*(sigmaMuT**2+sigmaT**2+amuT**2*( \
        sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2))**(-1)
    c2 = (-1/2)*(bmuT+amuT*(bmuZ+amuZ*muY))**2*(sigmaMuT**2+sigmaT**2+ \
        amuT**2*(sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2))**(-1)
    muPhi = muV
    sigmaPhi = (sigmaMuV**2+sigmaV**2)**(1/2)
    ConstantExp1 = exp(1)**((-1/4)*a1**(-1)*b1**2+c1)*pi**(1/2)
    ConstantExp2 = exp(1)**((2+(-4)*a2*sigmaPhi**2)**(-1)*(2*c2+2*b2*muPhi+2*a2* \
        muPhi**2+b2**2*sigmaPhi**2+(-4)*a2*c2*sigmaPhi**2))
    ConstantErf = exp(1)**((-1/4)*a2**(-1)*b2**2+c2)*pi**(1/2)*((-1)+erf((1/2)*a2**( \
        -1)*(b2+2*a2*muPhi)*((-1)*a2**(-1)+2*sigmaPhi**2)**(-1/2)))
        
    #
    ## Polynomial Coefficients
    #################
    d0 = p1c0
    P_1 = (-1/2)*((-1)*a2)**(-1/2)*ConstantErf*d0
    
    #################
    d0 = p1c0*sigmaMuV
    P_2 = ((-1)*a1)**(-1/2)*ConstantExp1*d0*(2*pi)**(-1/2)*(1+sigmaMuV**2* \
        sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p2c0
    d1 = p2c1
    P_3 = (1/4)*(((-1)*a2)**(-3/2)*ConstantErf*(2*a2*d0+(-1)*b2*d1)+(-2)* \
        a2**(-1)*ConstantExp2*d1*(1+(-2)*a2*sigmaPhi**2)**(-1/2))
    
    #################
    d0 = p3c0
    d1 = p3c1
    P_4 = (1/4)*(((-1)*a2)**(-3/2)*ConstantErf*(2*a2*d0+(-1)*b2*d1)+(-2)* \
        a2**(-1)*ConstantExp2*d1*(1+(-2)*a2*sigmaPhi**2)**(-1/2))
    
    #################
    d0 = p4c0
    d1 = p4c1
    P_5 = (1/4)*(((-1)*a2)**(-3/2)*ConstantErf*(2*a2*d0+(-1)*b2*d1)+(-2)* \
        a2**(-1)*ConstantExp2*d1*(1+(-2)*a2*sigmaPhi**2)**(-1/2))
    
    #################
    d0 = (-1)*muV*p1c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV
    d1 = p1c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_6 = (1/4)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*(2*pi) \
        **(-1/2)*(sigmaMuV**2+sigmaV**2)**(-2)
    
    #################
    d0 = p2c0*sigmaMuV
    d1 = p2c1*sigmaMuV
    P_7 = (1/2)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*(2*pi) \
        **(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p3c0*sigmaMuV
    d1 = p3c1*sigmaMuV
    P_8 = (1/2)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*(2*pi) \
        **(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p4c0*sigmaMuV
    d1 = p4c1*sigmaMuV
    P_9 = (1/2)*((-1)*a1)**(-3/2)*ConstantExp1*((-2)*a1*d0+b1*d1)*(2*pi) \
        **(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p5c0
    d1 = p5c1
    d2 = p5c2
    P_10 = (-1/8)*((-1)*a2)**(-5/2)*ConstantErf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = p6c0
    d1 = p6c1
    d2 = p6c2
    P_11 = (-1/8)*((-1)*a2)**(-5/2)*ConstantErf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = p7c0
    d1 = p7c1
    d2 = p7c2
    P_12 = (-1/8)*((-1)*a2)**(-5/2)*ConstantErf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = p8c0
    d1 = p8c1
    d2 = p8c2
    P_13 = (-1/8)*((-1)*a2)**(-5/2)*ConstantErf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = p9c0
    d1 = p9c1
    d2 = p9c2
    P_14 = (-1/8)*((-1)*a2)**(-5/2)*ConstantErf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = p10c0
    d1 = p10c1
    d2 = p10c2
    P_15 = (-1/8)*((-1)*a2)**(-5/2)*ConstantErf*(4*a2**2*d0+b2**2*d2+(-2)* \
        a2*(b2*d1+d2))+(1/4)*a2**(-2)*ConstantExp2*(1+(-2)*a2* \
        sigmaPhi**2)**(-3/2)*(b2*d2+4*a2**2*d1*sigmaPhi**2+(-2)*a2*(d1+ \
        d2*(muPhi+2*b2*sigmaPhi**2)))
    
    #################
    d0 = muV**2*p1c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+ \
        (-1)*p1c0*sigmaMuV**5*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+( \
        -1)*p1c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV**3
    d1 = (-2)*muV*p1c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV
    d2 = p1c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_16 = (1/24)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(sigmaMuV**2+sigmaV**2)**(-3)
    
    #################
    d0 = (-1)*muV*p2c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV
    d1 = p2c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+(-1)* \
        muV*p2c1*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    d2 = p2c1*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_17 = (1/8)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(sigmaMuV**2+sigmaV**2)**(-2)
    
    #################
    d0 = (-1)*muV*p3c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV
    d1 = p3c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+(-1)* \
        muV*p3c1*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    d2 = p3c1*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_18 = (1/8)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(sigmaMuV**2+sigmaV**2)**(-2)
    
    #################
    d0 = (-1)*muV*p4c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV
    d1 = p4c0*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+(-1)* \
        muV*p4c1*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    d2 = p4c1*sigmaMuV**2*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_19 = (1/8)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(sigmaMuV**2+sigmaV**2)**(-2)
    
    #################
    d0 = p5c0*sigmaMuV
    d1 = p5c1*sigmaMuV
    d2 = p5c2*sigmaMuV
    P_20 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2) \
        *sigmaV**(-1)
    
    #################
    d0 = p6c0*sigmaMuV
    d1 = p6c1*sigmaMuV
    d2 = p6c2*sigmaMuV
    P_21 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2) \
        *sigmaV**(-1)
    
    #################
    d0 = p7c0*sigmaMuV
    d1 = p7c1*sigmaMuV
    d2 = p7c2*sigmaMuV
    P_22 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2) \
        *sigmaV**(-1)
    
    #################
    d0 = p8c0*sigmaMuV
    d1 = p8c1*sigmaMuV
    d2 = p8c2*sigmaMuV
    P_23 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2) \
        *sigmaV**(-1)
    
    #################
    d0 = p9c0*sigmaMuV
    d1 = p9c1*sigmaMuV
    d2 = p9c2*sigmaMuV
    P_24 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2) \
        *sigmaV**(-1)
    
    #################
    d0 = p10c0*sigmaMuV
    d1 = p10c1*sigmaMuV
    d2 = p10c2*sigmaMuV
    P_25 = (1/4)*((-1)*a1)**(-5/2)*ConstantExp1*(4*a1**2*d0+b1**2*d2+(-2)* \
        a1*(b1*d1+d2))*(2*pi)**(-1/2)*(1+sigmaMuV**2*sigmaV**(-2))**(-1/2) \
        *sigmaV**(-1)
    
    #################
    d0 = p11c0
    d1 = p11c1
    d2 = p11c2
    d3 = p11c3
    P_26 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p12c0
    d1 = p12c1
    d2 = p12c2
    d3 = p12c3
    P_27 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p13c0
    d1 = p13c1
    d2 = p13c2
    d3 = p13c3
    P_28 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p14c0
    d1 = p14c1
    d2 = p14c2
    d3 = p14c3
    P_29 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p15c0
    d1 = p15c1
    d2 = p15c2
    d3 = p15c3
    P_30 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p16c0
    d1 = p16c1
    d2 = p16c2
    d3 = p16c3
    P_31 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p17c0
    d1 = p17c1
    d2 = p17c2
    d3 = p17c3
    P_32 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p18c0
    d1 = p18c1
    d2 = p18c2
    d3 = p18c3
    P_33 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p19c0
    d1 = p19c1
    d2 = p19c2
    d3 = p19c3
    P_34 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = p20c0
    d1 = p20c1
    d2 = p20c2
    d3 = p20c3
    P_35 = (1/16)*(((-1)*a2)**(-7/2)*ConstantErf*(8*a2**3*d0+(-4)*a2**2*( \
        b2*d1+d2)+(-1)*b2**3*d3+2*a2*b2*(b2*d2+3*d3))+(-2)*a2**(-3)* \
        ConstantExp2*(1+(-2)*a2*sigmaPhi**2)**(-5/2)*(b2**2*d3+16*a2**4* \
        d1*sigmaPhi**4+(-2)*a2*(2*d3+b2*(d2+d3*muPhi)+3*b2**2*d3* \
        sigmaPhi**2)+(-8)*a2**3*sigmaPhi**2*(2*d1+3*d3*sigmaPhi**2+d2*( \
        muPhi+2*b2*sigmaPhi**2))+4*a2**2*(d1+d2*(muPhi+3*b2*sigmaPhi**2)+ \
        d3*(muPhi**2+5*sigmaPhi**2+3*b2*muPhi*sigmaPhi**2+3*b2**2* \
        sigmaPhi**4))))
    
    #################
    d0 = (-1)*muV**3*p1c0*sigmaMuV**4+3*muV*p1c0*sigmaMuV**4*(sigmaMuV**2+ \
        sigmaV**2)
    d1 = 3*muV**2*p1c0*sigmaMuV**4+(-3)*p1c0*sigmaMuV**4*(sigmaMuV**2+ \
        sigmaV**2)
    d2 = (-3)*muV*p1c0*sigmaMuV**4
    d3 = p1c0*sigmaMuV**4
    P_36 = (1/192)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1/2)*(2+2* \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)*(sigmaMuV**2+ \
        sigmaV**2)**(-3)
    
    #################
    d0 = muV**2*p2c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+ \
        (-1)*p2c0*sigmaMuV**5*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+( \
        -1)*p2c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV**3
    d1 = (-2)*muV*p2c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+muV**2*p2c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+(-1)*p2c1*sigmaMuV**5*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+(-1)*p2c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV**3
    d2 = p2c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+(-2)* \
        muV*p2c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    d3 = p2c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_37 = (1/48)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*( \
        sigmaMuV**2+sigmaV**2)**(-3)
    
    #################
    d0 = muV**2*p3c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+ \
        (-1)*p3c0*sigmaMuV**5*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+( \
        -1)*p3c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV**3
    d1 = (-2)*muV*p3c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+muV**2*p3c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+(-1)*p3c1*sigmaMuV**5*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+(-1)*p3c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV**3
    d2 = p3c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+(-2)* \
        muV*p3c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    d3 = p3c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_38 = (1/48)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*( \
        sigmaMuV**2+sigmaV**2)**(-3)
    
    #################
    d0 = muV**2*p4c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+ \
        (-1)*p4c0*sigmaMuV**5*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+( \
        -1)*p4c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV**3
    d1 = (-2)*muV*p4c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+muV**2*p4c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+(-1)*p4c1*sigmaMuV**5*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV+(-1)*p4c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)* \
        sigmaV**3
    d2 = p4c0*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV+(-2)* \
        muV*p4c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    d3 = p4c1*sigmaMuV**3*(1+sigmaMuV**2*sigmaV**(-2))**(1/2)*sigmaV
    P_39 = (1/48)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*( \
        sigmaMuV**2+sigmaV**2)**(-3)
    
    #################
    d0 = (-1)*muV*p5c0*sigmaMuV**2
    d1 = p5c0*sigmaMuV**2+(-1)*muV*p5c1*sigmaMuV**2
    d2 = p5c1*sigmaMuV**2+(-1)*muV*p5c2*sigmaMuV**2
    d3 = p5c2*sigmaMuV**2
    P_40 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1/2)*(2+2* \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)*(sigmaMuV**2+ \
        sigmaV**2)**(-1)
    
    #################
    d0 = (-1)*muV*p6c0*sigmaMuV**2
    d1 = p6c0*sigmaMuV**2+(-1)*muV*p6c1*sigmaMuV**2
    d2 = p6c1*sigmaMuV**2+(-1)*muV*p6c2*sigmaMuV**2
    d3 = p6c2*sigmaMuV**2
    P_41 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1/2)*(2+2* \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)*(sigmaMuV**2+ \
        sigmaV**2)**(-1)
    
    #################
    d0 = (-1)*muV*p7c0*sigmaMuV**2
    d1 = p7c0*sigmaMuV**2+(-1)*muV*p7c1*sigmaMuV**2
    d2 = p7c1*sigmaMuV**2+(-1)*muV*p7c2*sigmaMuV**2
    d3 = p7c2*sigmaMuV**2
    P_42 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1/2)*(2+2* \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)*(sigmaMuV**2+ \
        sigmaV**2)**(-1)
    
    #################
    d0 = (-1)*muV*p8c0*sigmaMuV**2
    d1 = p8c0*sigmaMuV**2+(-1)*muV*p8c1*sigmaMuV**2
    d2 = p8c1*sigmaMuV**2+(-1)*muV*p8c2*sigmaMuV**2
    d3 = p8c2*sigmaMuV**2
    P_43 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1/2)*(2+2* \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)*(sigmaMuV**2+ \
        sigmaV**2)**(-1)
    
    #################
    d0 = (-1)*muV*p9c0*sigmaMuV**2
    d1 = p9c0*sigmaMuV**2+(-1)*muV*p9c1*sigmaMuV**2
    d2 = p9c1*sigmaMuV**2+(-1)*muV*p9c2*sigmaMuV**2
    d3 = p9c2*sigmaMuV**2
    P_44 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1/2)*(2+2* \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)*(sigmaMuV**2+ \
        sigmaV**2)**(-1)
    
    #################
    d0 = (-1)*muV*p10c0*sigmaMuV**2
    d1 = p10c0*sigmaMuV**2+(-1)*muV*p10c1*sigmaMuV**2
    d2 = p10c1*sigmaMuV**2+(-1)*muV*p10c2*sigmaMuV**2
    d3 = p10c2*sigmaMuV**2
    P_45 = (1/16)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*pi**(-1/2)*(2+2* \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)*(sigmaMuV**2+ \
        sigmaV**2)**(-1)
    
    #################
    d0 = p11c0*sigmaMuV
    d1 = p11c1*sigmaMuV
    d2 = p11c2*sigmaMuV
    d3 = p11c3*sigmaMuV
    P_46 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p12c0*sigmaMuV
    d1 = p12c1*sigmaMuV
    d2 = p12c2*sigmaMuV
    d3 = p12c3*sigmaMuV
    P_47 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p13c0*sigmaMuV
    d1 = p13c1*sigmaMuV
    d2 = p13c2*sigmaMuV
    d3 = p13c3*sigmaMuV
    P_48 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p14c0*sigmaMuV
    d1 = p14c1*sigmaMuV
    d2 = p14c2*sigmaMuV
    d3 = p14c3*sigmaMuV
    P_49 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p15c0*sigmaMuV
    d1 = p15c1*sigmaMuV
    d2 = p15c2*sigmaMuV
    d3 = p15c3*sigmaMuV
    P_50 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p16c0*sigmaMuV
    d1 = p16c1*sigmaMuV
    d2 = p16c2*sigmaMuV
    d3 = p16c3*sigmaMuV
    P_51 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p17c0*sigmaMuV
    d1 = p17c1*sigmaMuV
    d2 = p17c2*sigmaMuV
    d3 = p17c3*sigmaMuV
    P_52 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p18c0*sigmaMuV
    d1 = p18c1*sigmaMuV
    d2 = p18c2*sigmaMuV
    d3 = p18c3*sigmaMuV
    P_53 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p19c0*sigmaMuV
    d1 = p19c1*sigmaMuV
    d2 = p19c2*sigmaMuV
    d3 = p19c3*sigmaMuV
    P_54 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p20c0*sigmaMuV
    d1 = p20c1*sigmaMuV
    d2 = p20c2*sigmaMuV
    d3 = p20c3*sigmaMuV
    P_55 = (1/8)*((-1)*a1)**(-7/2)*ConstantExp1*((-8)*a1**3*d0+4*a1**2*( \
        b1*d1+d2)+b1**3*d3+(-2)*a1*b1*(b1*d2+3*d3))*(2*pi)**(-1/2)*(1+ \
        sigmaMuV**2*sigmaV**(-2))**(-1/2)*sigmaV**(-1)
    
    #################
    d0 = p21c0
    d1 = p21c1
    d2 = p21c2
    d3 = p21c3
    d4 = p21c4
    P_56 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p22c0
    d1 = p22c1
    d2 = p22c2
    d3 = p22c3
    d4 = p22c4
    P_57 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p23c0
    d1 = p23c1
    d2 = p23c2
    d3 = p23c3
    d4 = p23c4
    P_58 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p24c0
    d1 = p24c1
    d2 = p24c2
    d3 = p24c3
    d4 = p24c4
    P_59 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p25c0
    d1 = p25c1
    d2 = p25c2
    d3 = p25c3
    d4 = p25c4
    P_60 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p26c0
    d1 = p26c1
    d2 = p26c2
    d3 = p26c3
    d4 = p26c4
    P_61 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p27c0
    d1 = p27c1
    d2 = p27c2
    d3 = p27c3
    d4 = p27c4
    P_62 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p28c0
    d1 = p28c1
    d2 = p28c2
    d3 = p28c3
    d4 = p28c4
    P_63 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p29c0
    d1 = p29c1
    d2 = p29c2
    d3 = p29c3
    d4 = p29c4
    P_64 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p30c0
    d1 = p30c1
    d2 = p30c2
    d3 = p30c3
    d4 = p30c4
    P_65 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p31c0
    d1 = p31c1
    d2 = p31c2
    d3 = p31c3
    d4 = p31c4
    P_66 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p32c0
    d1 = p32c1
    d2 = p32c2
    d3 = p32c3
    d4 = p32c4
    P_67 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p33c0
    d1 = p33c1
    d2 = p33c2
    d3 = p33c3
    d4 = p33c4
    P_68 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p34c0
    d1 = p34c1
    d2 = p34c2
    d3 = p34c3
    d4 = p34c4
    P_69 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    d0 = p35c0
    d1 = p35c1
    d2 = p35c2
    d3 = p35c3
    d4 = p35c4
    P_70 = (1/32)*(((-1)*a2)**(-9/2)*ConstantErf*((-16)*a2**4*d0+8*a2**3*( \
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
    P_output = zeros((n_sites, 70))
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
    P_output[:,35] = P_36
    P_output[:,36] = P_37
    P_output[:,37] = P_38
    P_output[:,38] = P_39
    P_output[:,39] = P_40
    P_output[:,40] = P_41
    P_output[:,41] = P_42
    P_output[:,42] = P_43
    P_output[:,43] = P_44
    P_output[:,44] = P_45
    P_output[:,45] = P_46
    P_output[:,46] = P_47
    P_output[:,47] = P_48
    P_output[:,48] = P_49
    P_output[:,49] = P_50
    P_output[:,50] = P_51
    P_output[:,51] = P_52
    P_output[:,52] = P_53
    P_output[:,53] = P_54
    P_output[:,54] = P_55
    P_output[:,55] = P_56
    P_output[:,56] = P_57
    P_output[:,57] = P_58
    P_output[:,58] = P_59
    P_output[:,59] = P_60
    P_output[:,60] = P_61
    P_output[:,61] = P_62
    P_output[:,62] = P_63
    P_output[:,63] = P_64
    P_output[:,64] = P_65
    P_output[:,65] = P_66
    P_output[:,66] = P_67
    P_output[:,67] = P_68
    P_output[:,68] = P_69
    P_output[:,69] = P_70
    
    # return
    return P_output