import numpy as np
from numba import njit, float64
from .pc_coeffs_second_int import pc_coeffs_second_int

@njit(
    float64[:,:](float64[:],float64[:],float64[:],
                 float64[:],float64[:],float64[:],float64[:],
                 float64[:],float64[:],float64[:],float64[:],
                 float64[:],float64[:],float64[:],float64[:]),
    fastmath=True,
    cache=True
)
def pc_coeffs_third_int(muY, sigmaMuY, sigmaY,  \
        amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT, amuV, bmuV,  \
        sigmaMuV, sigmaV): #
    
    
    # dimensions
    n_sites = len(muY)
    
    ## PC Coefficients from First Integration!!;
    P_output_2  =  pc_coeffs_second_int(muY,  \
        sigmaMuY, sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT,  \
        sigmaT)
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
    n_sites  =  len(muY)
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
