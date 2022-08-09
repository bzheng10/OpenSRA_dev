import numpy as np
from .pc_util import erf
from numba import njit, float64
from .pc_coeffs_third_int import pc_coeffs_third_int

@njit(
    # float64[:,:,:](float64[:],float64[:],float64[:],float64[:],
    #                float64[:],float64[:],float64[:],float64[:],
    #                float64[:],float64[:],float64[:],float64[:],
    #                float64[:],float64[:],float64[:],float64[:]),
    # fastmath=True,
    # cache=True
)
def pc_coeffs_cdf_triple_int(v, muY, sigmaMuY, sigmaY, amuZ, bmuZ,  \
        sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT, amuV, bmuV, sigmaMuV,  \
        sigmaV): #
    
    # dimensions
    n_sites = len(muY)
    n_pts_v = len(v)
    
    ## PC Coefficients from Third Integration!!;
    P_output_3 = pc_coeffs_third_int(muY, sigmaMuY, \
        sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT, \
        amuV, bmuV, sigmaMuV, sigmaV)
    
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
    p12c0 = np.zeros((1,n_sites))
    p12c1 = np.zeros((1,n_sites))
    p12c2 = np.zeros((1,n_sites))
    p13c0 = np.zeros((1,n_sites))
    p13c1 = np.zeros((1,n_sites))
    p13c2 = np.zeros((1,n_sites))
    p14c0 = np.zeros((1,n_sites))
    p14c1 = np.zeros((1,n_sites))
    p14c2 = np.zeros((1,n_sites))
    p15c0 = np.zeros((1,n_sites))
    p15c1 = np.zeros((1,n_sites))
    p15c2 = np.zeros((1,n_sites))
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
    p22c0 = np.zeros((1,n_sites))
    p22c1 = np.zeros((1,n_sites))
    p22c2 = np.zeros((1,n_sites))
    p22c3 = np.zeros((1,n_sites))
    p23c0 = np.zeros((1,n_sites))
    p23c1 = np.zeros((1,n_sites))
    p23c2 = np.zeros((1,n_sites))
    p23c3 = np.zeros((1,n_sites))
    p24c0 = np.zeros((1,n_sites))
    p24c1 = np.zeros((1,n_sites))
    p24c2 = np.zeros((1,n_sites))
    p24c3 = np.zeros((1,n_sites))
    p25c0 = np.zeros((1,n_sites))
    p25c1 = np.zeros((1,n_sites))
    p25c2 = np.zeros((1,n_sites))
    p25c3 = np.zeros((1,n_sites))
    p26c0 = np.zeros((1,n_sites))
    p26c1 = np.zeros((1,n_sites))
    p26c2 = np.zeros((1,n_sites))
    p26c3 = np.zeros((1,n_sites))
    p27c0 = np.zeros((1,n_sites))
    p27c1 = np.zeros((1,n_sites))
    p27c2 = np.zeros((1,n_sites))
    p27c3 = np.zeros((1,n_sites))
    p28c0 = np.zeros((1,n_sites))
    p28c1 = np.zeros((1,n_sites))
    p28c2 = np.zeros((1,n_sites))
    p28c3 = np.zeros((1,n_sites))
    p29c0 = np.zeros((1,n_sites))
    p29c1 = np.zeros((1,n_sites))
    p29c2 = np.zeros((1,n_sites))
    p29c3 = np.zeros((1,n_sites))
    p30c0 = np.zeros((1,n_sites))
    p30c1 = np.zeros((1,n_sites))
    p30c2 = np.zeros((1,n_sites))
    p30c3 = np.zeros((1,n_sites))
    p31c0 = np.zeros((1,n_sites))
    p31c1 = np.zeros((1,n_sites))
    p31c2 = np.zeros((1,n_sites))
    p31c3 = np.zeros((1,n_sites))
    p32c0 = np.zeros((1,n_sites))
    p32c1 = np.zeros((1,n_sites))
    p32c2 = np.zeros((1,n_sites))
    p32c3 = np.zeros((1,n_sites))
    p33c0 = np.zeros((1,n_sites))
    p33c1 = np.zeros((1,n_sites))
    p33c2 = np.zeros((1,n_sites))
    p33c3 = np.zeros((1,n_sites))
    p34c0 = np.zeros((1,n_sites))
    p34c1 = np.zeros((1,n_sites))
    p34c2 = np.zeros((1,n_sites))
    p34c3 = np.zeros((1,n_sites))
    p35c0 = np.zeros((1,n_sites))
    p35c1 = np.zeros((1,n_sites))
    p35c2 = np.zeros((1,n_sites))
    p35c3 = np.zeros((1,n_sites))
    p36c0 = np.zeros((1,n_sites))
    p36c1 = np.zeros((1,n_sites))
    p36c2 = np.zeros((1,n_sites))
    p36c3 = np.zeros((1,n_sites))
    p36c4 = np.zeros((1,n_sites))
    p37c0 = np.zeros((1,n_sites))
    p37c1 = np.zeros((1,n_sites))
    p37c2 = np.zeros((1,n_sites))
    p37c3 = np.zeros((1,n_sites))
    p37c4 = np.zeros((1,n_sites))
    p38c0 = np.zeros((1,n_sites))
    p38c1 = np.zeros((1,n_sites))
    p38c2 = np.zeros((1,n_sites))
    p38c3 = np.zeros((1,n_sites))
    p38c4 = np.zeros((1,n_sites))
    p39c0 = np.zeros((1,n_sites))
    p39c1 = np.zeros((1,n_sites))
    p39c2 = np.zeros((1,n_sites))
    p39c3 = np.zeros((1,n_sites))
    p39c4 = np.zeros((1,n_sites))
    p40c0 = np.zeros((1,n_sites))
    p40c1 = np.zeros((1,n_sites))
    p40c2 = np.zeros((1,n_sites))
    p40c3 = np.zeros((1,n_sites))
    p40c4 = np.zeros((1,n_sites))
    p41c0 = np.zeros((1,n_sites))
    p41c1 = np.zeros((1,n_sites))
    p41c2 = np.zeros((1,n_sites))
    p41c3 = np.zeros((1,n_sites))
    p41c4 = np.zeros((1,n_sites))
    p42c0 = np.zeros((1,n_sites))
    p42c1 = np.zeros((1,n_sites))
    p42c2 = np.zeros((1,n_sites))
    p42c3 = np.zeros((1,n_sites))
    p42c4 = np.zeros((1,n_sites))
    p43c0 = np.zeros((1,n_sites))
    p43c1 = np.zeros((1,n_sites))
    p43c2 = np.zeros((1,n_sites))
    p43c3 = np.zeros((1,n_sites))
    p43c4 = np.zeros((1,n_sites))
    p44c0 = np.zeros((1,n_sites))
    p44c1 = np.zeros((1,n_sites))
    p44c2 = np.zeros((1,n_sites))
    p44c3 = np.zeros((1,n_sites))
    p44c4 = np.zeros((1,n_sites))
    p45c0 = np.zeros((1,n_sites))
    p45c1 = np.zeros((1,n_sites))
    p45c2 = np.zeros((1,n_sites))
    p45c3 = np.zeros((1,n_sites))
    p45c4 = np.zeros((1,n_sites))
    p46c0 = np.zeros((1,n_sites))
    p46c1 = np.zeros((1,n_sites))
    p46c2 = np.zeros((1,n_sites))
    p46c3 = np.zeros((1,n_sites))
    p46c4 = np.zeros((1,n_sites))
    p47c0 = np.zeros((1,n_sites))
    p47c1 = np.zeros((1,n_sites))
    p47c2 = np.zeros((1,n_sites))
    p47c3 = np.zeros((1,n_sites))
    p47c4 = np.zeros((1,n_sites))
    p48c0 = np.zeros((1,n_sites))
    p48c1 = np.zeros((1,n_sites))
    p48c2 = np.zeros((1,n_sites))
    p48c3 = np.zeros((1,n_sites))
    p48c4 = np.zeros((1,n_sites))
    p49c0 = np.zeros((1,n_sites))
    p49c1 = np.zeros((1,n_sites))
    p49c2 = np.zeros((1,n_sites))
    p49c3 = np.zeros((1,n_sites))
    p49c4 = np.zeros((1,n_sites))
    p50c0 = np.zeros((1,n_sites))
    p50c1 = np.zeros((1,n_sites))
    p50c2 = np.zeros((1,n_sites))
    p50c3 = np.zeros((1,n_sites))
    p50c4 = np.zeros((1,n_sites))
    p51c0 = np.zeros((1,n_sites))
    p51c1 = np.zeros((1,n_sites))
    p51c2 = np.zeros((1,n_sites))
    p51c3 = np.zeros((1,n_sites))
    p51c4 = np.zeros((1,n_sites))
    p52c0 = np.zeros((1,n_sites))
    p52c1 = np.zeros((1,n_sites))
    p52c2 = np.zeros((1,n_sites))
    p52c3 = np.zeros((1,n_sites))
    p52c4 = np.zeros((1,n_sites))
    p53c0 = np.zeros((1,n_sites))
    p53c1 = np.zeros((1,n_sites))
    p53c2 = np.zeros((1,n_sites))
    p53c3 = np.zeros((1,n_sites))
    p53c4 = np.zeros((1,n_sites))
    p54c0 = np.zeros((1,n_sites))
    p54c1 = np.zeros((1,n_sites))
    p54c2 = np.zeros((1,n_sites))
    p54c3 = np.zeros((1,n_sites))
    p54c4 = np.zeros((1,n_sites))
    p55c0 = np.zeros((1,n_sites))
    p55c1 = np.zeros((1,n_sites))
    p55c2 = np.zeros((1,n_sites))
    p55c3 = np.zeros((1,n_sites))
    p55c4 = np.zeros((1,n_sites))
    p56c0 = np.zeros((1,n_sites))
    p56c1 = np.zeros((1,n_sites))
    p56c2 = np.zeros((1,n_sites))
    p56c3 = np.zeros((1,n_sites))
    p56c4 = np.zeros((1,n_sites))
    p57c0 = np.zeros((1,n_sites))
    p57c1 = np.zeros((1,n_sites))
    p57c2 = np.zeros((1,n_sites))
    p57c3 = np.zeros((1,n_sites))
    p57c4 = np.zeros((1,n_sites))
    p58c0 = np.zeros((1,n_sites))
    p58c1 = np.zeros((1,n_sites))
    p58c2 = np.zeros((1,n_sites))
    p58c3 = np.zeros((1,n_sites))
    p58c4 = np.zeros((1,n_sites))
    p59c0 = np.zeros((1,n_sites))
    p59c1 = np.zeros((1,n_sites))
    p59c2 = np.zeros((1,n_sites))
    p59c3 = np.zeros((1,n_sites))
    p59c4 = np.zeros((1,n_sites))
    p60c0 = np.zeros((1,n_sites))
    p60c1 = np.zeros((1,n_sites))
    p60c2 = np.zeros((1,n_sites))
    p60c3 = np.zeros((1,n_sites))
    p60c4 = np.zeros((1,n_sites))
    p61c0 = np.zeros((1,n_sites))
    p61c1 = np.zeros((1,n_sites))
    p61c2 = np.zeros((1,n_sites))
    p61c3 = np.zeros((1,n_sites))
    p61c4 = np.zeros((1,n_sites))
    p62c0 = np.zeros((1,n_sites))
    p62c1 = np.zeros((1,n_sites))
    p62c2 = np.zeros((1,n_sites))
    p62c3 = np.zeros((1,n_sites))
    p62c4 = np.zeros((1,n_sites))
    p63c0 = np.zeros((1,n_sites))
    p63c1 = np.zeros((1,n_sites))
    p63c2 = np.zeros((1,n_sites))
    p63c3 = np.zeros((1,n_sites))
    p63c4 = np.zeros((1,n_sites))
    p64c0 = np.zeros((1,n_sites))
    p64c1 = np.zeros((1,n_sites))
    p64c2 = np.zeros((1,n_sites))
    p64c3 = np.zeros((1,n_sites))
    p64c4 = np.zeros((1,n_sites))
    p65c0 = np.zeros((1,n_sites))
    p65c1 = np.zeros((1,n_sites))
    p65c2 = np.zeros((1,n_sites))
    p65c3 = np.zeros((1,n_sites))
    p65c4 = np.zeros((1,n_sites))
    p66c0 = np.zeros((1,n_sites))
    p66c1 = np.zeros((1,n_sites))
    p66c2 = np.zeros((1,n_sites))
    p66c3 = np.zeros((1,n_sites))
    p66c4 = np.zeros((1,n_sites))
    p67c0 = np.zeros((1,n_sites))
    p67c1 = np.zeros((1,n_sites))
    p67c2 = np.zeros((1,n_sites))
    p67c3 = np.zeros((1,n_sites))
    p67c4 = np.zeros((1,n_sites))
    p68c0 = np.zeros((1,n_sites))
    p68c1 = np.zeros((1,n_sites))
    p68c2 = np.zeros((1,n_sites))
    p68c3 = np.zeros((1,n_sites))
    p68c4 = np.zeros((1,n_sites))
    p69c0 = np.zeros((1,n_sites))
    p69c1 = np.zeros((1,n_sites))
    p69c2 = np.zeros((1,n_sites))
    p69c3 = np.zeros((1,n_sites))
    p69c4 = np.zeros((1,n_sites))
    p70c0 = np.zeros((1,n_sites))
    p70c1 = np.zeros((1,n_sites))
    p70c2 = np.zeros((1,n_sites))
    p70c3 = np.zeros((1,n_sites))
    p70c4 = np.zeros((1,n_sites))
    
    p1c0[0,:] = P_output_3[0, :]
    p2c0[0,:] = P_output_3[1, :]
    p2c1[0,:] = P_output_3[2, :]
    p3c0[0,:] = P_output_3[3, :]
    p3c1[0,:] = P_output_3[4, :]
    p4c0[0,:] = P_output_3[5, :]
    p4c1[0,:] = P_output_3[6, :]
    p5c0[0,:] = P_output_3[7, :]
    p5c1[0,:] = P_output_3[8, :]
    p6c0[0,:] = P_output_3[9, :]
    p6c1[0,:] = P_output_3[10, :]
    p6c2[0,:] = P_output_3[11, :]
    p7c0[0,:] = P_output_3[12, :]
    p7c1[0,:] = P_output_3[13, :]
    p7c2[0,:] = P_output_3[14, :]
    p8c0[0,:] = P_output_3[15, :]
    p8c1[0,:] = P_output_3[16, :]
    p8c2[0,:] = P_output_3[17, :]
    p9c0[0,:] = P_output_3[18, :]
    p9c1[0,:] = P_output_3[19, :]
    p9c2[0,:] = P_output_3[20, :]
    p10c0[0,:] = P_output_3[21, :]
    p10c1[0,:] = P_output_3[22, :]
    p10c2[0,:] = P_output_3[23, :]
    p11c0[0,:] = P_output_3[24, :]
    p11c1[0,:] = P_output_3[25, :]
    p11c2[0,:] = P_output_3[26, :]
    p12c0[0,:] = P_output_3[27, :]
    p12c1[0,:] = P_output_3[28, :]
    p12c2[0,:] = P_output_3[29, :]
    p13c0[0,:] = P_output_3[30, :]
    p13c1[0,:] = P_output_3[31, :]
    p13c2[0,:] = P_output_3[32, :]
    p14c0[0,:] = P_output_3[33, :]
    p14c1[0,:] = P_output_3[34, :]
    p14c2[0,:] = P_output_3[35, :]
    p15c0[0,:] = P_output_3[36, :]
    p15c1[0,:] = P_output_3[37, :]
    p15c2[0,:] = P_output_3[38, :]
    p16c0[0,:] = P_output_3[39, :]
    p16c1[0,:] = P_output_3[40, :]
    p16c2[0,:] = P_output_3[41, :]
    p16c3[0,:] = P_output_3[42, :]
    p17c0[0,:] = P_output_3[43, :]
    p17c1[0,:] = P_output_3[44, :]
    p17c2[0,:] = P_output_3[45, :]
    p17c3[0,:] = P_output_3[46, :]
    p18c0[0,:] = P_output_3[47, :]
    p18c1[0,:] = P_output_3[48, :]
    p18c2[0,:] = P_output_3[49, :]
    p18c3[0,:] = P_output_3[50, :]
    p19c0[0,:] = P_output_3[51, :]
    p19c1[0,:] = P_output_3[52, :]
    p19c2[0,:] = P_output_3[53, :]
    p19c3[0,:] = P_output_3[54, :]
    p20c0[0,:] = P_output_3[55, :]
    p20c1[0,:] = P_output_3[56, :]
    p20c2[0,:] = P_output_3[57, :]
    p20c3[0,:] = P_output_3[58, :]
    p21c0[0,:] = P_output_3[59, :]
    p21c1[0,:] = P_output_3[60, :]
    p21c2[0,:] = P_output_3[61, :]
    p21c3[0,:] = P_output_3[62, :]
    p22c0[0,:] = P_output_3[63, :]
    p22c1[0,:] = P_output_3[64, :]
    p22c2[0,:] = P_output_3[65, :]
    p22c3[0,:] = P_output_3[66, :]
    p23c0[0,:] = P_output_3[67, :]
    p23c1[0,:] = P_output_3[68, :]
    p23c2[0,:] = P_output_3[69, :]
    p23c3[0,:] = P_output_3[70, :]
    p24c0[0,:] = P_output_3[71, :]
    p24c1[0,:] = P_output_3[72, :]
    p24c2[0,:] = P_output_3[73, :]
    p24c3[0,:] = P_output_3[74, :]
    p25c0[0,:] = P_output_3[75, :]
    p25c1[0,:] = P_output_3[76, :]
    p25c2[0,:] = P_output_3[77, :]
    p25c3[0,:] = P_output_3[78, :]
    p26c0[0,:] = P_output_3[79, :]
    p26c1[0,:] = P_output_3[80, :]
    p26c2[0,:] = P_output_3[81, :]
    p26c3[0,:] = P_output_3[82, :]
    p27c0[0,:] = P_output_3[83, :]
    p27c1[0,:] = P_output_3[84, :]
    p27c2[0,:] = P_output_3[85, :]
    p27c3[0,:] = P_output_3[86, :]
    p28c0[0,:] = P_output_3[87, :]
    p28c1[0,:] = P_output_3[88, :]
    p28c2[0,:] = P_output_3[89, :]
    p28c3[0,:] = P_output_3[90, :]
    p29c0[0,:] = P_output_3[91, :]
    p29c1[0,:] = P_output_3[92, :]
    p29c2[0,:] = P_output_3[93, :]
    p29c3[0,:] = P_output_3[94, :]
    p30c0[0,:] = P_output_3[95, :]
    p30c1[0,:] = P_output_3[96, :]
    p30c2[0,:] = P_output_3[97, :]
    p30c3[0,:] = P_output_3[98, :]
    p31c0[0,:] = P_output_3[99, :]
    p31c1[0,:] = P_output_3[100, :]
    p31c2[0,:] = P_output_3[101, :]
    p31c3[0,:] = P_output_3[102, :]
    p32c0[0,:] = P_output_3[103, :]
    p32c1[0,:] = P_output_3[104, :]
    p32c2[0,:] = P_output_3[105, :]
    p32c3[0,:] = P_output_3[106, :]
    p33c0[0,:] = P_output_3[107, :]
    p33c1[0,:] = P_output_3[108, :]
    p33c2[0,:] = P_output_3[109, :]
    p33c3[0,:] = P_output_3[110, :]
    p34c0[0,:] = P_output_3[111, :]
    p34c1[0,:] = P_output_3[112, :]
    p34c2[0,:] = P_output_3[113, :]
    p34c3[0,:] = P_output_3[114, :]
    p35c0[0,:] = P_output_3[115, :]
    p35c1[0,:] = P_output_3[116, :]
    p35c2[0,:] = P_output_3[117, :]
    p35c3[0,:] = P_output_3[118, :]
    p36c0[0,:] = P_output_3[119, :]
    p36c1[0,:] = P_output_3[120, :]
    p36c2[0,:] = P_output_3[121, :]
    p36c3[0,:] = P_output_3[122, :]
    p36c4[0,:] = P_output_3[123, :]
    p37c0[0,:] = P_output_3[124, :]
    p37c1[0,:] = P_output_3[125, :]
    p37c2[0,:] = P_output_3[126, :]
    p37c3[0,:] = P_output_3[127, :]
    p37c4[0,:] = P_output_3[128, :]
    p38c0[0,:] = P_output_3[129, :]
    p38c1[0,:] = P_output_3[130, :]
    p38c2[0,:] = P_output_3[131, :]
    p38c3[0,:] = P_output_3[132, :]
    p38c4[0,:] = P_output_3[133, :]
    p39c0[0,:] = P_output_3[134, :]
    p39c1[0,:] = P_output_3[135, :]
    p39c2[0,:] = P_output_3[136, :]
    p39c3[0,:] = P_output_3[137, :]
    p39c4[0,:] = P_output_3[138, :]
    p40c0[0,:] = P_output_3[139, :]
    p40c1[0,:] = P_output_3[140, :]
    p40c2[0,:] = P_output_3[141, :]
    p40c3[0,:] = P_output_3[142, :]
    p40c4[0,:] = P_output_3[143, :]
    p41c0[0,:] = P_output_3[144, :]
    p41c1[0,:] = P_output_3[145, :]
    p41c2[0,:] = P_output_3[146, :]
    p41c3[0,:] = P_output_3[147, :]
    p41c4[0,:] = P_output_3[148, :]
    p42c0[0,:] = P_output_3[149, :]
    p42c1[0,:] = P_output_3[150, :]
    p42c2[0,:] = P_output_3[151, :]
    p42c3[0,:] = P_output_3[152, :]
    p42c4[0,:] = P_output_3[153, :]
    p43c0[0,:] = P_output_3[154, :]
    p43c1[0,:] = P_output_3[155, :]
    p43c2[0,:] = P_output_3[156, :]
    p43c3[0,:] = P_output_3[157, :]
    p43c4[0,:] = P_output_3[158, :]
    p44c0[0,:] = P_output_3[159, :]
    p44c1[0,:] = P_output_3[160, :]
    p44c2[0,:] = P_output_3[161, :]
    p44c3[0,:] = P_output_3[162, :]
    p44c4[0,:] = P_output_3[163, :]
    p45c0[0,:] = P_output_3[164, :]
    p45c1[0,:] = P_output_3[165, :]
    p45c2[0,:] = P_output_3[166, :]
    p45c3[0,:] = P_output_3[167, :]
    p45c4[0,:] = P_output_3[168, :]
    p46c0[0,:] = P_output_3[169, :]
    p46c1[0,:] = P_output_3[170, :]
    p46c2[0,:] = P_output_3[171, :]
    p46c3[0,:] = P_output_3[172, :]
    p46c4[0,:] = P_output_3[173, :]
    p47c0[0,:] = P_output_3[174, :]
    p47c1[0,:] = P_output_3[175, :]
    p47c2[0,:] = P_output_3[176, :]
    p47c3[0,:] = P_output_3[177, :]
    p47c4[0,:] = P_output_3[178, :]
    p48c0[0,:] = P_output_3[179, :]
    p48c1[0,:] = P_output_3[180, :]
    p48c2[0,:] = P_output_3[181, :]
    p48c3[0,:] = P_output_3[182, :]
    p48c4[0,:] = P_output_3[183, :]
    p49c0[0,:] = P_output_3[184, :]
    p49c1[0,:] = P_output_3[185, :]
    p49c2[0,:] = P_output_3[186, :]
    p49c3[0,:] = P_output_3[187, :]
    p49c4[0,:] = P_output_3[188, :]
    p50c0[0,:] = P_output_3[189, :]
    p50c1[0,:] = P_output_3[190, :]
    p50c2[0,:] = P_output_3[191, :]
    p50c3[0,:] = P_output_3[192, :]
    p50c4[0,:] = P_output_3[193, :]
    p51c0[0,:] = P_output_3[194, :]
    p51c1[0,:] = P_output_3[195, :]
    p51c2[0,:] = P_output_3[196, :]
    p51c3[0,:] = P_output_3[197, :]
    p51c4[0,:] = P_output_3[198, :]
    p52c0[0,:] = P_output_3[199, :]
    p52c1[0,:] = P_output_3[200, :]
    p52c2[0,:] = P_output_3[201, :]
    p52c3[0,:] = P_output_3[202, :]
    p52c4[0,:] = P_output_3[203, :]
    p53c0[0,:] = P_output_3[204, :]
    p53c1[0,:] = P_output_3[205, :]
    p53c2[0,:] = P_output_3[206, :]
    p53c3[0,:] = P_output_3[207, :]
    p53c4[0,:] = P_output_3[208, :]
    p54c0[0,:] = P_output_3[209, :]
    p54c1[0,:] = P_output_3[210, :]
    p54c2[0,:] = P_output_3[211, :]
    p54c3[0,:] = P_output_3[212, :]
    p54c4[0,:] = P_output_3[213, :]
    p55c0[0,:] = P_output_3[214, :]
    p55c1[0,:] = P_output_3[215, :]
    p55c2[0,:] = P_output_3[216, :]
    p55c3[0,:] = P_output_3[217, :]
    p55c4[0,:] = P_output_3[218, :]
    p56c0[0,:] = P_output_3[219, :]
    p56c1[0,:] = P_output_3[220, :]
    p56c2[0,:] = P_output_3[221, :]
    p56c3[0,:] = P_output_3[222, :]
    p56c4[0,:] = P_output_3[223, :]
    p57c0[0,:] = P_output_3[224, :]
    p57c1[0,:] = P_output_3[225, :]
    p57c2[0,:] = P_output_3[226, :]
    p57c3[0,:] = P_output_3[227, :]
    p57c4[0,:] = P_output_3[228, :]
    p58c0[0,:] = P_output_3[229, :]
    p58c1[0,:] = P_output_3[230, :]
    p58c2[0,:] = P_output_3[231, :]
    p58c3[0,:] = P_output_3[232, :]
    p58c4[0,:] = P_output_3[233, :]
    p59c0[0,:] = P_output_3[234, :]
    p59c1[0,:] = P_output_3[235, :]
    p59c2[0,:] = P_output_3[236, :]
    p59c3[0,:] = P_output_3[237, :]
    p59c4[0,:] = P_output_3[238, :]
    p60c0[0,:] = P_output_3[239, :]
    p60c1[0,:] = P_output_3[240, :]
    p60c2[0,:] = P_output_3[241, :]
    p60c3[0,:] = P_output_3[242, :]
    p60c4[0,:] = P_output_3[243, :]
    p61c0[0,:] = P_output_3[244, :]
    p61c1[0,:] = P_output_3[245, :]
    p61c2[0,:] = P_output_3[246, :]
    p61c3[0,:] = P_output_3[247, :]
    p61c4[0,:] = P_output_3[248, :]
    p62c0[0,:] = P_output_3[249, :]
    p62c1[0,:] = P_output_3[250, :]
    p62c2[0,:] = P_output_3[251, :]
    p62c3[0,:] = P_output_3[252, :]
    p62c4[0,:] = P_output_3[253, :]
    p63c0[0,:] = P_output_3[254, :]
    p63c1[0,:] = P_output_3[255, :]
    p63c2[0,:] = P_output_3[256, :]
    p63c3[0,:] = P_output_3[257, :]
    p63c4[0,:] = P_output_3[258, :]
    p64c0[0,:] = P_output_3[259, :]
    p64c1[0,:] = P_output_3[260, :]
    p64c2[0,:] = P_output_3[261, :]
    p64c3[0,:] = P_output_3[262, :]
    p64c4[0,:] = P_output_3[263, :]
    p65c0[0,:] = P_output_3[264, :]
    p65c1[0,:] = P_output_3[265, :]
    p65c2[0,:] = P_output_3[266, :]
    p65c3[0,:] = P_output_3[267, :]
    p65c4[0,:] = P_output_3[268, :]
    p66c0[0,:] = P_output_3[269, :]
    p66c1[0,:] = P_output_3[270, :]
    p66c2[0,:] = P_output_3[271, :]
    p66c3[0,:] = P_output_3[272, :]
    p66c4[0,:] = P_output_3[273, :]
    p67c0[0,:] = P_output_3[274, :]
    p67c1[0,:] = P_output_3[275, :]
    p67c2[0,:] = P_output_3[276, :]
    p67c3[0,:] = P_output_3[277, :]
    p67c4[0,:] = P_output_3[278, :]
    p68c0[0,:] = P_output_3[279, :]
    p68c1[0,:] = P_output_3[280, :]
    p68c2[0,:] = P_output_3[281, :]
    p68c3[0,:] = P_output_3[282, :]
    p68c4[0,:] = P_output_3[283, :]
    p69c0[0,:] = P_output_3[284, :]
    p69c1[0,:] = P_output_3[285, :]
    p69c2[0,:] = P_output_3[286, :]
    p69c3[0,:] = P_output_3[287, :]
    p69c4[0,:] = P_output_3[288, :]
    p70c0[0,:] = P_output_3[289, :]
    p70c1[0,:] = P_output_3[290, :]
    p70c2[0,:] = P_output_3[291, :]
    p70c3[0,:] = P_output_3[292, :]
    p70c4[0,:] = P_output_3[293, :]
    
    # precalculate
    a = (-1/2)*(sigmaMuV**2+sigmaV**2+amuV**2*(sigmaMuT**2+sigmaT**2+amuT**2* \
        (sigmaMuZ**2+amuZ**2*(sigmaMuY**2+sigmaY**2)+sigmaZ**2)))**(-1)
    b = (bmuV+amuV*(bmuT+amuT*(bmuZ+amuZ*muY)))*(sigmaMuV**2+sigmaV**2+ \
        amuV**2*(sigmaMuT**2+sigmaT**2+amuT**2*(sigmaMuZ**2+amuZ**2*( \
        sigmaMuY**2+sigmaY**2)+sigmaZ**2)))**(-1)
    
    # resize
    a = np.expand_dims(a,axis=0)
    b = np.expand_dims(b,axis=0)
    v = np.expand_dims(v,axis=1)
    
    
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
    P_5 = (1/4)*a**(-1)*(2*expCst*p5c1+((-1)*a)**(-3/2)*a*((-1)+ErfCst)*( \
        2*a*p5c0+(-1)*b*p5c1)*np.pi**(1/2))
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
    P_11 = (1/8)*((-1)*a)**(-5/2)*(b*p11c2*((-2)*((-1)*a)**(1/2)*expCst+( \
        -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p11c0* \
        np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p11c1+p11c2)*np.pi**(1/2)+2*((-1)*a) \
        **(1/2)*expCst*(p11c1+p11c2*v)))
    P_12 = (1/8)*((-1)*a)**(-5/2)*(b*p12c2*((-2)*((-1)*a)**(1/2)*expCst+( \
        -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p12c0* \
        np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p12c1+p12c2)*np.pi**(1/2)+2*((-1)*a) \
        **(1/2)*expCst*(p12c1+p12c2*v)))
    P_13 = (1/8)*((-1)*a)**(-5/2)*(b*p13c2*((-2)*((-1)*a)**(1/2)*expCst+( \
        -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p13c0* \
        np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p13c1+p13c2)*np.pi**(1/2)+2*((-1)*a) \
        **(1/2)*expCst*(p13c1+p13c2*v)))
    P_14 = (1/8)*((-1)*a)**(-5/2)*(b*p14c2*((-2)*((-1)*a)**(1/2)*expCst+( \
        -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p14c0* \
        np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p14c1+p14c2)*np.pi**(1/2)+2*((-1)*a) \
        **(1/2)*expCst*(p14c1+p14c2*v)))
    P_15 = (1/8)*((-1)*a)**(-5/2)*(b*p15c2*((-2)*((-1)*a)**(1/2)*expCst+( \
        -1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-4)*a**2*((-1)+ErfCst)*p15c0* \
        np.pi**(1/2)+2*a*(((-1)+ErfCst)*(b*p15c1+p15c2)*np.pi**(1/2)+2*((-1)*a) \
        **(1/2)*expCst*(p15c1+p15c2*v)))
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
    P_21 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p21c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p21c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p21c3+b**2*((-1)+ErfCst)* \
        p21c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p21c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p21c2+p21c3*v))+4*a**2*(((-1)+ErfCst)*(b*p21c1+ \
        p21c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p21c1+v*(p21c2+p21c3* \
        v))))
    P_22 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p22c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p22c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p22c3+b**2*((-1)+ErfCst)* \
        p22c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p22c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p22c2+p22c3*v))+4*a**2*(((-1)+ErfCst)*(b*p22c1+ \
        p22c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p22c1+v*(p22c2+p22c3* \
        v))))
    P_23 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p23c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p23c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p23c3+b**2*((-1)+ErfCst)* \
        p23c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p23c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p23c2+p23c3*v))+4*a**2*(((-1)+ErfCst)*(b*p23c1+ \
        p23c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p23c1+v*(p23c2+p23c3* \
        v))))
    P_24 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p24c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p24c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p24c3+b**2*((-1)+ErfCst)* \
        p24c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p24c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p24c2+p24c3*v))+4*a**2*(((-1)+ErfCst)*(b*p24c1+ \
        p24c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p24c1+v*(p24c2+p24c3* \
        v))))
    P_25 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p25c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p25c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p25c3+b**2*((-1)+ErfCst)* \
        p25c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p25c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p25c2+p25c3*v))+4*a**2*(((-1)+ErfCst)*(b*p25c1+ \
        p25c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p25c1+v*(p25c2+p25c3* \
        v))))
    P_26 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p26c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p26c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p26c3+b**2*((-1)+ErfCst)* \
        p26c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p26c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p26c2+p26c3*v))+4*a**2*(((-1)+ErfCst)*(b*p26c1+ \
        p26c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p26c1+v*(p26c2+p26c3* \
        v))))
    P_27 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p27c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p27c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p27c3+b**2*((-1)+ErfCst)* \
        p27c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p27c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p27c2+p27c3*v))+4*a**2*(((-1)+ErfCst)*(b*p27c1+ \
        p27c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p27c1+v*(p27c2+p27c3* \
        v))))
    P_28 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p28c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p28c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p28c3+b**2*((-1)+ErfCst)* \
        p28c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p28c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p28c2+p28c3*v))+4*a**2*(((-1)+ErfCst)*(b*p28c1+ \
        p28c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p28c1+v*(p28c2+p28c3* \
        v))))
    P_29 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p29c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p29c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p29c3+b**2*((-1)+ErfCst)* \
        p29c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p29c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p29c2+p29c3*v))+4*a**2*(((-1)+ErfCst)*(b*p29c1+ \
        p29c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p29c1+v*(p29c2+p29c3* \
        v))))
    P_30 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p30c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p30c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p30c3+b**2*((-1)+ErfCst)* \
        p30c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p30c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p30c2+p30c3*v))+4*a**2*(((-1)+ErfCst)*(b*p30c1+ \
        p30c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p30c1+v*(p30c2+p30c3* \
        v))))
    P_31 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p31c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p31c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p31c3+b**2*((-1)+ErfCst)* \
        p31c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p31c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p31c2+p31c3*v))+4*a**2*(((-1)+ErfCst)*(b*p31c1+ \
        p31c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p31c1+v*(p31c2+p31c3* \
        v))))
    P_32 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p32c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p32c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p32c3+b**2*((-1)+ErfCst)* \
        p32c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p32c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p32c2+p32c3*v))+4*a**2*(((-1)+ErfCst)*(b*p32c1+ \
        p32c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p32c1+v*(p32c2+p32c3* \
        v))))
    P_33 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p33c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p33c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p33c3+b**2*((-1)+ErfCst)* \
        p33c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p33c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p33c2+p33c3*v))+4*a**2*(((-1)+ErfCst)*(b*p33c1+ \
        p33c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p33c1+v*(p33c2+p33c3* \
        v))))
    P_34 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p34c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p34c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p34c3+b**2*((-1)+ErfCst)* \
        p34c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p34c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p34c2+p34c3*v))+4*a**2*(((-1)+ErfCst)*(b*p34c1+ \
        p34c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p34c1+v*(p34c2+p34c3* \
        v))))
    P_35 = (-1/16)*((-1)*a)**(-7/2)*(b**2*p35c3*(2*((-1)*a)**(1/2)*expCst+ \
        b*((-1)+ErfCst)*np.pi**(1/2))+(-8)*a**3*((-1)+ErfCst)*p35c0*np.pi**(1/2) \
        +(-2)*a*(4*((-1)*a)**(1/2)*expCst*p35c3+b**2*((-1)+ErfCst)* \
        p35c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p35c3*np.pi**(1/2)+2*((-1)*a)**( \
        1/2)*b*expCst*(p35c2+p35c3*v))+4*a**2*(((-1)+ErfCst)*(b*p35c1+ \
        p35c2)*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(p35c1+v*(p35c2+p35c3* \
        v))))
    P_36 = (1/32)*((-1)*a)**(-9/2)*(b**3*p36c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p36c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p36c4+b**2*(( \
        -1)+ErfCst)*p36c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p36c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p36c3+p36c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p36c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p36c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p36c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p36c3+3* \
        p36c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p36c2+v*(p36c3+p36c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p36c1+p36c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p36c1+v*(p36c2+v*(p36c3+p36c4*v)))))
    P_37 = (1/32)*((-1)*a)**(-9/2)*(b**3*p37c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p37c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p37c4+b**2*(( \
        -1)+ErfCst)*p37c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p37c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p37c3+p37c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p37c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p37c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p37c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p37c3+3* \
        p37c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p37c2+v*(p37c3+p37c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p37c1+p37c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p37c1+v*(p37c2+v*(p37c3+p37c4*v)))))
    P_38 = (1/32)*((-1)*a)**(-9/2)*(b**3*p38c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p38c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p38c4+b**2*(( \
        -1)+ErfCst)*p38c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p38c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p38c3+p38c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p38c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p38c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p38c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p38c3+3* \
        p38c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p38c2+v*(p38c3+p38c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p38c1+p38c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p38c1+v*(p38c2+v*(p38c3+p38c4*v)))))
    P_39 = (1/32)*((-1)*a)**(-9/2)*(b**3*p39c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p39c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p39c4+b**2*(( \
        -1)+ErfCst)*p39c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p39c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p39c3+p39c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p39c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p39c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p39c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p39c3+3* \
        p39c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p39c2+v*(p39c3+p39c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p39c1+p39c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p39c1+v*(p39c2+v*(p39c3+p39c4*v)))))
    P_40 = (1/32)*((-1)*a)**(-9/2)*(b**3*p40c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p40c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p40c4+b**2*(( \
        -1)+ErfCst)*p40c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p40c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p40c3+p40c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p40c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p40c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p40c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p40c3+3* \
        p40c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p40c2+v*(p40c3+p40c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p40c1+p40c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p40c1+v*(p40c2+v*(p40c3+p40c4*v)))))
    P_41 = (1/32)*((-1)*a)**(-9/2)*(b**3*p41c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p41c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p41c4+b**2*(( \
        -1)+ErfCst)*p41c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p41c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p41c3+p41c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p41c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p41c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p41c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p41c3+3* \
        p41c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p41c2+v*(p41c3+p41c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p41c1+p41c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p41c1+v*(p41c2+v*(p41c3+p41c4*v)))))
    P_42 = (1/32)*((-1)*a)**(-9/2)*(b**3*p42c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p42c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p42c4+b**2*(( \
        -1)+ErfCst)*p42c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p42c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p42c3+p42c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p42c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p42c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p42c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p42c3+3* \
        p42c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p42c2+v*(p42c3+p42c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p42c1+p42c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p42c1+v*(p42c2+v*(p42c3+p42c4*v)))))
    P_43 = (1/32)*((-1)*a)**(-9/2)*(b**3*p43c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p43c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p43c4+b**2*(( \
        -1)+ErfCst)*p43c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p43c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p43c3+p43c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p43c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p43c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p43c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p43c3+3* \
        p43c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p43c2+v*(p43c3+p43c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p43c1+p43c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p43c1+v*(p43c2+v*(p43c3+p43c4*v)))))
    P_44 = (1/32)*((-1)*a)**(-9/2)*(b**3*p44c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p44c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p44c4+b**2*(( \
        -1)+ErfCst)*p44c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p44c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p44c3+p44c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p44c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p44c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p44c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p44c3+3* \
        p44c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p44c2+v*(p44c3+p44c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p44c1+p44c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p44c1+v*(p44c2+v*(p44c3+p44c4*v)))))
    P_45 = (1/32)*((-1)*a)**(-9/2)*(b**3*p45c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p45c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p45c4+b**2*(( \
        -1)+ErfCst)*p45c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p45c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p45c3+p45c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p45c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p45c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p45c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p45c3+3* \
        p45c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p45c2+v*(p45c3+p45c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p45c1+p45c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p45c1+v*(p45c2+v*(p45c3+p45c4*v)))))
    P_46 = (1/32)*((-1)*a)**(-9/2)*(b**3*p46c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p46c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p46c4+b**2*(( \
        -1)+ErfCst)*p46c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p46c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p46c3+p46c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p46c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p46c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p46c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p46c3+3* \
        p46c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p46c2+v*(p46c3+p46c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p46c1+p46c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p46c1+v*(p46c2+v*(p46c3+p46c4*v)))))
    P_47 = (1/32)*((-1)*a)**(-9/2)*(b**3*p47c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p47c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p47c4+b**2*(( \
        -1)+ErfCst)*p47c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p47c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p47c3+p47c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p47c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p47c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p47c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p47c3+3* \
        p47c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p47c2+v*(p47c3+p47c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p47c1+p47c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p47c1+v*(p47c2+v*(p47c3+p47c4*v)))))
    P_48 = (1/32)*((-1)*a)**(-9/2)*(b**3*p48c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p48c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p48c4+b**2*(( \
        -1)+ErfCst)*p48c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p48c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p48c3+p48c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p48c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p48c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p48c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p48c3+3* \
        p48c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p48c2+v*(p48c3+p48c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p48c1+p48c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p48c1+v*(p48c2+v*(p48c3+p48c4*v)))))
    P_49 = (1/32)*((-1)*a)**(-9/2)*(b**3*p49c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p49c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p49c4+b**2*(( \
        -1)+ErfCst)*p49c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p49c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p49c3+p49c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p49c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p49c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p49c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p49c3+3* \
        p49c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p49c2+v*(p49c3+p49c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p49c1+p49c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p49c1+v*(p49c2+v*(p49c3+p49c4*v)))))
    P_50 = (1/32)*((-1)*a)**(-9/2)*(b**3*p50c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p50c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p50c4+b**2*(( \
        -1)+ErfCst)*p50c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p50c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p50c3+p50c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p50c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p50c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p50c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p50c3+3* \
        p50c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p50c2+v*(p50c3+p50c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p50c1+p50c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p50c1+v*(p50c2+v*(p50c3+p50c4*v)))))
    P_51 = (1/32)*((-1)*a)**(-9/2)*(b**3*p51c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p51c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p51c4+b**2*(( \
        -1)+ErfCst)*p51c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p51c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p51c3+p51c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p51c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p51c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p51c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p51c3+3* \
        p51c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p51c2+v*(p51c3+p51c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p51c1+p51c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p51c1+v*(p51c2+v*(p51c3+p51c4*v)))))
    P_52 = (1/32)*((-1)*a)**(-9/2)*(b**3*p52c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p52c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p52c4+b**2*(( \
        -1)+ErfCst)*p52c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p52c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p52c3+p52c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p52c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p52c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p52c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p52c3+3* \
        p52c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p52c2+v*(p52c3+p52c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p52c1+p52c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p52c1+v*(p52c2+v*(p52c3+p52c4*v)))))
    P_53 = (1/32)*((-1)*a)**(-9/2)*(b**3*p53c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p53c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p53c4+b**2*(( \
        -1)+ErfCst)*p53c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p53c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p53c3+p53c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p53c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p53c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p53c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p53c3+3* \
        p53c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p53c2+v*(p53c3+p53c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p53c1+p53c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p53c1+v*(p53c2+v*(p53c3+p53c4*v)))))
    P_54 = (1/32)*((-1)*a)**(-9/2)*(b**3*p54c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p54c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p54c4+b**2*(( \
        -1)+ErfCst)*p54c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p54c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p54c3+p54c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p54c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p54c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p54c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p54c3+3* \
        p54c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p54c2+v*(p54c3+p54c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p54c1+p54c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p54c1+v*(p54c2+v*(p54c3+p54c4*v)))))
    P_55 = (1/32)*((-1)*a)**(-9/2)*(b**3*p55c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p55c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p55c4+b**2*(( \
        -1)+ErfCst)*p55c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p55c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p55c3+p55c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p55c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p55c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p55c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p55c3+3* \
        p55c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p55c2+v*(p55c3+p55c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p55c1+p55c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p55c1+v*(p55c2+v*(p55c3+p55c4*v)))))
    P_56 = (1/32)*((-1)*a)**(-9/2)*(b**3*p56c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p56c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p56c4+b**2*(( \
        -1)+ErfCst)*p56c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p56c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p56c3+p56c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p56c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p56c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p56c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p56c3+3* \
        p56c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p56c2+v*(p56c3+p56c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p56c1+p56c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p56c1+v*(p56c2+v*(p56c3+p56c4*v)))))
    P_57 = (1/32)*((-1)*a)**(-9/2)*(b**3*p57c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p57c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p57c4+b**2*(( \
        -1)+ErfCst)*p57c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p57c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p57c3+p57c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p57c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p57c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p57c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p57c3+3* \
        p57c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p57c2+v*(p57c3+p57c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p57c1+p57c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p57c1+v*(p57c2+v*(p57c3+p57c4*v)))))
    P_58 = (1/32)*((-1)*a)**(-9/2)*(b**3*p58c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p58c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p58c4+b**2*(( \
        -1)+ErfCst)*p58c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p58c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p58c3+p58c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p58c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p58c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p58c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p58c3+3* \
        p58c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p58c2+v*(p58c3+p58c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p58c1+p58c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p58c1+v*(p58c2+v*(p58c3+p58c4*v)))))
    P_59 = (1/32)*((-1)*a)**(-9/2)*(b**3*p59c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p59c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p59c4+b**2*(( \
        -1)+ErfCst)*p59c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p59c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p59c3+p59c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p59c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p59c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p59c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p59c3+3* \
        p59c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p59c2+v*(p59c3+p59c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p59c1+p59c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p59c1+v*(p59c2+v*(p59c3+p59c4*v)))))
    P_60 = (1/32)*((-1)*a)**(-9/2)*(b**3*p60c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p60c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p60c4+b**2*(( \
        -1)+ErfCst)*p60c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p60c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p60c3+p60c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p60c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p60c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p60c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p60c3+3* \
        p60c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p60c2+v*(p60c3+p60c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p60c1+p60c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p60c1+v*(p60c2+v*(p60c3+p60c4*v)))))
    P_61 = (1/32)*((-1)*a)**(-9/2)*(b**3*p61c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p61c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p61c4+b**2*(( \
        -1)+ErfCst)*p61c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p61c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p61c3+p61c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p61c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p61c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p61c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p61c3+3* \
        p61c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p61c2+v*(p61c3+p61c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p61c1+p61c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p61c1+v*(p61c2+v*(p61c3+p61c4*v)))))
    P_62 = (1/32)*((-1)*a)**(-9/2)*(b**3*p62c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p62c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p62c4+b**2*(( \
        -1)+ErfCst)*p62c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p62c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p62c3+p62c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p62c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p62c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p62c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p62c3+3* \
        p62c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p62c2+v*(p62c3+p62c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p62c1+p62c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p62c1+v*(p62c2+v*(p62c3+p62c4*v)))))
    P_63 = (1/32)*((-1)*a)**(-9/2)*(b**3*p63c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p63c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p63c4+b**2*(( \
        -1)+ErfCst)*p63c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p63c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p63c3+p63c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p63c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p63c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p63c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p63c3+3* \
        p63c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p63c2+v*(p63c3+p63c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p63c1+p63c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p63c1+v*(p63c2+v*(p63c3+p63c4*v)))))
    P_64 = (1/32)*((-1)*a)**(-9/2)*(b**3*p64c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p64c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p64c4+b**2*(( \
        -1)+ErfCst)*p64c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p64c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p64c3+p64c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p64c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p64c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p64c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p64c3+3* \
        p64c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p64c2+v*(p64c3+p64c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p64c1+p64c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p64c1+v*(p64c2+v*(p64c3+p64c4*v)))))
    P_65 = (1/32)*((-1)*a)**(-9/2)*(b**3*p65c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p65c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p65c4+b**2*(( \
        -1)+ErfCst)*p65c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p65c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p65c3+p65c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p65c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p65c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p65c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p65c3+3* \
        p65c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p65c2+v*(p65c3+p65c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p65c1+p65c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p65c1+v*(p65c2+v*(p65c3+p65c4*v)))))
    P_66 = (1/32)*((-1)*a)**(-9/2)*(b**3*p66c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p66c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p66c4+b**2*(( \
        -1)+ErfCst)*p66c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p66c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p66c3+p66c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p66c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p66c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p66c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p66c3+3* \
        p66c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p66c2+v*(p66c3+p66c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p66c1+p66c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p66c1+v*(p66c2+v*(p66c3+p66c4*v)))))
    P_67 = (1/32)*((-1)*a)**(-9/2)*(b**3*p67c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p67c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p67c4+b**2*(( \
        -1)+ErfCst)*p67c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p67c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p67c3+p67c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p67c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p67c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p67c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p67c3+3* \
        p67c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p67c2+v*(p67c3+p67c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p67c1+p67c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p67c1+v*(p67c2+v*(p67c3+p67c4*v)))))
    P_68 = (1/32)*((-1)*a)**(-9/2)*(b**3*p68c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p68c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p68c4+b**2*(( \
        -1)+ErfCst)*p68c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p68c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p68c3+p68c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p68c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p68c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p68c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p68c3+3* \
        p68c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p68c2+v*(p68c3+p68c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p68c1+p68c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p68c1+v*(p68c2+v*(p68c3+p68c4*v)))))
    P_69 = (1/32)*((-1)*a)**(-9/2)*(b**3*p69c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p69c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p69c4+b**2*(( \
        -1)+ErfCst)*p69c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p69c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p69c3+p69c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p69c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p69c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p69c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p69c3+3* \
        p69c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p69c2+v*(p69c3+p69c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p69c1+p69c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p69c1+v*(p69c2+v*(p69c3+p69c4*v)))))
    P_70 = (1/32)*((-1)*a)**(-9/2)*(b**3*p70c4*((-2)*((-1)*a)**(1/2)* \
        expCst+(-1)*b*((-1)+ErfCst)*np.pi**(1/2))+(-16)*a**4*((-1)+ErfCst)* \
        p70c0*np.pi**(1/2)+2*a*b*(10*((-1)*a)**(1/2)*expCst*p70c4+b**2*(( \
        -1)+ErfCst)*p70c3*np.pi**(1/2)+6*b*((-1)+ErfCst)*p70c4*np.pi**(1/2)+2*( \
        (-1)*a)**(1/2)*b*expCst*(p70c3+p70c4*v))+(-4)*a**2*(b**2*((-1)+ \
        ErfCst)*p70c2*np.pi**(1/2)+3*b*((-1)+ErfCst)*p70c3*np.pi**(1/2)+3*((-1) \
        +ErfCst)*p70c4*np.pi**(1/2)+2*((-1)*a)**(1/2)*expCst*(2*p70c3+3* \
        p70c4*v)+2*((-1)*a)**(1/2)*b*expCst*(p70c2+v*(p70c3+p70c4*v)))+ \
        8*a**3*(((-1)+ErfCst)*(b*p70c1+p70c2)*np.pi**(1/2)+2*((-1)*a)**(1/2) \
        *expCst*(p70c1+v*(p70c2+v*(p70c3+p70c4*v)))))
        
    # store to output
    # P_output = np.vstack((P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9, P_10, P_11, P_12, P_13,  
    #     P_14, P_15, P_16, P_17, P_18, P_19, P_20, P_21, P_22, P_23, P_24, P_25,  
    #     P_26, P_27, P_28, P_29, P_30, P_31, P_32, P_33, P_34, P_35, P_36, P_37,  
    #     P_38, P_39, P_40, P_41, P_42, P_43, P_44, P_45, P_46, P_47, P_48, P_49,  
    #     P_50, P_51, P_52, P_53, P_54, P_55, P_56, P_57, P_58, P_59, P_60, P_61,  
    #     P_62, P_63, P_64, P_65, P_66, P_67, P_68, P_69, P_70))
    P_output = np.zeros((n_pts_v, n_sites, 70))
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
    P_output[:,:,35] = P_36
    P_output[:,:,36] = P_37
    P_output[:,:,37] = P_38
    P_output[:,:,38] = P_39
    P_output[:,:,39] = P_40
    P_output[:,:,40] = P_41
    P_output[:,:,41] = P_42
    P_output[:,:,42] = P_43
    P_output[:,:,43] = P_44
    P_output[:,:,44] = P_45
    P_output[:,:,45] = P_46
    P_output[:,:,46] = P_47
    P_output[:,:,47] = P_48
    P_output[:,:,48] = P_49
    P_output[:,:,49] = P_50
    P_output[:,:,50] = P_51
    P_output[:,:,51] = P_52
    P_output[:,:,52] = P_53
    P_output[:,:,53] = P_54
    P_output[:,:,54] = P_55
    P_output[:,:,55] = P_56
    P_output[:,:,56] = P_57
    P_output[:,:,57] = P_58
    P_output[:,:,58] = P_59
    P_output[:,:,59] = P_60
    P_output[:,:,60] = P_61
    P_output[:,:,61] = P_62
    P_output[:,:,62] = P_63
    P_output[:,:,63] = P_64
    P_output[:,:,64] = P_65
    P_output[:,:,65] = P_66
    P_output[:,:,66] = P_67
    P_output[:,:,67] = P_68
    P_output[:,:,68] = P_69
    P_output[:,:,69] = P_70
    
    # return
    return P_output
