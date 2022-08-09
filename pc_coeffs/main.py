import numpy as np
import pc
import time

def main():

    ## Checks run time of PC Coeffs!!

    n = 1000* 1
    num_pts_v = 10
    lnv_vector = np.logspace(0, 2, num_pts_v)
    n_event = 100

    ### Scalar values at one site

    muY = 0
    sigmaMuY = 0.4
    sigmaY = 0.6
    amuZ = 1.7367
    bmuZ = 5.1508
    sigmaMuZ = 0.5
    sigmaZ = 0.67
    amuT = 0.2736
    bmuT = -2.6315
    sigmaMuT = 0.5
    sigmaT = 0.7
    amuV = 0.3201
    bmuV = 0.0356
    sigmaMuV = 0.4
    sigmaV = 0.6

    ### Vectorized inputs
    muY_vector = muY * np.ones((1, n))
    sigmaMuY_vector = sigmaMuY * np.ones((1, n))
    sigmaY_vector = sigmaY * np.ones((1, n))
    amuZ_vector = amuZ * np.ones((1, n))
    bmuZ_vector = bmuZ * np.ones((1, n))
    sigmaMuZ_vector = sigmaMuZ * np.ones((1, n))
    sigmaZ_vector = sigmaZ * np.ones((1, n))
    amuT_vector = amuT * np.ones((1, n))
    bmuT_vector = bmuT * np.ones((1, n))
    sigmaMuT_vector = sigmaMuT * np.ones((1, n))
    sigmaT_vector = sigmaT * np.ones((1, n))

    amuV_vector = amuV * np.ones((1, n))
    bmuV_vector = bmuV * np.ones((1, n))
    sigmaMuV_vector = sigmaMuV * np.ones((1, n))
    sigmaV_vector = sigmaV * np.ones((1, n))

    for i in range(n_event):
        # PC_Coeffs_Integrated_array = pc.PC_Coeffs_CDF_Risk_array_function(
        pc.PC_Coeffs_CDF_Risk_array_function(
            lnv_vector, muY_vector, sigmaMuY_vector, sigmaY_vector, \
            amuZ_vector, bmuZ_vector, sigmaMuZ_vector, sigmaZ_vector, \
            amuT_vector, bmuT_vector, sigmaMuT_vector, sigmaT_vector, \
            amuV_vector, bmuV_vector, sigmaMuV_vector, sigmaV_vector
        )
    
if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print(f'Time = {round(time_end-time_start,1)}')