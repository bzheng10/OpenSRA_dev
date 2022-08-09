%% Checks run time of PC Coeffs!!

n = 10000* 1;

num_pts_domain = 10;

% Domain for Triple Integral

dv_bound_left = 1.3759e-04;
dv_bound_right = 32.0686;

dv_domain_vector = logspace(log10(dv_bound_left), log10(dv_bound_right), num_pts_domain);
lnv_vector = log(dv_domain_vector);

%%% Scalar values at one site

muY = 0;
sigmaMuY = 0.4;
sigmaY = 0.6;
amuZ = 1.7367;
bmuZ = 5.1508;
sigmaMuZ = 0.5;
sigmaZ = 0.67;
amuT = 0.2736;
bmuT = -2.6315;
sigmaMuT = 0.5;
sigmaT = 0.7;
amuV = 0.3201;
bmuV = 0.0356;
sigmaMuV = 0.4;
sigmaV = 0.6;

%%% Vectorized inputs

muY_vector = muY * ones(1, n);
sigmaMuY_vector = sigmaMuY * ones(1, n);
sigmaY_vector = sigmaY * ones(1, n);
amuZ_vector = amuZ * ones(1, n);
bmuZ_vector = bmuZ * ones(1, n);
sigmaMuZ_vector = sigmaMuZ * ones(1, n);
sigmaZ_vector = sigmaZ * ones(1, n);
amuT_vector = amuT * ones(1, n);
bmuT_vector = bmuT * ones(1, n);
sigmaMuT_vector = sigmaMuT * ones(1, n);
sigmaT_vector = sigmaT * ones(1, n);

amuV_vector = amuV * ones(1, n);
bmuV_vector = bmuV * ones(1, n);
sigmaMuV_vector = sigmaMuV * ones(1, n);
sigmaV_vector = sigmaV * ones(1, n);


%% Triple Integral


tic
PC_Coeffs_Integrated_array_3 = PC_Coeffs_Simpler_CDF_Triple_Integral_array_function(lnv_vector, muY_vector, sigmaMuY_vector, sigmaY_vector, amuZ_vector, bmuZ_vector, sigmaMuZ_vector, sigmaZ_vector, amuT_vector, bmuT_vector, sigmaMuT_vector, sigmaT_vector, amuV_vector, bmuV_vector, sigmaMuV_vector, sigmaV_vector);
toc