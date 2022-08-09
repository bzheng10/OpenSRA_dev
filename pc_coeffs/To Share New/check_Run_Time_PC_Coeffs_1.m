%% Checks run time of PC Coeffs!!

n = 10000* 1;

num_pts_domain = 100;


% Domain for Single Integral

edp_bound_left = 8.9874e-07;
edp_bound_right = 3.1725e+04;

edp_domain_vector = logspace(log10(edp_bound_left), log10(edp_bound_right), num_pts_domain);
lnz_vector = log(edp_domain_vector);


%%% Scalar values at one site

muY = 0;
sigmaMuY = 0.4;
sigmaY = 0.6;
amuZ = 1.7367;
bmuZ = 5.1508;
sigmaMuZ = 0.5;
sigmaZ = 0.67;

%%% Vectorized inputs

muY_vector = muY * ones(1, n);
sigmaMuY_vector = sigmaMuY * ones(1, n);
sigmaY_vector = sigmaY * ones(1, n);
amuZ_vector = amuZ * ones(1, n);
bmuZ_vector = bmuZ * ones(1, n);
sigmaMuZ_vector = sigmaMuZ * ones(1, n);
sigmaZ_vector = sigmaZ * ones(1, n);


%% Single Integral


% tic
% PC_Coeffs_Integrated_1 = PC_Coeffs_Simpler_CDF_Single_Integral_function(lnz_vector, muY_vector, sigmaMuY_vector, sigmaY_vector, amuZ_vector, bmuZ_vector, sigmaMuZ_vector, sigmaZ_vector);
% toc

tic
PC_Coeffs_Integrated_array_1 = PC_Coeffs_Simpler_CDF_Single_Integral_array_function(lnz_vector, muY_vector, sigmaMuY_vector, sigmaY_vector, amuZ_vector, bmuZ_vector, sigmaMuZ_vector, sigmaZ_vector);
toc
