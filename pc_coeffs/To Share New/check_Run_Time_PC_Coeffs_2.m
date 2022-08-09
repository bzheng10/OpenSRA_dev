%% Checks run time of PC Coeffs!!

n = 100* 1;

num_pts_domain = 50;



% Domain for Double Integral

dm_bound_left = 1.3806e-06;
dm_bound_right = 6.2127;

dm_domain_vector = logspace(log10(dm_bound_left), log10(dm_bound_right), num_pts_domain);
lnt_vector = log(dm_domain_vector);


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

%% Double Integral


% tic
% PC_Coeffs_Integrated_2 = PC_Coeffs_Simpler_CDF_Double_Integral_function(lnt_vector, muY_vector, sigmaMuY_vector, sigmaY_vector, amuZ_vector, bmuZ_vector, sigmaMuZ_vector, sigmaZ_vector, amuT_vector, bmuT_vector, sigmaMuT_vector, sigmaT_vector);
% toc

tic
for i = 1:1000
    PC_Coeffs_Integrated_array_2 = PC_Coeffs_Simpler_CDF_Double_Integral_array_function(lnt_vector, muY_vector, sigmaMuY_vector, sigmaY_vector, amuZ_vector, bmuZ_vector, sigmaMuZ_vector, sigmaZ_vector, amuT_vector, bmuT_vector, sigmaMuT_vector, sigmaT_vector);
end
toc