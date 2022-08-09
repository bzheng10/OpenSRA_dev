%% Input parameters!!


Mag_scenario = 7;


% SA!!

mean_of_mu_SA = -2;

sigma_of_mu_SA = 0.3 * 1;

sigma_SA = 0.6;



% Disp!!

sigma_of_mu_disp = 0.3*1;

point_of_tangent_disp = mean_of_mu_SA - 0.0;


% DV!!


muT = 0 * 1;

sigmaMuT = 0.3 * 1;

sigmaT = 0.7;



%%% PC Parameters

PC_order = 4;

KL_dim = 3;

num_pc_terms_total = nchoosek(PC_order + KL_dim, KL_dim);


%%% Number of Monte-Carlo samples!!

N_real_xi = 1000;

xi_samples = normrnd(0, 1, N_real_xi, KL_dim);



%%% Numerical integration parameters!!

num_pts_SA = 1E4;

num_pts_disp = 1E4;

%% Domains of models!!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SA!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_vector_SA = linspace(0, 1, num_pts_SA);

SA_bound_left = exp(mean_of_mu_SA - 5*sigma_SA -4*sigma_of_mu_SA);
SA_bound_right = exp(mean_of_mu_SA + 2.5*sigma_SA + 2*sigma_of_mu_SA);

SA_domain_vector = logspace(log10(SA_bound_left), log10(SA_bound_right), num_pts_SA); 

delta_SA_vector = [diff(SA_domain_vector) (SA_domain_vector(end) - SA_domain_vector(end-1))];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EDP!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input parameters

mean_ky = 0.1; 

mean_Ts = 1.0;

[mean_of_mu_disp_vector, sigma_disp_vector] = Bray_Macedo_2018(mean_ky, mean_Ts, SA_domain_vector, Mag_scenario, mean_of_mu_SA, 'quadratic');

sigma_of_mu_disp_vector = sigma_of_mu_disp * ones(1, num_pts_SA);

% Domain vector!!

disp_bound_left = min(exp(mean_of_mu_disp_vector -  5*sigma_disp_vector - 5*sigma_of_mu_disp_vector));
disp_bound_right = max(exp(mean_of_mu_disp_vector +  3*sigma_disp_vector + 2*sigma_of_mu_disp_vector));

disp_domain_vector = logspace(log10(disp_bound_left), log10(disp_bound_right), num_pts_disp); 
delta_disp_vector = [diff(disp_domain_vector) (disp_domain_vector(end) - disp_domain_vector(end-1))];

% Tangent!!

if mean_Ts <0.05
  coeff1 = -0.22;
  coeff2 = 0.0;
else
  coeff1 = -1.10;
  coeff2 = 1.50;
end

slope_disp = 0.566*log(mean_ky) + 3.04 - 2*0.244*point_of_tangent_disp;

intercept_disp = coeff1-2.83*log(mean_ky)-0.333*(log(mean_ky)).^2+0.566*log(mean_ky).*point_of_tangent_disp+3.04*point_of_tangent_disp-...
                    0.244*(point_of_tangent_disp).^2 + coeff2*mean_Ts+0.278*1*(Mag_scenario-7); 

tangent_disp_vector = slope_disp *  (log(SA_domain_vector) - point_of_tangent_disp) + intercept_disp;



% Plots model and tangent!!

% fs = 17;
% lw = 2;

% figure
% plot(log(SA_domain_vector), mean_of_mu_disp_vector, 'Linewidth', lw)
% hold on
% plot(log(SA_domain_vector), tangent_disp_vector, 'Linewidth', lw)
% % plot(mean_of_mu_SA, intercept_disp, 'or', 'Linewidth', lw)
% plot(point_of_tangent_disp, intercept_disp, 'or', 'Linewidth', lw)
% % plot(log(SA_domain_vector), p_coeffs(1)*log(SA_domain_vector) + p_coeffs(2), 'Linewidth', lw)
% 
% xlabel('ln SA')
% ylabel('Median ln Disp')
% legend('Bray & Macedo 2018', 'Tangent at m_{\mu}(SA)','Point of Tangent')%, 'Linear Regression')
% set(gca, 'Fontsize', fs)  

% Computes where to take the tangent of disp for the DM model!!

[mean_of_mu_disp, ~] = Bray_Macedo_2018(mean_ky, mean_Ts, exp(mean_of_mu_SA), Mag_scenario, mean_of_mu_SA, 'quadratic');






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computes PC Coefficients!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IM parameters

muY = mean_of_mu_SA;
sigmaMuY = sigma_of_mu_SA;
sigmaY = sigma_SA;


% EDP parameters

amuZ = slope_disp;
bmuZ = slope_disp.*(-mean_of_mu_SA) + intercept_disp;
sigmaMuZ = sigma_of_mu_disp_vector(1);
sigmaZ = sigma_disp_vector(1);



% PC Table

index_PC_table_KLdim_3 = index_table_function(KL_dim, PC_order);

Hermite_PC_Table = zeros(num_pc_terms_total, N_real_xi);

% Y: xi1
% Z: xi2
% T: xi3

for i = 1:num_pc_terms_total
        
        Hermite_PC_Table(i, :) = Hermite_Proba_new(xi_samples(:, 3)', index_PC_table_KLdim_3(i,1)).*...
                                 Hermite_Proba_new(xi_samples(:, 2)', index_PC_table_KLdim_3(i,2)).*...
                                 Hermite_Proba_new(xi_samples(:, 1)', index_PC_table_KLdim_3(i,3));

end




%% Computes PC Coefficients of First Integration!!


P_output_1 = PC_Coeffs_Simpler_First_Integration_function(muY, sigmaMuY, ...
   sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ);

a2=(-1/2).*(sigmaMuZ.^2+amuZ.^2.*(sigmaMuY.^2+sigmaY.^2)+sigmaZ.^2).^(-1); ...
  
b2=(bmuZ+amuZ.*muY).*(sigmaMuZ.^2+amuZ.^2.*(sigmaMuY.^2+sigmaY.^2)+ ...
  sigmaZ.^2).^(-1);
c2=(-1/2).*(bmuZ+amuZ.*muY).^2.*(sigmaMuZ.^2+amuZ.^2.*(sigmaMuY.^2+ ...
  sigmaY.^2)+sigmaZ.^2).^(-1);


alphaZ = 1./disp_domain_vector .* exp(a2*log(disp_domain_vector).^2 + b2 * log(disp_domain_vector) + c2);


index_PC_table_KLdim_2 = index_table_function(KL_dim-1, PC_order);
num_pc_terms_KLdim_2 = size(index_PC_table_KLdim_2, 1);




PC_Coeffs_PDF_ZY = cell(num_pc_terms_KLdim_2, 1);

p1c0=P_output_1(1, :);
p2c0=P_output_1(2, :);
p2c1=P_output_1(3, :);
p3c0=P_output_1(4, :);
p3c1=P_output_1(5, :);
p4c0=P_output_1(6, :);
p4c1=P_output_1(7, :);
p4c2=P_output_1(8, :);
p5c0=P_output_1(9, :);
p5c1=P_output_1(10, :);
p5c2=P_output_1(11, :);
p6c0=P_output_1(12, :);
p6c1=P_output_1(13, :);
p6c2=P_output_1(14, :);
p7c0=P_output_1(15, :);
p7c1=P_output_1(16, :);
p7c2=P_output_1(17, :);
p7c3=P_output_1(18, :);
p8c0=P_output_1(19, :);
p8c1=P_output_1(20, :);
p8c2=P_output_1(21, :);
p8c3=P_output_1(22, :);
p9c0=P_output_1(23, :);
p9c1=P_output_1(24, :);
p9c2=P_output_1(25, :);
p9c3=P_output_1(26, :);
p10c0=P_output_1(27, :);
p10c1=P_output_1(28, :);
p10c2=P_output_1(29, :);
p10c3=P_output_1(30, :);
p11c0=P_output_1(31, :);
p11c1=P_output_1(32, :);
p11c2=P_output_1(33, :);
p11c3=P_output_1(34, :);
p11c4=P_output_1(35, :);
p12c0=P_output_1(36, :);
p12c1=P_output_1(37, :);
p12c2=P_output_1(38, :);
p12c3=P_output_1(39, :);
p12c4=P_output_1(40, :);
p13c0=P_output_1(41, :);
p13c1=P_output_1(42, :);
p13c2=P_output_1(43, :);
p13c3=P_output_1(44, :);
p13c4=P_output_1(45, :);
p14c0=P_output_1(46, :);
p14c1=P_output_1(47, :);
p14c2=P_output_1(48, :);
p14c3=P_output_1(49, :);
p14c4=P_output_1(50, :);
p15c0=P_output_1(51, :);
p15c1=P_output_1(52, :);
p15c2=P_output_1(53, :);
p15c3=P_output_1(54, :);
p15c4=P_output_1(55, :);


%%% Remultiplies by alphaZ after integration!!

PC_Coeffs_PDF_ZY{1} = polyval(p1c0, log(disp_domain_vector)).*alphaZ;

PC_Coeffs_PDF_ZY{2} = polyval([p2c1 p2c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{3} = polyval([p3c1 p3c0], log(disp_domain_vector)).*alphaZ;

PC_Coeffs_PDF_ZY{4} = polyval([p4c2 p4c1 p4c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{5} = polyval([p5c2 p5c1 p5c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{6} = polyval([p6c2 p6c1 p6c0], log(disp_domain_vector)).*alphaZ;

PC_Coeffs_PDF_ZY{7} = polyval([p7c3 p7c2 p7c1 p7c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{8} = polyval([p8c3 p8c2 p8c1 p8c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{9} = polyval([p9c3 p9c2 p9c1 p9c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{10} = polyval([p10c3 p10c2 p10c1 p10c0], log(disp_domain_vector)).*alphaZ;

PC_Coeffs_PDF_ZY{11} = polyval([p11c4 p11c3 p11c2 p11c1 p11c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{12} = polyval([p12c4 p12c3 p12c2 p12c1 p12c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{13} = polyval([p13c4 p13c3 p13c2 p13c1 p13c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{14} = polyval([p14c4 p14c3 p14c2 p14c1 p14c0], log(disp_domain_vector)).*alphaZ;
PC_Coeffs_PDF_ZY{15} = polyval([p15c4 p15c3 p15c2 p15c1 p15c0], log(disp_domain_vector)).*alphaZ;



%% Generates samples of First Integration PC Coefficients!!


PDF_ZY_samples = zeros(length(disp_domain_vector), N_real_xi);



for i = 1:num_pc_terms_KLdim_2
        
   
    PDF_ZY_samples = PDF_ZY_samples + ...
                    PC_Coeffs_PDF_ZY{i}'.* ...
                    Hermite_Proba_new(xi_samples(:, 2)', index_PC_table_KLdim_2(i,1)).*...
                    Hermite_Proba_new(xi_samples(:, 1)', index_PC_table_KLdim_2(i,2));

end

%% PC Coefficients of CDF DV!!



a = -sigmaMuT^2/(2*sigmaT^2) - 1/2;

b =  (log(disp_domain_vector) - muT) * sigmaMuT/(sigmaT^2);

c = - (log(disp_domain_vector) - muT).^2/(2*sigmaT^2);

alpha = sigmaMuT/(sigmaT *2*sqrt(pi)) * exp(c - b.^2/(4*a));

             

PC_Coeffs_CDF_T = cell(1, PC_order + 1);


PC_Coeffs_CDF_T{1} = 1/1 * normcdf((log(disp_domain_vector) - muT)/sqrt(sigmaMuT^2 + sigmaT^2));


PC_Coeffs_CDF_T{2} = 1/1 * (1/sqrt(-a)) * alpha;


PC_Coeffs_CDF_T{3} = 1/2 * alpha .*b/(2*(-a)^(3/2));
                  
                  
PC_Coeffs_CDF_T{4} = 1/6 * alpha .*(-2*a*(1 + 2*a) + b.^2)/(4*(-a)^(5/2));


PC_Coeffs_CDF_T{5} = 1/24 * alpha .* (-b).*(6*a*(1 + 2*a) - b.^2)/(8*(-a)^(7/2));





%%% Generates samples!!

CDF_DV_PC_samples = zeros(length(disp_domain_vector), N_real_xi);



for i = 1:PC_order + 1 
   

    CDF_DV_PC_samples = CDF_DV_PC_samples + ...
                        PC_Coeffs_CDF_T{i}'.*...
                        Hermite_Proba(xi_samples(:, 3), i-1)';

end




% %%% Monte-Carlo Check!! OK!!
% 
% CDF_DV_MC_samples = zeros(length(disp_domain_vector), N_real_xi);
% 
% for mc = 1:N_real_xi
% 
%     CDF_DV_MC_samples(:, mc) = normcdf((log(disp_domain_vector) - (muT + sigmaMuT*xi_samples(mc, 3)))/sigmaT);
% 
% end


%% Performs Second integration over SAMPLES EDP values!!

CDF_DV_Integrated_DM_PC_samples = zeros(1, N_real_xi);

for i = 1:N_real_xi

    CDF_DV_Integrated_DM_PC_samples(i)= sum(PDF_ZY_samples(:, i).*...
                                            CDF_DV_PC_samples(:, i).*...
                                            delta_disp_vector', 1);

end



%% Performs Second Integration over PC terms!!



index_PC_table_KLdim_2to3(1, :) = [1, 1];
index_PC_table_KLdim_2to3(2, :) = [2, 1];
index_PC_table_KLdim_2to3(3, :) = [1, 2];
index_PC_table_KLdim_2to3(4, :) = [1, 3];
index_PC_table_KLdim_2to3(5, :) = [3, 1];
index_PC_table_KLdim_2to3(6, :) = [2, 2];
index_PC_table_KLdim_2to3(7, :) = [2, 3];
index_PC_table_KLdim_2to3(8, :) = [1, 4];
index_PC_table_KLdim_2to3(9, :) = [1, 5];
index_PC_table_KLdim_2to3(10, :) = [1, 6];
index_PC_table_KLdim_2to3(11, :) = [4, 1];
index_PC_table_KLdim_2to3(12, :) = [3, 2];
index_PC_table_KLdim_2to3(13, :) = [3, 3];
index_PC_table_KLdim_2to3(14, :) = [2, 4];
index_PC_table_KLdim_2to3(15, :) = [2, 5];
index_PC_table_KLdim_2to3(16, :) = [2, 6];
index_PC_table_KLdim_2to3(17, :) = [1, 7];
index_PC_table_KLdim_2to3(18, :) = [1, 8];
index_PC_table_KLdim_2to3(19, :) = [1, 9];
index_PC_table_KLdim_2to3(20, :) = [1, 10];
index_PC_table_KLdim_2to3(21, :) = [5, 1];
index_PC_table_KLdim_2to3(22, :) = [4, 2];
index_PC_table_KLdim_2to3(23, :) = [4, 3];
index_PC_table_KLdim_2to3(24, :) = [3, 4];
index_PC_table_KLdim_2to3(25, :) = [3, 5];
index_PC_table_KLdim_2to3(26, :) = [3, 6];
index_PC_table_KLdim_2to3(27, :) = [2, 7];
index_PC_table_KLdim_2to3(28, :) = [2, 8];
index_PC_table_KLdim_2to3(29, :) = [2, 9];
index_PC_table_KLdim_2to3(30, :) = [2, 10];
index_PC_table_KLdim_2to3(31, :) = [1, 11];
index_PC_table_KLdim_2to3(32, :) = [1, 12];
index_PC_table_KLdim_2to3(33, :) = [1, 13];
index_PC_table_KLdim_2to3(34, :) = [1, 14];
index_PC_table_KLdim_2to3(35, :) = [1, 15];


%% Numerical Integration of Second Integral over PC Coefficients only!!


PC_Coeffs_CDF_PDF_Numerical_Integration = zeros(num_pc_terms_total, 1);

for i = 1:num_pc_terms_total

    PC_Coeffs_CDF_PDF_Numerical_Integration(i) = sum(PC_Coeffs_CDF_T{index_PC_table_KLdim_2to3(i, 1)}.*...
                                                     PC_Coeffs_PDF_ZY{index_PC_table_KLdim_2to3(i, 2)}.*...
                                                     delta_disp_vector);


end



%%% Samples from PC Numerical Integration!!


CDF_Samples_From_PC_Numerical_Integration = zeros(1, N_real_xi);

for i = 1:num_pc_terms_total

    CDF_Samples_From_PC_Numerical_Integration = CDF_Samples_From_PC_Numerical_Integration + ...
                                                PC_Coeffs_CDF_PDF_Numerical_Integration(i) * ...
                                                Hermite_PC_Table(i, :);
end




%% ANALYTICAL PC SOLUTION of Double Integration!!

PC_Coeffs_CDF_PDF_Analytical_Integration = PC_Coeffs_Second_Integration_New_Models_function(muY, sigmaMuY, sigmaY,  ...
                                           amuZ, bmuZ, sigmaMuZ, sigmaZ, muT, sigmaMuT, sigmaT);


PC_Coeffs_CDF_PDF_Analytical_Integration = cell2mat(PC_Coeffs_CDF_PDF_Analytical_Integration);



%%% Samples from PC Analytical Integration!!


CDF_Samples_From_PC_Analytical_Integration = zeros(1, N_real_xi);

for i = 1:num_pc_terms_total

    CDF_Samples_From_PC_Analytical_Integration = CDF_Samples_From_PC_Analytical_Integration + ...
                                                 PC_Coeffs_CDF_PDF_Analytical_Integration(i) * ...
                                                 Hermite_PC_Table(i, :);
end



%% Gets percentiles!!



pct_low = 10;
pct_middle = 50;
pct_high = 90;



fs = 17;
lw = 2;




%% Histogram of Final Results!! 

figure
h1 = histogram(CDF_DV_Integrated_DM_PC_samples);


h1_bins = h1.BinEdges;

hold on
h2 = histogram(CDF_Samples_From_PC_Analytical_Integration);
h2.BinEdges = h1_bins;


h3 = histogram(CDF_Samples_From_PC_Numerical_Integration);
h3.BinEdges = h1_bins;

set(gca, 'Fontsize', fs)
xlabel('Probability of Rupture')

legend('Numerical from PC Coefficients', 'Analytical', 'Numerical from Second Integration Only')

disp(strcat(num2str(pct_low), 'th percentile,', " ", ...
            num2str(pct_middle), 'th percentile,', " ", ...
            'mean,', " ",...
            num2str(pct_high), 'th percentile'))


disp('Percentiles Samples from PC Numerical Integration')

[prctile(CDF_DV_Integrated_DM_PC_samples, pct_low),...
 prctile(CDF_DV_Integrated_DM_PC_samples, pct_middle),...
 mean(CDF_DV_Integrated_DM_PC_samples),...
 prctile(CDF_DV_Integrated_DM_PC_samples, pct_high)]




disp('Percentiles Samples from PC Analytical Integration')

[prctile(CDF_Samples_From_PC_Analytical_Integration, pct_low),...
 prctile(CDF_Samples_From_PC_Analytical_Integration, pct_middle),...
 mean(CDF_Samples_From_PC_Analytical_Integration), ...
 prctile(CDF_Samples_From_PC_Analytical_Integration, pct_high)]



disp('Samples from Numerical Second Integration Only')


[prctile(CDF_Samples_From_PC_Numerical_Integration, pct_low),...
 prctile(CDF_Samples_From_PC_Numerical_Integration, pct_middle),...
 mean(CDF_Samples_From_PC_Numerical_Integration),...
 prctile(CDF_Samples_From_PC_Numerical_Integration, pct_high)]
