%% Input parameters!!


% SA

mean_of_mu_SA = 0;

sigma_of_mu_SA = 0.5 *1;

sigma_SA = 0.6;



% DV

muZ = -0.46 * 1;

sigmaMuZ = 0.5 * 1;

sigmaZ = 0.6;



%%% PC Parameters

PC_order = 4;

KL_dim = 2;

num_pc_terms_total = nchoosek(PC_order + KL_dim, KL_dim);


%%% Number of Monte-Carlo samples!!

N_real_xi = 1000;

xi_samples = normrnd(0, 1, N_real_xi, KL_dim);



%%% Numerical integration parameters!!

num_pts_SA = 1000;



%% Domains of models!!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SA!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_vector_SA = linspace(0, 1, num_pts_SA);

SA_bound_left = exp(mean_of_mu_SA - 5*sigma_SA -5*sigma_of_mu_SA);
SA_bound_right = exp(mean_of_mu_SA + 3*sigma_SA + 3*sigma_of_mu_SA);

SA_domain_vector = SA_bound_left + x_vector_SA.*(SA_bound_right - SA_bound_left); 
delta_SA = SA_domain_vector(2) - SA_domain_vector(1);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computes PC Coefficients!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IM parameters

muY = mean_of_mu_SA;
sigmaMuY = sigma_of_mu_SA;
sigmaY = sigma_SA;




% PC Table

index_PC_table_total = index_table_function(KL_dim, PC_order);

Hermite_PC_Table = zeros(num_pc_terms_total, N_real_xi);

% Y: xi_1
% Z: xi_2

for i = 1:num_pc_terms_total
        
        Hermite_PC_Table(i, :) = Hermite_Proba_new(xi_samples(:, 2)', index_PC_table_total(i,1)).*...
                                 Hermite_Proba_new(xi_samples(:, 1)', index_PC_table_total(i,2));

end




%% Computes PC Coefficients PDF IM!!


delta_SA_vector = [diff(SA_domain_vector) (SA_domain_vector(end) - SA_domain_vector(end-1))];


PC_Coeffs_PDF_Y = cell(1, 5);


a = -sigmaMuY^2/(2*sigmaY^2) - 1/2;

b =  (log(SA_domain_vector) - muY) * sigmaMuY/(sigmaY^2);

c = - (log(SA_domain_vector) - muY).^2/(2*sigmaY^2);

alpha = 1./(sigmaY*SA_domain_vector) * 1/(1 *2*sqrt(pi)) .* exp(c - b.^2/(4*a));

             


PC_Coeffs_PDF_Y{1} = 1/1 * (1/sqrt(-a)) * alpha;


PC_Coeffs_PDF_Y{2} =  1/1 * alpha .*b/(2*(-a)^(3/2));
                  
                  
PC_Coeffs_PDF_Y{3} = 1/2 * alpha .*(-2*a*(1 + 2*a) + b.^2)/(4*(-a)^(5/2));


PC_Coeffs_PDF_Y{4} = 1/6 * alpha .* (-b).*(6*a*(1 + 2*a) - b.^2)/(8*(-a)^(7/2));


PC_Coeffs_PDF_Y{5} = 1/24 * alpha .* (12*a^2*(1+ 2*a)^2 - 12*a*(1 + 2*a)*b.^2 + b.^4)/(16*(-a)^(9/2));




%% Computes PC Coefficients CDF DV!!


a = -sigmaMuZ^2/(2*sigmaZ^2) - 1/2;

b =  (log(SA_domain_vector) - muZ) * sigmaMuZ/(sigmaZ^2);

c = - (log(SA_domain_vector) - muZ).^2/(2*sigmaZ^2);

alpha = sigmaMuZ/(sigmaZ *2*sqrt(pi)) * exp(c - b.^2/(4*a));

             

PC_Coeffs_CDF_Z = cell(1, PC_order + 1);


PC_Coeffs_CDF_Z{1} = 1/1 * (normcdf((log(SA_domain_vector) - muZ)/sqrt(sigmaMuZ^2 + sigmaZ^2)));


PC_Coeffs_CDF_Z{2} = 1/1 * (1/sqrt(-a)) * alpha;


PC_Coeffs_CDF_Z{3} =  1/2 * alpha .*b/(2*(-a)^(3/2));
                  
                  
PC_Coeffs_CDF_Z{4} = 1/6 * alpha .*(-2*a*(1 + 2*a) + b.^2)/(4*(-a)^(5/2));


PC_Coeffs_CDF_Z{5} = 1/24 * alpha .* (-b).*(6*a*(1 + 2*a) - b.^2)/(8*(-a)^(7/2));




%% Performs First Integration over PC terms!!

PC_Coeffs_CDF_PDF_Numerical_Integration = zeros(num_pc_terms_total, 1);

for i = 1:num_pc_terms_total % cross product then sum/integration

    PC_Coeffs_CDF_PDF_Numerical_Integration(i) = sum(PC_Coeffs_CDF_Z{index_PC_table_total(i, 1)+1}.*...
                                                     PC_Coeffs_PDF_Y{index_PC_table_total(i, 2)+1}.*...
                                                     delta_SA_vector);


end








%% ANALYTICAL PC SOLUTION!!

PC_Coeffs_CDF_PDF_Analytical_Integration=PC_Coeffs_First_Integration_New_Models_function(muY, sigmaMuY, sigmaY,  ...
  muZ, sigmaMuZ, sigmaZ);

PC_Coeffs_CDF_PDF_Analytical_Integration = cell2mat(PC_Coeffs_CDF_PDF_Analytical_Integration);


%% Generates samples!!

%%% Samples from PC Numerical Integration!!


CDF_Samples_From_PC_Numerical_Integration = zeros(1, N_real_xi);

for i = 1:num_pc_terms_total

    CDF_Samples_From_PC_Numerical_Integration = CDF_Samples_From_PC_Numerical_Integration + ...
                                                PC_Coeffs_CDF_PDF_Numerical_Integration(i) * ...
                                                Hermite_PC_Table(i, :);
end


%%% Samples from PC Analytical Integration!!


CDF_Samples_From_PC_Analytical_Integration = zeros(1, N_real_xi);

for i = 1:num_pc_terms_total

    CDF_Samples_From_PC_Analytical_Integration = CDF_Samples_From_PC_Analytical_Integration + ...
                                                 PC_Coeffs_CDF_PDF_Analytical_Integration(i) * Hermite_PC_Table(i, :);
end





%% Histogram of Final Results!! 

pct_low = 10;
pct_middle = 50;
pct_high = 90;



fs = 17;
lw = 3;


figure
h1 = histogram(CDF_Samples_From_PC_Numerical_Integration);
h1_bins = h1.BinEdges;

hold on
h2 = histogram(CDF_Samples_From_PC_Analytical_Integration);
h2.BinEdges = h1_bins;

set(gca, 'Fontsize', fs)
xlabel('Probability of Rupture')

legend('Numerical', 'Analytical')

disp(strcat(num2str(pct_low), 'th percentile,', " ", ...
            num2str(pct_middle), 'th percentile,', " ", ...
            'mean,', " ",...
            num2str(pct_high), 'th percentile'))


disp('Percentiles Samples from PC Numerical Integration')

[prctile(CDF_Samples_From_PC_Numerical_Integration, pct_low),...
 prctile(CDF_Samples_From_PC_Numerical_Integration, pct_middle),...
 mean(CDF_Samples_From_PC_Numerical_Integration),...
 prctile(CDF_Samples_From_PC_Numerical_Integration, pct_high)]

disp('Percentiles Samples from PC Analytical Integration')

[prctile(CDF_Samples_From_PC_Analytical_Integration, pct_low),...
 prctile(CDF_Samples_From_PC_Analytical_Integration, pct_middle),...
 mean(CDF_Samples_From_PC_Analytical_Integration), ...
 prctile(CDF_Samples_From_PC_Analytical_Integration, pct_high)]

