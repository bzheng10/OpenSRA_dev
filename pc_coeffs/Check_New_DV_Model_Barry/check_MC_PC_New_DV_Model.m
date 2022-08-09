%% Input parameters!!

close all


% model_type = 'linear';
model_type = 'quadratic'; % Keep this

% correlation_type = 'full';
correlation_type = 'independent';


%%% Input scenario!!

Mag_scenario = 3;

mean_of_mu_SA = log(10);


%%% Epistemic uncertainty and Aleatory variablity!!

% SA!!


sigma_of_mu_SA = 0.25 *1;

sigma_SA = 0.26;


% Disp!!

sigma_of_mu_disp = 0.4*1;

point_of_tangent_disp = mean_of_mu_SA - 0.0;


% DM!!

sigma_of_mu_dm = 0.3*1;
sigma_dm = 0.7;



%%% Number of Monte-Carlo samples!!

N_real_xi = 1000;

% xi_samples = normrnd(0, 1, N_real_xi, 3);
xi_samples = randn(N_real_xi, 3);



% Number of independent random variables IF 'Full' is selected:
%[1 2 ... ...] is not changing since we are assuming independence for GM
% and other variables
% Mostly change bewteen [1 2 2 2] (full correlation), [1 2 2 4] (partial)
% and [1 2 3 4] (independent)


xi_order_linear = [1 2 3]; % [IM EDP DM DV] % independent

% If the epistemic uncertainty is dominated by DV, only 2 indep r.v. OK


%%% Numerical integration parameters!!

num_pts_SA = 200;

num_pts_disp = num_pts_SA;

num_pts_dm = num_pts_SA;

%% Domains of models!!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SA!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_vector_SA = linspace(0, 1, num_pts_SA);

SA_bound_left = exp(mean_of_mu_SA - 5*sigma_SA -5*sigma_of_mu_SA);
SA_bound_right = exp(mean_of_mu_SA + 3*sigma_SA + 3*sigma_of_mu_SA);

% SA from log space -7 to 2, 100 pts

SA_domain_vector = SA_bound_left + x_vector_SA.*(SA_bound_right - SA_bound_left); 
delta_SA = SA_domain_vector(2) - SA_domain_vector(1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EDP!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input parameters

mean_ky = 0.39; 

mean_Ts = 1.0;

[mean_of_mu_disp_vector, sigma_disp_vector] = Bray_Macedo_2018(mean_ky, mean_Ts, SA_domain_vector, Mag_scenario, mean_of_mu_SA, model_type);

sigma_of_mu_disp_vector = sigma_of_mu_disp * ones(1, num_pts_SA);

% Domain vector!!

disp_bound_left = min(exp(mean_of_mu_disp_vector -  2*sigma_disp_vector - 2*sigma_of_mu_disp_vector));
disp_bound_right = max(exp(mean_of_mu_disp_vector +  2*sigma_disp_vector + 2*sigma_of_mu_disp_vector));

disp_domain_vector = logspace(log10(disp_bound_left), log10(disp_bound_right), num_pts_disp); 
% disp_domain_vector = disp_bound_left + x_vector_SA.*(disp_bound_right - disp_bound_left); 

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

fs = 17;
lw = 2;

figure
plot(log(SA_domain_vector), mean_of_mu_disp_vector, 'Linewidth', lw)
hold on
plot(log(SA_domain_vector), tangent_disp_vector, 'Linewidth', lw)
% plot(mean_of_mu_SA, intercept_disp, 'or', 'Linewidth', lw)
plot(point_of_tangent_disp, intercept_disp, 'or', 'Linewidth', lw)
% plot(log(SA_domain_vector), p_coeffs(1)*log(SA_domain_vector) + p_coeffs(2), 'Linewidth', lw)

xlabel('ln SA')
ylabel('Median ln Disp')
legend('Bray & Macedo 2018', 'Tangent at m_{\mu}(SA)','Point of Tangent')%, 'Linear Regression')
set(gca, 'Fontsize', fs)  

% Computes where to take the tangent of disp for the DM model!!

[mean_of_mu_disp, ~] = Bray_Macedo_2018(mean_ky, mean_Ts, exp(mean_of_mu_SA), Mag_scenario, mean_of_mu_SA, model_type);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Domain Input!!

disp_bound_left = 5.6040e-10;
disp_bound_right =7.9706e+03;

disp_domain_vector = logspace(log10(disp_bound_left), log10(disp_bound_right), num_pts_disp); 


%% Input parameters sampling!!

N_samples_DM = 1000;

%%% Mean!!

D_pipe_mean = 610;
t_pipe_mean = 10.2;
sy_pipe_mean = 359000;
n_pipe_mean = 14;
r_pipe_mean = 8.5;

% Geotechnical parameters!!

gamma_t_soil_mean = 19;
H_soil_mean = 1.2;
L_soil_mean = 100;
phi_soil_mean = 38;
delta_soil_mean = 0.75;

%%% Sigma!!

D_pipe_sigma = 1* 0.25 * D_pipe_mean;
t_pipe_sigma =  1* 0.20 * t_pipe_mean;
sy_pipe_sigma = 1*  0.15 * sy_pipe_mean;
n_pipe_sigma =  1* 3;
r_pipe_sigma =  1* 1.5;

% Geotechnical parameters!!

gamma_t_soil_sigma = 1*  0.09 * gamma_t_soil_mean;
H_soil_sigma =  1* 0.15 * H_soil_mean;
L_soil_sigma =  1* 0.90 * L_soil_mean;
phi_soil_sigma =  1* 0.15 * phi_soil_mean;
delta_soil_sigma =  1* 0.14;

%%%%%%%

% D_pipe = normrnd(D_pipe_mean, D_pipe_sigma, 1, N_samples_DM);
% t_pipe= normrnd(t_pipe_mean, t_pipe_sigma, 1, N_samples_DM);
% sy_pipe= normrnd(sy_pipe_mean, sy_pipe_sigma, 1, N_samples_DM);
% n_pipe= normrnd(n_pipe_mean, n_pipe_sigma, 1, N_samples_DM);
% r_pipe= normrnd(r_pipe_mean, r_pipe_sigma, 1, N_samples_DM);

D_pipe = randn(1, N_samples_DM) * D_pipe_sigma + D_pipe_mean;
t_pipe= randn(1, N_samples_DM)* t_pipe_sigma+ t_pipe_mean;
sy_pipe= randn(1, N_samples_DM)* sy_pipe_sigma + sy_pipe_mean;
n_pipe= randn(1, N_samples_DM) * n_pipe_sigma + n_pipe_mean;
r_pipe= randn(1, N_samples_DM) * r_pipe_sigma + r_pipe_mean;

% Geotechnical parameters!!

mu_gamma_t_soil = log((gamma_t_soil_mean^2)/sqrt(gamma_t_soil_sigma+gamma_t_soil_mean^2));
sigma_gamma_t_soil = sqrt(log(gamma_t_soil_sigma/(gamma_t_soil_mean^2)+1));

mu_H_soil = log((H_soil_mean^2)/sqrt(H_soil_sigma+H_soil_mean^2));
sigma_H_soil = sqrt(log(H_soil_sigma/(H_soil_mean^2)+1));

mu_L_soil = log((L_soil_mean^2)/sqrt(L_soil_sigma+L_soil_mean^2));
sigma_L_soil = sqrt(log(L_soil_sigma/(L_soil_mean^2)+1));

mu_phi_soil = log((phi_soil_mean^2)/sqrt(phi_soil_sigma+phi_soil_mean^2));
sigma_phi_soil = sqrt(log(phi_soil_sigma/(phi_soil_mean^2)+1));

% gamma_t_soil= lognrnd(mu_gamma_t_soil, sigma_gamma_t_soil, 1, N_samples_DM);
% H_soil= lognrnd(mu_H_soil, sigma_H_soil, 1, N_samples_DM);
% L_soil= lognrnd(mu_L_soil, sigma_L_soil, 1, N_samples_DM);
% phi_soil= lognrnd(mu_phi_soil, sigma_phi_soil, 1, N_samples_DM);
% delta_soil= normrnd(delta_soil_mean, delta_soil_sigma, 1, N_samples_DM);
gamma_t_soil= exp(randn(1,1, N_samples_DM) * sigma_gamma_t_soil + mu_gamma_t_soil);
H_soil= exp(randn(1,1, N_samples_DM) * sigma_H_soil + mu_H_soil);
L_soil= exp(randn(1,1, N_samples_DM) * sigma_L_soil + mu_L_soil);
phi_soil= exp(randn(1,1, N_samples_DM) * sigma_phi_soil + mu_phi_soil);
delta_soil= exp(randn(1,1, N_samples_DM) * delta_soil_sigma + delta_soil_mean);

%%% Monte-Carlo For Loop!!
tu_pipe = zeros(1, N_samples_DM);
beta_p_pipe = zeros(1, N_samples_DM);

LE_pipe = zeros(N_samples_DM, num_pts_disp);
L_star_pipe = zeros(N_samples_DM, num_pts_disp);
eps_pipe = zeros(N_samples_DM, num_pts_disp);
mean_of_mu_dm_array = zeros(N_samples_DM, num_pts_disp);

for i = 1:N_samples_DM
% Intermediate parameters!!

tu_pipe(i) = gamma_t_soil(i) * (H_soil(i)+(D_pipe(i)/1000)/2)*((1+1)/2)*tan(deg2rad(phi_soil(i))*delta_soil(i))*pi*(D_pipe(i)/1000);
beta_p_pipe(i) = tu_pipe(i)/((0.25*pi*(D_pipe(i)/1000)^2)-(0.25*pi*((D_pipe(i)/1000)-((2*t_pipe(i))/1000))^2));
LE_pipe(i, :) = exp(0.613+0.853*log(t_pipe(i))-0.084*log(D_pipe(i))+0.751*log(sy_pipe(i))-0.735*log(H_soil(i))-0.863*log(gamma_t_soil(i))-1.005*log(phi_soil(i))-log(delta_soil(i))+0.136*log(disp_domain_vector/100));
% LE_pipe(i) = exp(0.613+0.853*log(t_pipe(i))-0.084*log(D_pipe(i))+0.751*log(sy_pipe(i))-0.735*log(H_soil(i))-0.863*log(gamma_t_soil(i))-1.005*log(phi_soil(i))-log(delta_soil(i))+0.136*log(0.5));
L_star_pipe(i, :) = min(L_soil(i)/2,LE_pipe(i, :));

% eps_pipe(i, :) = ((beta_p_pipe(i).*L_star_pipe(i, :))/200000000).*(1+(n_pipe(i)/(1+r_pipe(i))).*((beta_p_pipe(i).*L_star_pipe(i, :))/sy_pipe(i)).^r_pipe(i))*100;
eps_pipe(i, :) = (((beta_p_pipe(i).*L_star_pipe(i, :))/200000000).*(1+(n_pipe(i)/(1+r_pipe(i))).*((beta_p_pipe(i).*L_star_pipe(i, :))/sy_pipe(i)).^r_pipe(i))*100);

mean_of_mu_dm_array(i, :) = log(eps_pipe(i, :));

end



mean_of_mu_dm_vector = mean(mean_of_mu_dm_array, 1);


%% Plots!!


%%% Gets tangent approximation at mean Median Disp!!

index_mean_of_mu_disp = find(log(disp_domain_vector)>mean_of_mu_disp, 1);

intercept_dm = mean_of_mu_dm_vector(index_mean_of_mu_disp);

slope_dm = (mean_of_mu_dm_vector(index_mean_of_mu_disp+1) - mean_of_mu_dm_vector(index_mean_of_mu_disp))/...
           (log(disp_domain_vector(index_mean_of_mu_disp+1)) - log(disp_domain_vector(index_mean_of_mu_disp)));



tangent_dm_vector = slope_dm * (log(disp_domain_vector) - mean_of_mu_disp) + intercept_dm;

figure
plot(log(disp_domain_vector), mean_of_mu_dm_vector, '-b', 'Linewidth', lw)
hold on
plot(log(disp_domain_vector), tangent_dm_vector, '-r', 'Linewidth', lw)
plot(mean_of_mu_disp,  intercept_dm, 'or', 'Linewidth', lw)

set(gca, 'Fontsize', fs)
xlabel('ln(EDP)')
ylabel('Median DM')


%% DM vectors!!

mean_of_mu_dm_vector = mean(mean_of_mu_dm_array, 1);
sigma_of_mu_dm_vector = sigma_of_mu_dm * ones(1, num_pts_disp);
sigma_dm_vector = sigma_dm * ones(1, num_pts_disp);

dm_bound_left = min(exp(mean_of_mu_dm_vector -  2*sigma_dm_vector - 2*sigma_of_mu_dm_vector));
dm_bound_right = max(exp(mean_of_mu_dm_vector +  2*sigma_dm_vector + 2*sigma_of_mu_dm_vector));
dm_domain_vector = logspace(log10(dm_bound_left), log10(dm_bound_right), num_pts_disp);


%% Computes First Integral!!

tic 

% Linear!!

PDF_Z_Integrated_Y_MC_Linear_samples = zeros(num_pts_disp, N_real_xi);

for mc = 1:N_real_xi
    
    pdf_LN_SA_sample = 1./(sigma_SA * SA_domain_vector).*(normpdf((log(SA_domain_vector) - ...
                            (mean_of_mu_SA + sigma_of_mu_SA*xi_samples(mc, xi_order_linear(1))))/sigma_SA));
    
    pdf_LN_disp_sample = zeros(num_pts_disp, num_pts_SA);
    
    for sa = 1:num_pts_SA
        
        if strcmp(correlation_type, 'full') == 1
        
            pdf_LN_SA_disp_with_sa = 1./(sigma_disp_vector(1) * disp_domain_vector).*(normpdf((log(disp_domain_vector) - ...
                                (tangent_disp_vector(sa) + sigma_of_mu_disp_vector(sa)*xi_samples(mc, xi_order_linear(2))))/sigma_disp_vector(1)));
                            
        elseif strcmp(correlation_type, 'independent')
            
            pdf_LN_SA_disp_with_sa = 1./(sigma_disp_vector(1) * disp_domain_vector).*(normpdf((log(disp_domain_vector) - ...
                                (tangent_disp_vector(sa) + sigma_of_mu_disp_vector(sa)*xi_samples(mc, 2)))/sigma_disp_vector(1)));
                            
        end
        pdf_LN_disp_sample(:, sa) = pdf_LN_SA_disp_with_sa;
        
    end
    
    PDF_Z_Integrated_Y_MC_Linear_samples(:, mc) = sum(pdf_LN_disp_sample .* pdf_LN_SA_sample * delta_SA, 2);
    
end


%% Computes CDF Second Integral!!

delta_disp_vector = [diff(disp_domain_vector) (disp_domain_vector(end) - disp_domain_vector(end-1))];

% Linear!!

PDF_T_Integrated_Z_MC_Linear_samples = zeros(num_pts_dm, N_real_xi);

for mc = 1:N_real_xi
        
    pdf_LN_dm_sample = zeros(num_pts_dm, num_pts_disp);
    
    for disp_loop = 1:num_pts_disp
        
        if strcmp(correlation_type, 'full') == 1
            
            pdf_LN_disp_dm_with_disp = 1./(sigma_dm_vector(1) * dm_domain_vector).*(normpdf((log(dm_domain_vector) - ...
                                (tangent_dm_vector(disp_loop) + sigma_of_mu_dm_vector(disp_loop)*xi_samples(mc, xi_order_linear(3))))/sigma_dm_vector(1)));
                            
        elseif strcmp(correlation_type, 'independent') == 1
            
            pdf_LN_disp_dm_with_disp = 1./(sigma_dm_vector(1) * dm_domain_vector).*(normpdf((log(dm_domain_vector) - ...
                                (tangent_dm_vector(disp_loop) + sigma_of_mu_dm_vector(disp_loop)*xi_samples(mc, 3)))/sigma_dm_vector(1))); 
                            
        end
        pdf_LN_dm_sample(:, disp_loop) = pdf_LN_disp_dm_with_disp;
        
    end
    
    pdf_prod_loop = pdf_LN_dm_sample.*(PDF_Z_Integrated_Y_MC_Linear_samples(:, mc))';
    
    PDF_T_Integrated_Z_MC_Linear_samples(:, mc) = sum(pdf_prod_loop.*delta_disp_vector, 2);
    
end


toc





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computes PC Coefficients!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IM parameters

muY = mean_of_mu_SA;
sigmaMuY = sigma_of_mu_SA;
sigmaY = sigma_SA;


% EDP parameters

% amuZ = slope_disp;
% bmuZ = slope_disp.*(-mean_of_mu_SA) + intercept_disp;
% sigmaMuZ = sigma_of_mu_disp_vector(1);
% sigmaZ = sigma_disp_vector(1);
amuZ = 2.2945988;
bmuZ = -4.47664608;
sigmaMuZ = sigma_of_mu_disp_vector(1);
sigmaZ = sigma_disp_vector(1);

% DM parameters

% amuT = slope_dm;
% bmuT = slope_dm.*(-mean_of_mu_disp) + intercept_dm;
amuT = 0.02145702;
bmuT = -3.40633162;
sigmaMuT = sigma_of_mu_dm_vector(1);
sigmaT = sigma_dm_vector(1);


lnT = log(dm_domain_vector);

% PC Table

num_pc_terms_indep = nchoosek(3 + 4, 3);
index_PC_table_indep = index_table_function(3, 4);

Hermite_Proba_Table_indep = zeros(num_pc_terms_indep, N_real_xi);

for i = 1:num_pc_terms_indep
        
        Hermite_Proba_Table_indep(i, :) = (Hermite_Proba_new(xi_samples(:, 3)', index_PC_table_indep(i,1)) .* ...
                                           Hermite_Proba_new(xi_samples(:, 2)', index_PC_table_indep(i,2)).*...
                                           Hermite_Proba_new(xi_samples(:, 1)', index_PC_table_indep(i,3)));
end




%%% CDF!!

tic
% PC_Coeffs_CDF = PC_Coeffs_Simpler_CDF_Double_Integral_function(lnT, muY, sigmaMuY, sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT);
PC_Coeffs_CDF = PC_Coeffs_Simpler_CDF_Double_Integral_function(lnT, muY, sigmaMuY, sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT);
% PC_Coeffs_CDF_array = PC_Coeffs_Simpler_CDF_Double_Integral_array_function(lnT, muY, sigmaMuY, sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT);
toc


%% Computes PC samples!!

%%% CDF


% CDF_T_Integrated_samples = zeros(num_pts_dm, N_real_xi);
% 
% 
% 
% for i = 1:num_pc_terms_indep
%         
% 
% CDF_T_Integrated_samples = CDF_T_Integrated_samples + ...
%                            PC_Coeffs_CDF{i}.*...
%                            Hermite_Proba_Table_indep(i, :);
% 
% 
% end
% 
% 
% CDF_T_Integrated_samples = 1-CDF_T_Integrated_samples;

%%% PDF!!


delta_dm_vector = [diff(dm_domain_vector) (dm_domain_vector(end) - dm_domain_vector(end-1))];


PC_Coeffs_PDF = cell(1, size(PC_Coeffs_CDF, 2));


PDF_T_Integrated_samples = zeros(num_pts_dm, N_real_xi);



for i = 1:num_pc_terms_indep
        
    PC_Coeffs_PDF{i} = diff(PC_Coeffs_CDF{i});
    PC_Coeffs_PDF{i} = [PC_Coeffs_PDF{i} ; PC_Coeffs_PDF{i}(end)]./delta_dm_vector';

    PDF_T_Integrated_samples = PDF_T_Integrated_samples + ...
                               PC_Coeffs_PDF{i}.*...
                               Hermite_Proba_Table_indep(i, :);


end



%% CDF DV only!!

% CDF_V_Integrated_T_MC_Linear_samples = zeros(num_pts_dm, N_real_xi);

mean_dv = -mean(1.617 * log(D_pipe./t_pipe) - 1.709);
sigma_dv = 0.5;
sigma_mu_dv = 0.3;


a = -sigma_mu_dv^2/(2*sigma_dv^2) - 1/2;

b =  (log(dm_domain_vector) - mean_dv) * sigma_mu_dv/(sigma_dv^2);

c = - (log(dm_domain_vector) - mean_dv).^2/(2*sigma_dv^2);

alpha = sigma_mu_dv/(sigma_dv *2*sqrt(pi)) * exp(c - b.^2/(4*a));

             
            
PC_term_0 = 1/1 * (1 - normcdf(real((log(dm_domain_vector) - mean_dv)/sqrt(sigma_mu_dv^2 + sigma_dv^2))));


PC_term_1 = 1/1 * (1/sqrt(-a)) * alpha;


PC_term_2 =  1/2 * alpha .*b/(2*(-a)^(3/2));
                  
                  
PC_term_3 = 1/6 * alpha .*(-2*a*(1 + 2*a) + b.^2)/(4*(-a)^(5/2));


PC_term_4 = 1/24 * alpha .* (-b).*(6*a*(1 + 2*a) - b.^2)/(8*(-a)^(7/2));



CDF_DV_PC_samples = zeros(length(dm_domain_vector), N_real_xi);

% xi_4 = normrnd(0, 1, 1, N_real_xi);
xi_4 = randn(1, N_real_xi);

for i = 1:length(dm_domain_vector)

    CDF_DV_PC_samples(i, :) =   PC_term_0(i) + ...
                                PC_term_1(i) * Hermite_Proba_new(xi_4, 1) + ...
                                PC_term_2(i) * Hermite_Proba_new(xi_4, 2) + ...
                                PC_term_3(i) * Hermite_Proba_new(xi_4, 3) + ...
                                PC_term_4(i) * Hermite_Proba_new(xi_4, 4);


end

CDF_DV_PC_samples = 1-CDF_DV_PC_samples;

%% Performs third integration over dm values!!

CDF_DV_Integrated_DM_PC_samples = zeros(1, N_real_xi);

for i = 1:N_real_xi

CDF_DV_Integrated_DM_PC_samples(i)= sum(PDF_T_Integrated_samples(:, i).*...
                                        CDF_DV_PC_samples(:, i).*...
                                        delta_dm_vector', 1);

end

CDF_DV_Integrated_DM_PC_samples = min(CDF_DV_Integrated_DM_PC_samples,1)


%% Gets percentiles!!


pct_low = 10;
pct_middle = 50;
pct_high = 90;



% CDF MC Linear!!

mean_MC_CDF_Linear_vector = mean(PDF_T_Integrated_Z_MC_Linear_samples, 2);

percentile_lower_MC_CDF_Linear_vector  = zeros(num_pts_dm, 1);
percentile_middle_MC_CDF_Linear_vector  = zeros(num_pts_dm, 1);
percentile_higher_MC_CDF_Linear_vector  = zeros(num_pts_dm, 1);

for dm = 1:num_pts_dm
%     percentile_lower_MC_CDF_Linear_vector(dm, 1) = prctile(PDF_T_Integrated_Z_MC_Linear_samples(dm, :), pct_low);
%     percentile_middle_MC_CDF_Linear_vector(dm, 1) = prctile(PDF_T_Integrated_Z_MC_Linear_samples(dm, :), pct_middle);
%     percentile_higher_MC_CDF_Linear_vector(dm, 1) = prctile(PDF_T_Integrated_Z_MC_Linear_samples(dm, :), pct_high);
    percentile_lower_MC_CDF_Linear_vector(dm, 1) = interp1(PDF_T_Integrated_Z_MC_Linear_samples(dm, :), pct_low);
    percentile_middle_MC_CDF_Linear_vector(dm, 1) = interp1(PDF_T_Integrated_Z_MC_Linear_samples(dm, :), pct_middle);
    percentile_higher_MC_CDF_Linear_vector(dm, 1) = interp1(PDF_T_Integrated_Z_MC_Linear_samples(dm, :), pct_high);
end


% PDF PC IM EDP DM!!

mean_PC_PDF_Linear_vector = mean(PDF_T_Integrated_samples, 2);

percentile_lower_PC_PDF_Linear_vector  = zeros(num_pts_dm, 1);
percentile_middle_PC_PDF_Linear_vector  = zeros(num_pts_dm, 1);
percentile_higher_PC_PDF_Linear_vector  = zeros(num_pts_dm, 1);

for dm = 1:num_pts_dm
%     percentile_lower_PC_PDF_Linear_vector(dm, 1) = prctile(PDF_T_Integrated_samples(dm, :), pct_low);
%     percentile_middle_PC_PDF_Linear_vector(dm, 1) = prctile(PDF_T_Integrated_samples(dm, :), pct_middle);
%     percentile_higher_PC_PDF_Linear_vector(dm, 1) = prctile(PDF_T_Integrated_samples(dm, :), pct_high);
    percentile_lower_PC_PDF_Linear_vector(dm, 1) = interp1(PDF_T_Integrated_samples(dm, :), pct_low);
    percentile_middle_PC_PDF_Linear_vector(dm, 1) = interp1(PDF_T_Integrated_samples(dm, :), pct_middle);
    percentile_higher_PC_PDF_Linear_vector(dm, 1) = interp1(PDF_T_Integrated_samples(dm, :), pct_high);
end



%%% CDF PC DV only!!


mean_PC_CDF_DV_vector = mean(CDF_DV_PC_samples, 2);

percentile_lower_PC_CDF_DV_vector  = zeros(num_pts_dm, 1);
percentile_middle_PC_CDF_DV_vector = zeros(num_pts_dm, 1);
percentile_higher_PC_CDF_DV_vector = zeros(num_pts_dm, 1);

for dm = 1:num_pts_dm
%     percentile_lower_PC_CDF_DV_vector(dm, 1) = prctile(CDF_DV_PC_samples(dm, :), pct_low);
%     percentile_middle_PC_CDF_DV_vector(dm, 1) = prctile(CDF_DV_PC_samples(dm, :), pct_middle);
%     percentile_higher_PC_CDF_DV_vector(dm, 1) = prctile(CDF_DV_PC_samples(dm, :), pct_high);
    percentile_lower_PC_CDF_DV_vector(dm, 1) = interp1(CDF_DV_PC_samples(dm, :), pct_low);
    percentile_middle_PC_CDF_DV_vector(dm, 1) = interp1(CDF_DV_PC_samples(dm, :), pct_middle);
    percentile_higher_PC_CDF_DV_vector(dm, 1) = interp1(CDF_DV_PC_samples(dm, :), pct_high);
end




%% Plots MC PDF IM EDP DM!!


fs = 17;
lw = 2;


indices_plot = 1:num_pts_dm;

plot_type = 'linear';
% plot_type = 'log';


figure


if strcmp(plot_type, 'linear')


%%% MC Linear!!

plot(dm_domain_vector(indices_plot), percentile_lower_MC_CDF_Linear_vector(indices_plot), '+k', 'Linewidth', lw)
hold on
plot(dm_domain_vector(indices_plot), mean_MC_CDF_Linear_vector(indices_plot), '+k', 'Linewidth', lw)
plot(dm_domain_vector(indices_plot), percentile_middle_MC_CDF_Linear_vector(indices_plot), '+r', 'Linewidth', lw)
plot(dm_domain_vector(indices_plot), percentile_higher_MC_CDF_Linear_vector(indices_plot), '+k', 'Linewidth', lw)

%%% PC!!

plot(dm_domain_vector(indices_plot), percentile_lower_PC_PDF_Linear_vector(indices_plot), '-.k', 'Linewidth', lw)
hold on
plot(dm_domain_vector(indices_plot), mean_PC_PDF_Linear_vector(indices_plot), '--k', 'Linewidth', lw)
plot(dm_domain_vector(indices_plot), percentile_middle_PC_PDF_Linear_vector(indices_plot), '-.r', 'Linewidth', lw)
plot(dm_domain_vector(indices_plot), percentile_higher_PC_PDF_Linear_vector(indices_plot), '-.k', 'Linewidth', lw)


elseif strcmp(plot_type, 'log')


%%% MC Linear!!


loglog(dm_domain_vector(indices_plot), percentile_lower_MC_CDF_Linear_vector(indices_plot), '+k', 'Linewidth', lw)
hold on
loglog(dm_domain_vector(indices_plot), mean_MC_CDF_Linear_vector(indices_plot), '+k', 'Linewidth', lw)
loglog(dm_domain_vector(indices_plot), percentile_middle_MC_CDF_Linear_vector(indices_plot), '+r', 'Linewidth', lw)
loglog(dm_domain_vector(indices_plot), percentile_higher_MC_CDF_Linear_vector(indices_plot), '+k', 'Linewidth', lw)


%%% PC!!

loglog(dm_domain_vector(indices_plot), percentile_lower_PC_PDF_Linear_vector(indices_plot), '-.k', 'Linewidth', lw)
hold on
loglog(dm_domain_vector(indices_plot), mean_PC_PDF_Linear_vector(indices_plot), '--k', 'Linewidth', lw)
loglog(dm_domain_vector(indices_plot), percentile_middle_PC_PDF_Linear_vector(indices_plot), '-.r', 'Linewidth', lw)
loglog(dm_domain_vector(indices_plot), percentile_higher_PC_PDF_Linear_vector(indices_plot), '-.k', 'Linewidth', lw)


end


set(gca, 'Fontsize', fs)
xlabel('DV')
% ylabel('PDF : \int f(Z|Y)*f(Y)dY ')
legend('10th Percentile MC Linear',...
        'Mean MC Linear', ...
        '50th Percentile MC Linear', ...
        '90th Percentile MC Linear', ...
         '10th Percentile PC Linear',...
        'Mean PC Linear', ...
        '50th Percentile PC Linear', ...
        '90th Percentile PC Linear', ...
        'Location', 'Southwest')
    
    
title(strcat('Monte-Carlo Linear vs PC', ...
    '  \sigma_{\mu}(IM) = ', num2str(sigma_of_mu_SA), ...
    '  \sigma_{\mu}(EDP) = ', num2str(sigma_of_mu_disp), ...
    '  \sigma_{\mu}(DM) = ', num2str(sigma_of_mu_dm), ...
    '\newline', ...
    {'                                                            '}, ...
    '\xi_', num2str(xi_order_linear(1)), ...
    '                \xi_', num2str(xi_order_linear(2)), ...
'                  \xi_', num2str(xi_order_linear(3))),...
    'Interpreter', 'tex')

% ylim([1E-3 1])


%% Plots CDF DV only!!

figure

if strcmp(plot_type, 'linear')


%%% PC!!

plot(dm_domain_vector(indices_plot), percentile_lower_PC_CDF_DV_vector(indices_plot), '-.k', 'Linewidth', lw)
hold on
plot(dm_domain_vector(indices_plot), mean_PC_CDF_DV_vector(indices_plot), '--k', 'Linewidth', lw)
plot(dm_domain_vector(indices_plot), percentile_middle_PC_CDF_DV_vector(indices_plot), '-.r', 'Linewidth', lw)
plot(dm_domain_vector(indices_plot), percentile_higher_PC_CDF_DV_vector(indices_plot), '-.k', 'Linewidth', lw)


elseif strcmp(plot_type, 'log')


%%% PC!!

loglog(dm_domain_vector(indices_plot), percentile_lower_PC_CDF_DV_vector(indices_plot), '-.k', 'Linewidth', lw)
hold on
loglog(dm_domain_vector(indices_plot), mean_PC_CDF_DV_vector(indices_plot), '--k', 'Linewidth', lw)
loglog(dm_domain_vector(indices_plot), percentile_middle_PC_CDF_DV_vector(indices_plot), '-.r', 'Linewidth', lw)
loglog(dm_domain_vector(indices_plot), percentile_higher_PC_CDF_DV_vector(indices_plot), '-.k', 'Linewidth', lw)


end


set(gca, 'Fontsize', fs)
xlabel('DV')
% ylabel('PDF : \int f(Z|Y)*f(Y)dY ')
legend('10th Percentile PC Linear',...
        'Mean PC Linear', ...
        '50th Percentile PC Linear', ...
        '90th Percentile PC Linear', ...
        'Location', 'Southwest')

title('CDF DV only with Epistemic Uncertainty')
%% Histogram of Final Results!! 
figure
% histogram(CDF_DV_Integrated_DM_PC_samples)
histogram(real(CDF_DV_Integrated_DM_PC_samples))
set(gca, 'Fontsize', fs)
xlabel('Probability of Rupture')
ylim([0,250]);
xlim([-0.01,1.01]);