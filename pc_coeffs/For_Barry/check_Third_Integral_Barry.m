%% Input parameters!!

model_type = 'quadratic'; % Bray and Macedo model approximation


Mag_scenario = 7;


% SA!!

mean_of_mu_SA = -2;

sigma_of_mu_SA = 0.3 * 1;

sigma_SA = 0.6;



% Disp!!

sigma_of_mu_disp = 0.3*1;

point_of_tangent_disp = mean_of_mu_SA - 0.0;


% DM!!

sigma_of_mu_dm = 0.3*1;

sigma_dm = 0.1;


% DV!!


muV = -3 * 1;

sigmaMuV = 0.3 * 1;

sigmaV = 0.55;



%%% PC Parameters

PC_order = 4;

KL_dim = 4;

num_pc_terms_total = nchoosek(PC_order + KL_dim, KL_dim);


%%% Number of Monte-Carlo samples!!

N_real_xi = 1000;

xi_samples = normrnd(0, 1, N_real_xi, KL_dim);



%%% Numerical integration parameters!!

num_pts_SA = 1E4;

num_pts_disp = 1E4;

num_pts_dm = 1E4;


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

[mean_of_mu_disp_vector, sigma_disp_vector] = Bray_Macedo_2018(mean_ky, mean_Ts, SA_domain_vector, Mag_scenario, mean_of_mu_SA, model_type);

sigma_of_mu_disp_vector = sigma_of_mu_disp * ones(1, num_pts_SA);

% Domain vector!!

disp_bound_left = min(exp(mean_of_mu_disp_vector -  5*sigma_disp_vector - 5*sigma_of_mu_disp_vector));
disp_bound_right = max(exp(mean_of_mu_disp_vector +  4*sigma_disp_vector + 3*sigma_of_mu_disp_vector));

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
% 
% Computes where to take the tangent of disp for the DM model!!

[mean_of_mu_disp, ~] = Bray_Macedo_2018(mean_ky, mean_Ts, exp(mean_of_mu_SA), Mag_scenario, mean_of_mu_SA, model_type);




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

points_to_fit_dm = [-6 -6 ; 0 -3 ; 6 -1]; % 1st col is ln(disp), 2nd is ln(strain)
p_dm = polyfit(points_to_fit_dm(:, 1), points_to_fit_dm(:, 2), 2);


% mean_of_mu_dm_vector = DM_slope * log(disp_domain_vector) + DM_intercept;
mean_of_mu_dm_vector = polyval(p_dm, log(disp_domain_vector));

sigma_dm_vector = sigma_dm * ones(1, num_pts_disp);

sigma_of_mu_dm_vector = sigma_of_mu_dm * ones(1, num_pts_disp);

% Tangent!!

slope_dm = 2*p_dm(1)*mean_of_mu_disp + p_dm(2);
intercept_dm = polyval(p_dm, mean_of_mu_disp);

dm_bound_left = min(exp(mean_of_mu_dm_vector -  5*sigma_dm_vector - 5*sigma_of_mu_dm_vector));
dm_bound_right = max(exp(mean_of_mu_dm_vector +  3*sigma_dm_vector + 3*sigma_of_mu_dm_vector));

dm_domain_vector = logspace(log10(dm_bound_left), log10(dm_bound_right), num_pts_dm);
tangent_dm_vector = slope_dm * (log(disp_domain_vector)-mean_of_mu_disp) + intercept_dm;

delta_dm_vector = [diff(dm_domain_vector) (dm_domain_vector(end) - dm_domain_vector(end-1))];


% figure
% plot(log(disp_domain_vector), mean_of_mu_dm_vector, 'Linewidth', lw)
% hold on
% plot(log(disp_domain_vector), tangent_dm_vector, 'Linewidth', lw)
% plot(mean_of_mu_disp,  intercept_dm, 'or', 'Linewidth', lw)
% 
% xlabel('ln Disp')
% ylabel('Median ln DM')
% legend('Built Model', 'Tangent at m_{\mu}(Disp)','Point of Tangent')
% set(gca, 'Fontsize', fs)


% Computes where to take the tangent of dm for the DV model!!

mean_of_mu_dm = polyval(p_dm, mean_of_mu_disp);

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


% DM parameters

amuT = slope_dm;
bmuT = slope_dm.*(-mean_of_mu_disp) + intercept_dm;
sigmaMuT = sigma_of_mu_dm_vector(1);
sigmaT = sigma_dm_vector(1);


% DV parameters



% PC Table

index_PC_table_KLdim_4 = index_table_function(KL_dim, PC_order);

Hermite_PC_Table = zeros(num_pc_terms_total, N_real_xi);

% Y: xi1
% Z: xi2
% T: xi3
% V: xi4

for i = 1:num_pc_terms_total
        
        Hermite_PC_Table(i, :) = Hermite_Proba_new(xi_samples(:, 4)', index_PC_table_KLdim_4(i,1)).*...
                                 Hermite_Proba_new(xi_samples(:, 3)', index_PC_table_KLdim_4(i,2)).*...
                                 Hermite_Proba_new(xi_samples(:, 2)', index_PC_table_KLdim_4(i,3)).*...
                                 Hermite_Proba_new(xi_samples(:, 1)', index_PC_table_KLdim_4(i,4));

end




%% Computes PC Coefficients of Second Integration Results!!




index_PC_table_KLdim_3to4(1, 1 : 2) = [1, 1];
index_PC_table_KLdim_3to4(2, 1 : 2) = [2, 1];
index_PC_table_KLdim_3to4(3, 1 : 2) = [1, 2];
index_PC_table_KLdim_3to4(4, 1 : 2) = [1, 3];
index_PC_table_KLdim_3to4(5, 1 : 2) = [1, 4];
index_PC_table_KLdim_3to4(6, 1 : 2) = [3, 1];
index_PC_table_KLdim_3to4(7, 1 : 2) = [2, 2];
index_PC_table_KLdim_3to4(8, 1 : 2) = [2, 3];
index_PC_table_KLdim_3to4(9, 1 : 2) = [2, 4];
index_PC_table_KLdim_3to4(10, 1 : 2) = [1, 5];
index_PC_table_KLdim_3to4(11, 1 : 2) = [1, 6];
index_PC_table_KLdim_3to4(12, 1 : 2) = [1, 7];
index_PC_table_KLdim_3to4(13, 1 : 2) = [1, 8];
index_PC_table_KLdim_3to4(14, 1 : 2) = [1, 9];
index_PC_table_KLdim_3to4(15, 1 : 2) = [1, 10];
index_PC_table_KLdim_3to4(16, 1 : 2) = [4, 1];
index_PC_table_KLdim_3to4(17, 1 : 2) = [3, 2];
index_PC_table_KLdim_3to4(18, 1 : 2) = [3, 3];
index_PC_table_KLdim_3to4(19, 1 : 2) = [3, 4];
index_PC_table_KLdim_3to4(20, 1 : 2) = [2, 5];
index_PC_table_KLdim_3to4(21, 1 : 2) = [2, 6];
index_PC_table_KLdim_3to4(22, 1 : 2) = [2, 7];
index_PC_table_KLdim_3to4(23, 1 : 2) = [2, 8];
index_PC_table_KLdim_3to4(24, 1 : 2) = [2, 9];
index_PC_table_KLdim_3to4(25, 1 : 2) = [2, 10];
index_PC_table_KLdim_3to4(26, 1 : 2) = [1, 11];
index_PC_table_KLdim_3to4(27, 1 : 2) = [1, 12];
index_PC_table_KLdim_3to4(28, 1 : 2) = [1, 13];
index_PC_table_KLdim_3to4(29, 1 : 2) = [1, 14];
index_PC_table_KLdim_3to4(30, 1 : 2) = [1, 15];
index_PC_table_KLdim_3to4(31, 1 : 2) = [1, 16];
index_PC_table_KLdim_3to4(32, 1 : 2) = [1, 17];
index_PC_table_KLdim_3to4(33, 1 : 2) = [1, 18];
index_PC_table_KLdim_3to4(34, 1 : 2) = [1, 19];
index_PC_table_KLdim_3to4(35, 1 : 2) = [1, 20];


index_PC_table_KLdim_3to4(36, 1 : 2) = [5, 1];
index_PC_table_KLdim_3to4(37, 1 : 2) = [4, 2];
index_PC_table_KLdim_3to4(38, 1 : 2) = [4, 3];
index_PC_table_KLdim_3to4(39, 1 : 2) = [4, 4];
index_PC_table_KLdim_3to4(40, 1 : 2) = [3, 5];
index_PC_table_KLdim_3to4(41, 1 : 2) = [3, 6];
index_PC_table_KLdim_3to4(42, 1 : 2) = [3, 7];
index_PC_table_KLdim_3to4(43, 1 : 2) = [3, 8];
index_PC_table_KLdim_3to4(44, 1 : 2) = [3, 9];
index_PC_table_KLdim_3to4(45, 1 : 2) = [3, 10];
index_PC_table_KLdim_3to4(46, 1 : 2) = [2, 11];
index_PC_table_KLdim_3to4(47, 1 : 2) = [2, 12];
index_PC_table_KLdim_3to4(48, 1 : 2) = [2, 13];
index_PC_table_KLdim_3to4(49, 1 : 2) = [2, 14];
index_PC_table_KLdim_3to4(50, 1 : 2) = [2, 15];
index_PC_table_KLdim_3to4(51, 1 : 2) = [2, 16];
index_PC_table_KLdim_3to4(52, 1 : 2) = [2, 17];
index_PC_table_KLdim_3to4(53, 1 : 2) = [2, 18];
index_PC_table_KLdim_3to4(54, 1 : 2) = [2, 19];
index_PC_table_KLdim_3to4(55, 1 : 2) = [2, 20];
index_PC_table_KLdim_3to4(56, 1 : 2) = [1, 21];
index_PC_table_KLdim_3to4(57, 1 : 2) = [1, 22];
index_PC_table_KLdim_3to4(58, 1 : 2) = [1, 23];
index_PC_table_KLdim_3to4(59, 1 : 2) = [1, 24];
index_PC_table_KLdim_3to4(60, 1 : 2) = [1, 25];
index_PC_table_KLdim_3to4(61, 1 : 2) = [1, 26];
index_PC_table_KLdim_3to4(62, 1 : 2) = [1, 27];
index_PC_table_KLdim_3to4(63, 1 : 2) = [1, 28];
index_PC_table_KLdim_3to4(64, 1 : 2) = [1, 29];
index_PC_table_KLdim_3to4(65, 1 : 2) = [1, 30];
index_PC_table_KLdim_3to4(66, 1 : 2) = [1, 31];
index_PC_table_KLdim_3to4(67, 1 : 2) = [1, 32];
index_PC_table_KLdim_3to4(68, 1 : 2) = [1, 33];
index_PC_table_KLdim_3to4(69, 1 : 2) = [1, 34];
index_PC_table_KLdim_3to4(70, 1 : 2) = [1, 35];




P_output_2 = PC_Coeffs_Simpler_Second_Integration_function(muY, sigmaMuY, ...
   sigmaY, amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT);

a2=(-1/2).*(sigmaMuT.^2+sigmaT.^2+amuT.^2.*(sigmaMuZ.^2+amuZ.^2.*( ...
  sigmaMuY.^2+sigmaY.^2)+sigmaZ.^2)).^(-1);
b2=(bmuT+amuT.*(bmuZ+amuZ.*muY)).*(sigmaMuT.^2+sigmaT.^2+amuT.^2.*( ...
  sigmaMuZ.^2+amuZ.^2.*(sigmaMuY.^2+sigmaY.^2)+sigmaZ.^2)).^(-1);
c2=(-1/2).*(bmuT+amuT.*(bmuZ+amuZ.*muY)).^2.*(sigmaMuT.^2+sigmaT.^2+ ...
  amuT.^2.*(sigmaMuZ.^2+amuZ.^2.*(sigmaMuY.^2+sigmaY.^2)+sigmaZ.^2)).^(-1) ...
  ;


alphaT = 1./dm_domain_vector .* exp(a2*log(dm_domain_vector).^2 + b2 * log(dm_domain_vector) + c2);


index_PC_table_KLdim_3 = index_table_function(KL_dim-1, PC_order);
num_pc_terms_KLdim_3 = size(index_PC_table_KLdim_3, 1);




PC_Coeffs_PDF_TZ = cell(num_pc_terms_KLdim_3, 1);

p1c0=P_output_2(1, :);
p2c0=P_output_2(2, :);
p2c1=P_output_2(3, :);
p3c0=P_output_2(4, :);
p3c1=P_output_2(5, :);
p4c0=P_output_2(6, :);
p4c1=P_output_2(7, :);
p5c0=P_output_2(8, :);
p5c1=P_output_2(9, :);
p5c2=P_output_2(10, :);
p6c0=P_output_2(11, :);
p6c1=P_output_2(12, :);
p6c2=P_output_2(13, :);
p7c0=P_output_2(14, :);
p7c1=P_output_2(15, :);
p7c2=P_output_2(16, :);
p8c0=P_output_2(17, :);
p8c1=P_output_2(18, :);
p8c2=P_output_2(19, :);
p9c0=P_output_2(20, :);
p9c1=P_output_2(21, :);
p9c2=P_output_2(22, :);
p10c0=P_output_2(23, :);
p10c1=P_output_2(24, :);
p10c2=P_output_2(25, :);
p11c0=P_output_2(26, :);
p11c1=P_output_2(27, :);
p11c2=P_output_2(28, :);
p11c3=P_output_2(29, :);
p12c0=P_output_2(30, :);
p12c1=P_output_2(31, :);
p12c2=P_output_2(32, :);
p12c3=P_output_2(33, :);
p13c0=P_output_2(34, :);
p13c1=P_output_2(35, :);
p13c2=P_output_2(36, :);
p13c3=P_output_2(37, :);
p14c0=P_output_2(38, :);
p14c1=P_output_2(39, :);
p14c2=P_output_2(40, :);
p14c3=P_output_2(41, :);
p15c0=P_output_2(42, :);
p15c1=P_output_2(43, :);
p15c2=P_output_2(44, :);
p15c3=P_output_2(45, :);
p16c0=P_output_2(46, :);
p16c1=P_output_2(47, :);
p16c2=P_output_2(48, :);
p16c3=P_output_2(49, :);
p17c0=P_output_2(50, :);
p17c1=P_output_2(51, :);
p17c2=P_output_2(52, :);
p17c3=P_output_2(53, :);
p18c0=P_output_2(54, :);
p18c1=P_output_2(55, :);
p18c2=P_output_2(56, :);
p18c3=P_output_2(57, :);
p19c0=P_output_2(58, :);
p19c1=P_output_2(59, :);
p19c2=P_output_2(60, :);
p19c3=P_output_2(61, :);
p20c0=P_output_2(62, :);
p20c1=P_output_2(63, :);
p20c2=P_output_2(64, :);
p20c3=P_output_2(65, :);
p21c0=P_output_2(66, :);
p21c1=P_output_2(67, :);
p21c2=P_output_2(68, :);
p21c3=P_output_2(69, :);
p21c4=P_output_2(70, :);
p22c0=P_output_2(71, :);
p22c1=P_output_2(72, :);
p22c2=P_output_2(73, :);
p22c3=P_output_2(74, :);
p22c4=P_output_2(75, :);
p23c0=P_output_2(76, :);
p23c1=P_output_2(77, :);
p23c2=P_output_2(78, :);
p23c3=P_output_2(79, :);
p23c4=P_output_2(80, :);
p24c0=P_output_2(81, :);
p24c1=P_output_2(82, :);
p24c2=P_output_2(83, :);
p24c3=P_output_2(84, :);
p24c4=P_output_2(85, :);
p25c0=P_output_2(86, :);
p25c1=P_output_2(87, :);
p25c2=P_output_2(88, :);
p25c3=P_output_2(89, :);
p25c4=P_output_2(90, :);
p26c0=P_output_2(91, :);
p26c1=P_output_2(92, :);
p26c2=P_output_2(93, :);
p26c3=P_output_2(94, :);
p26c4=P_output_2(95, :);
p27c0=P_output_2(96, :);
p27c1=P_output_2(97, :);
p27c2=P_output_2(98, :);
p27c3=P_output_2(99, :);
p27c4=P_output_2(100, :);
p28c0=P_output_2(101, :);
p28c1=P_output_2(102, :);
p28c2=P_output_2(103, :);
p28c3=P_output_2(104, :);
p28c4=P_output_2(105, :);
p29c0=P_output_2(106, :);
p29c1=P_output_2(107, :);
p29c2=P_output_2(108, :);
p29c3=P_output_2(109, :);
p29c4=P_output_2(110, :);
p30c0=P_output_2(111, :);
p30c1=P_output_2(112, :);
p30c2=P_output_2(113, :);
p30c3=P_output_2(114, :);
p30c4=P_output_2(115, :);
p31c0=P_output_2(116, :);
p31c1=P_output_2(117, :);
p31c2=P_output_2(118, :);
p31c3=P_output_2(119, :);
p31c4=P_output_2(120, :);
p32c0=P_output_2(121, :);
p32c1=P_output_2(122, :);
p32c2=P_output_2(123, :);
p32c3=P_output_2(124, :);
p32c4=P_output_2(125, :);
p33c0=P_output_2(126, :);
p33c1=P_output_2(127, :);
p33c2=P_output_2(128, :);
p33c3=P_output_2(129, :);
p33c4=P_output_2(130, :);
p34c0=P_output_2(131, :);
p34c1=P_output_2(132, :);
p34c2=P_output_2(133, :);
p34c3=P_output_2(134, :);
p34c4=P_output_2(135, :);
p35c0=P_output_2(136, :);
p35c1=P_output_2(137, :);
p35c2=P_output_2(138, :);
p35c3=P_output_2(139, :);
p35c4=P_output_2(140, :);

%%% Remultiplies by alphaZ after integration!!

PC_Coeffs_PDF_TZ{1} = polyval(p1c0, log(dm_domain_vector)).*alphaT;

PC_Coeffs_PDF_TZ{2} = polyval([p2c1 p2c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{3} = polyval([p3c1 p3c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{4} = polyval([p4c1 p4c0], log(dm_domain_vector)).*alphaT;

PC_Coeffs_PDF_TZ{5} = polyval([p5c2 p5c1 p5c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{6} = polyval([p6c2 p6c1 p6c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{7} = polyval([p7c2 p7c1 p7c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{8} = polyval([p8c2 p8c1 p8c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{9} = polyval([p9c2 p9c1 p9c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{10} = polyval([p10c2 p10c1 p10c0], log(dm_domain_vector)).*alphaT;

PC_Coeffs_PDF_TZ{11} = polyval([p11c3 p11c2 p11c1 p11c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{12} = polyval([p12c3 p12c2 p12c1 p12c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{13} = polyval([p13c3 p13c2 p13c1 p13c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{14} = polyval([p14c3 p14c2 p14c1 p14c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{15} = polyval([p15c3 p15c2 p15c1 p15c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{16} = polyval([p16c3 p16c2 p16c1 p16c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{17} = polyval([p17c3 p17c2 p17c1 p17c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{18} = polyval([p18c3 p18c2 p18c1 p18c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{19} = polyval([p19c3 p19c2 p19c1 p19c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{20} = polyval([p20c3 p20c2 p20c1 p20c0], log(dm_domain_vector)).*alphaT;


PC_Coeffs_PDF_TZ{21} = polyval([p21c4 p21c3 p21c2 p21c1 p21c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{22} = polyval([p22c4 p22c3 p22c2 p22c1 p22c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{23} = polyval([p23c4 p23c3 p23c2 p23c1 p23c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{24} = polyval([p24c4 p24c3 p24c2 p24c1 p24c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{25} = polyval([p25c4 p25c3 p25c2 p25c1 p25c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{26} = polyval([p26c4 p26c3 p26c2 p26c1 p26c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{27} = polyval([p27c4 p27c3 p27c2 p27c1 p27c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{28} = polyval([p28c4 p28c3 p28c2 p28c1 p28c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{29} = polyval([p29c4 p29c3 p29c2 p29c1 p29c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{30} = polyval([p30c4 p30c3 p30c2 p30c1 p30c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{31} = polyval([p31c4 p31c3 p31c2 p31c1 p31c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{32} = polyval([p32c4 p32c3 p32c2 p32c1 p32c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{33} = polyval([p33c4 p33c3 p33c2 p33c1 p33c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{34} = polyval([p34c4 p34c3 p34c2 p34c1 p34c0], log(dm_domain_vector)).*alphaT;
PC_Coeffs_PDF_TZ{35} = polyval([p35c4 p35c3 p35c2 p35c1 p35c0], log(dm_domain_vector)).*alphaT;



%% PDF COEFFS PDF IM!!

PC_Coeffs_PDF_Y = cell(1, 5);


aofY = -sigmaMuY^2/(2*sigmaY^2) - 1/2;

bofY =  (log(SA_domain_vector) - muY) * sigmaMuY/(sigmaY^2);

cofY = - (log(SA_domain_vector) - muY).^2/(2*sigmaY^2);

alphaY = 1./(sigmaY*SA_domain_vector) * 1/(1 *2*sqrt(pi)) .* exp(cofY - bofY.^2/(4*aofY));

            
PC_Coeffs_PDF_Y{1} = 1/1 * (1/sqrt(-aofY)) * alphaY;

PC_Coeffs_PDF_Y{2} =  1/1 * alphaY .*bofY/(2*(-aofY)^(3/2));
                  
PC_Coeffs_PDF_Y{3} = 1/2 * alphaY .*(-2*aofY*(1 + 2*aofY) + bofY.^2)/(4*(-aofY)^(5/2));

PC_Coeffs_PDF_Y{4} = 1/6 * alphaY .* (-bofY).*(6*aofY*(1 + 2*aofY) - bofY.^2)/(8*(-aofY)^(7/2));

PC_Coeffs_PDF_Y{5} = 1/24 * alphaY .* (12*aofY^2*(1+ 2*aofY)^2 - 12*aofY*(1 + 2*aofY)*bofY.^2 + bofY.^4)/(16*(-aofY)^(9/2));


%% PC COEFFS PDF EDP!!


aofZ = -sigmaMuZ^2/(2*sigmaZ^2) - 1/2;

bofZ =  (log(disp_domain_vector') - (amuZ*log(SA_domain_vector) + bmuZ)) * sigmaMuZ/(sigmaZ^2);

cofZ = - (log(disp_domain_vector') - (amuZ*log(SA_domain_vector) + bmuZ)).^2/(2*sigmaZ^2);

alphaZ = 1./(sigmaZ*disp_domain_vector') * 1/(1 *2*sqrt(pi)) .* exp(cofZ - bofZ.^2/(4*aofZ));



PC_Coeffs_PDF_Z = cell(5, 1);

PC_Coeffs_PDF_Z{1} = 1/1 * (1/sqrt(-aofZ)) * alphaZ;

PC_Coeffs_PDF_Z{2} =  1/1 * alphaZ .*bofZ/(2*(-aofZ)^(3/2));
                  
PC_Coeffs_PDF_Z{3} = 1/2 * alphaZ .*(-2*aofZ*(1 + 2*aofZ) + bofZ.^2)/(4*(-aofZ)^(5/2));

PC_Coeffs_PDF_Z{4} = 1/6 * alphaZ .* (-bofZ).*(6*aofZ*(1 + 2*aofZ) - bofZ.^2)/(8*(-aofZ)^(7/2));

PC_Coeffs_PDF_Z{5} = 1/24 * alphaZ .* (12*aofZ^2*(1+ 2*aofZ)^2 - 12*aofZ*(1 + 2*aofZ)*bofZ.^2 + bofZ.^4)/(16*(-aofZ)^(9/2));


%% PC COEFFS PDF DM!!


aofT = -sigmaMuT^2/(2*sigmaT^2) - 1/2;

bofT =  (log(dm_domain_vector') - (amuT*log(disp_domain_vector) + bmuT)) * sigmaMuT/(sigmaT^2);

cofT = - (log(dm_domain_vector') - (amuT*log(disp_domain_vector) + bmuT)).^2/(2*sigmaT^2);

alphaT = 1./(sigmaT*dm_domain_vector') * 1/(1 *2*sqrt(pi)) .* exp(cofT - bofT.^2/(4*aofT));



PC_Coeffs_PDF_T = cell(5, 1);

PC_Coeffs_PDF_T{1} = 1/1 * (1/sqrt(-aofT)) * alphaT;

PC_Coeffs_PDF_T{2} =  1/1 * alphaT .*bofT/(2*(-aofT)^(3/2));
                  
PC_Coeffs_PDF_T{3} = 1/2 * alphaT .*(-2*aofT*(1 + 2*aofT) + bofT.^2)/(4*(-aofT)^(5/2));

PC_Coeffs_PDF_T{4} = 1/6 * alphaT .* (-bofT).*(6*aofT*(1 + 2*aofT) - bofT.^2)/(8*(-aofT)^(7/2));

PC_Coeffs_PDF_T{5} = 1/24 * alphaT .* (12*aofT^2*(1+ 2*aofT)^2 - 12*aofT*(1 + 2*aofT)*bofT.^2 + bofT.^4)/(16*(-aofT)^(9/2));


%% PC COEFFS CDF DV!!



a = -sigmaMuV^2/(2*sigmaV^2) - 1/2;

b =  (log(dm_domain_vector) - muV) * sigmaMuV/(sigmaV^2);

c = - (log(dm_domain_vector) - muV).^2/(2*sigmaV^2);

alpha = sigmaMuV/(sigmaV *2*sqrt(pi)) * exp(c - b.^2/(4*a));

             

PC_Coeffs_CDF_V = cell(1, PC_order + 1);


PC_Coeffs_CDF_V{1} = 1/1 * normcdf((log(dm_domain_vector) - muV)/sqrt(sigmaMuV^2 + sigmaV^2));


PC_Coeffs_CDF_V{2} = 1/1 * (1/sqrt(-a)) * alpha;


PC_Coeffs_CDF_V{3} =  1/2 * alpha .*b/(2*(-a)^(3/2));
                  
                  
PC_Coeffs_CDF_V{4} = 1/6 * alpha .*(-2*a*(1 + 2*a) + b.^2)/(4*(-a)^(5/2));


PC_Coeffs_CDF_V{5} = 1/24 * alpha .* (-b).*(6*a*(1 + 2*a) - b.^2)/(8*(-a)^(7/2));




%% Check Second Integration!!

% index_check = 2;
% 
% 
% 
% First_Integral_check = sum(PC_Coeffs_PDF_Z{index_PC_table_KLdim_4(index_check, 3)+1}.*...
%                            PC_Coeffs_PDF_Y{index_PC_table_KLdim_4(index_check, 4)+1}.*...
%                            delta_SA_vector, 2);
% 
% 
% Second_Integral_check = sum(PC_Coeffs_PDF_T{index_PC_table_KLdim_4(index_check, 2)+1}.*...
%                            First_Integral_check'.*...
%                            delta_disp_vector, 2);
% 
% 
% % plot(log(dm_domain_vector), Second_Integral_check)
% % hold on
% % plot(log(dm_domain_vector), PC_Coeffs_PDF_TZ{index_PC_table_KLdim_3to4(index_check, 2)})
% 
% %%% Second Integral: OK!!
% 
% %%% Third Integral: OK!!
% 
% Third_Integral_check = sum(PC_Coeffs_CDF_V{index_PC_table_KLdim_4(index_check, 1)+1}.*...
%                            Second_Integral_check'.*...
%                            delta_dm_vector, 2);
% 
% Third_Integral_check;
% 
% sum(PC_Coeffs_CDF_V{index_PC_table_KLdim_3to4(index_check, 1)}.*...
%     PC_Coeffs_PDF_TZ{index_PC_table_KLdim_3to4(index_check, 2)}.*...
%     delta_dm_vector);
% 

                 

%% Performs Third Integration over PC terms!!


PC_Coeffs_CDF_PDF_Numerical_Integration = zeros(num_pc_terms_total, 1);

for i = 1:num_pc_terms_total

    PC_Coeffs_CDF_PDF_Numerical_Integration(i) = sum(PC_Coeffs_CDF_V{index_PC_table_KLdim_3to4(i, 1)}.*...
                                                     PC_Coeffs_PDF_TZ{index_PC_table_KLdim_3to4(i, 2)}.*...
                                                     delta_dm_vector);


end



%% ANALYTICAL PC SOLUTION!!

PC_Coeffs_CDF_PDF_Analytical_Integration = PC_Coeffs_Third_Integration_New_Models_function(muY, sigmaMuY, sigmaY,  ...
  amuZ, bmuZ, sigmaMuZ, sigmaZ, amuT, bmuT, sigmaMuT, sigmaT, muV, sigmaMuV, sigmaV); %;

PC_Coeffs_CDF_PDF_Analytical_Integration = cell2mat(PC_Coeffs_CDF_PDF_Analytical_Integration);




%% Samples from PC Numerical Integration!!


CDF_Samples_From_PC_Numerical_Integration = zeros(1, N_real_xi);

for i = 1:num_pc_terms_total

    CDF_Samples_From_PC_Numerical_Integration = CDF_Samples_From_PC_Numerical_Integration + ...
                                                PC_Coeffs_CDF_PDF_Numerical_Integration(i) * Hermite_PC_Table(i, :);
end



%% Samples from PC Analytical Integration!!


CDF_Samples_From_PC_Analytical_Integration = zeros(1, N_real_xi);

for i = 1:num_pc_terms_total

    CDF_Samples_From_PC_Analytical_Integration = CDF_Samples_From_PC_Analytical_Integration + ...
                                                 PC_Coeffs_CDF_PDF_Analytical_Integration(i) * Hermite_PC_Table(i, :);
end





%% Checks Cross_Product and Integral!!

% index_check_kldim4 = 2;
% 
% icheck = index_PC_table_KLdim_3to4(index_check_kldim4, 1);
% jcheck = index_PC_table_KLdim_3to4(index_check_kldim4, 2);
% 
% 
% 
% 
% Cross_term_check = PC_Coeffs_CDF_V{icheck} .* PC_Coeffs_PDF_TZ{jcheck};
% 
% Cross_term_Integrated = sum(Cross_term_check.*delta_dm_vector, 2);
% 
% 
% PC_Coeffs_CDF_PDF_Analytical_Integration(index_check_kldim4);
% 
% % plot(log(dm_domain_vector),PC_Coeffs_CDF_V{icheck})
% % plot(log(dm_domain_vector),PC_Coeffs_PDF_TZ{icheck})
% 
% % Cross_term_Integrated)
% % PC_Coeffs_CDF_PDF_Numerical_Integration(index_check_kldim4))
%  


%% Histogram of Final Results!! 



pct_low = 10;
pct_middle = 50;
pct_high = 90;



fs = 17;
lw = 2;



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


disp('Percentiles of Samples from Numerical Integration')

[prctile(CDF_Samples_From_PC_Numerical_Integration, pct_low),...
 prctile(CDF_Samples_From_PC_Numerical_Integration, pct_middle),...
 mean(CDF_Samples_From_PC_Numerical_Integration),...
 prctile(CDF_Samples_From_PC_Numerical_Integration, pct_high)]

disp('Percentiles of Samples from direct PC Analytical Integration')

[prctile(CDF_Samples_From_PC_Analytical_Integration, pct_low),...
 prctile(CDF_Samples_From_PC_Analytical_Integration, pct_middle),...
 mean(CDF_Samples_From_PC_Analytical_Integration), ...
 prctile(CDF_Samples_From_PC_Analytical_Integration, pct_high)]



%% Plots Epistemic Samples!!


figure

plot(1:N_real_xi, CDF_Samples_From_PC_Numerical_Integration, 'LineWidth',lw)
hold on
plot(1:N_real_xi, CDF_Samples_From_PC_Analytical_Integration, '--', 'LineWidth',lw)

xlabel('Number of Sample')
ylabel('Probability of Rupture Per Epistemic Sample')

set(gca, 'Fontsize', fs)


%% Plots PC Terms!!

figure
plot(1:num_pc_terms_total, PC_Coeffs_CDF_PDF_Analytical_Integration, 'LineWidth',lw)
hold on
plot(1:num_pc_terms_total, PC_Coeffs_CDF_PDF_Numerical_Integration, '--', 'LineWidth',lw)

set(gca, 'Fontsize', fs)

xlabel('Number of PC Term')
ylabel('PC Term Value')