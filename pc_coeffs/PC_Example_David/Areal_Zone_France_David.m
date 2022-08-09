%% Site coordinates

site_coord = [43.6748 5.7664];


%% Imports Ground-Motion Realizations

GMM_data = xlsread('adj_RE_Site1_5Hz_SA20.xlsx');
n_rows = 5554;

grid_pts_coord = [GMM_data(4:n_rows, 2) GMM_data(4:n_rows, 3)]; 

median_samples = GMM_data(4:n_rows, 4:103);

median_adjustements_vector = GMM_data(4:n_rows, 104);
sd_adjustements_vector = GMM_data(4:n_rows, 105);


%% Interpolates grid points with finer mesh 

lat_lim = [min(grid_pts_coord(:, 1)) max(grid_pts_coord(:, 1))];
lon_lim = [min(grid_pts_coord(:, 2)) max(grid_pts_coord(:, 2))];

delta_lat = km2deg(1);
delta_lon = km2deg(1);

lat_finer_vector = lat_lim(1):delta_lat:lat_lim(2);
lon_finer_vector = lon_lim(1):delta_lon:lon_lim(2);


[lat_grid_finer, lon_grid_finer] = meshgrid(lat_finer_vector, lon_finer_vector);

grid_pts_coord_finer = [reshape(lat_grid_finer, [], 1) reshape(lon_grid_finer, [], 1) ];


%%% Interpolates median adjustement terms and standard deviations 

F_median = scatteredInterpolant(grid_pts_coord(:, 1), grid_pts_coord(:, 2), median_adjustements_vector);
F_sd = scatteredInterpolant(grid_pts_coord(:, 1), grid_pts_coord(:, 2), sd_adjustements_vector);

median_adjustments_finer = F_median(grid_pts_coord_finer(:, 1), grid_pts_coord_finer(:, 2));
sd_adjustments_finer = F_sd(grid_pts_coord_finer(:, 1), grid_pts_coord_finer(:, 2));




%% Imports Polygons

Polygons_SSC = xlsread('France_SSC_V5b.xlsx');


% Isolates polygons 

Polygons_only = find(abs(Polygons_SSC(:, 2)) > 1);
Polygons_only(1) = [];
Polygons_groups = find(abs(diff(Polygons_only))>1);



num_polygons = 1;

Polygons_indices{1} = 14:30;

Polygons_Edges{1} = [Polygons_SSC(Polygons_indices{1}, 2) Polygons_SSC(Polygons_indices{1}, 1)];





%% Finds non-ergodic median adjustment terms inside polygons

delta_mu_non_ergodic = cell(1, 1);
sigma_mu_non_ergodic = cell(1, 1);


grid_pts_inside_coord_finer = cell(1, 1);
grid_pts_inside_coord_coarser = cell(1, 1);


polygon_check = 1;


in_coarser = inpolygon(grid_pts_coord(:, 1), grid_pts_coord(:, 2), Polygons_Edges{polygon_check}(:, 1), Polygons_Edges{polygon_check}(:, 2));
in = inpolygon(grid_pts_coord_finer(:, 1), grid_pts_coord_finer(:, 2), Polygons_Edges{polygon_check}(:, 1), Polygons_Edges{polygon_check}(:, 2));

indices_inside_polygon_coarser = find(in_coarser);
indices_inside_polygon = find(in);

grid_pts_inside_coord_coarser{polygon_check} = [grid_pts_coord(indices_inside_polygon_coarser, 1) grid_pts_coord(indices_inside_polygon_coarser, 2)];

grid_pts_inside_coord_finer{polygon_check} = [grid_pts_coord_finer(indices_inside_polygon, 1) grid_pts_coord_finer(indices_inside_polygon, 2)];


delta_mu_non_ergodic{polygon_check} = median_adjustments_finer(indices_inside_polygon);
sigma_mu_non_ergodic{polygon_check} = sd_adjustments_finer(indices_inside_polygon);


%% Plots polygon of areal source zone

figure('Units','normalized','Position',[0 0 1 1])

geoplot(site_coord(:, 1), site_coord(:, 2), 'k^', 'Linewidth', 2)

hold on


geoplot(Polygons_Edges{polygon_check}(:, 1), Polygons_Edges{polygon_check}(:, 2), '-', 'Linewidth', 2)
geoscatter(grid_pts_inside_coord_coarser{polygon_check}(:, 1), grid_pts_inside_coord_coarser{polygon_check}(:, 2), '.r', 'Linewidth', 2)
% geoscatter(grid_pts_inside_coord_finer{polygon_check}(1:5:end, 1), grid_pts_inside_coord_finer{polygon_check}(1:5:end, 2), '.r', 'Linewidth', 0.1)


legend('Site', 'Polygon', 'Coarser Mesh for Areal Source')

geobasemap colorterrain

geolimits([40 45], [2 7])
set(gca, 'Fontsize', 17)



%% Hazard Parameters 


%%% Ground-motion values z

num_pts_z = 1*1E2;
z_vector = logspace(-3, 1, num_pts_z);


%%% Median Ergodic parameters

Magnitude_Scenario = 5.05;

Distance_Values = [1.0 3.0 5.0 7.5 10.0 15.0 20.0 30.0 40.0 50.0 70.0 100.0 150.0 205.0]; %km

% Ergodic median for 1 Magnitude_Scenario and 13 distance bins

median_SA_ergodic = {-0.7952   -1.1393   -1.4515   -1.7712   -2.0848   -2.5740   -2.9668   -3.5653   -4.0079   -4.3484   -4.8710   -5.5014   -6.3410};

% Aleatory sigma

sigma_SA = 0.4;

% Example activity rate

rate_bin = 1/10000;


Bins_vector = 1:(length(Distance_Values)-1);


%%% Taylor Expansion parameters 

orderTE = 5;


%%% Epistemic Uncertainty Parameters 

N_Samples_Monte_Carlo = 1E2;

xi_real = normrnd(0, 1, 1, N_Samples_Monte_Carlo);



%% Finds grid points within radius 


polygon_loop = 1;
    
dist_site_grid = sqrt(...
                     deg2km(site_coord(:, 1) - grid_pts_inside_coord_finer{polygon_loop}(:, 1)).^2 + ...
                     deg2km(site_coord(:, 2) - grid_pts_inside_coord_finer{polygon_loop}(:, 2)).^2 ...
                     );
                 
                     
indices_pts_in_bin = cell(length(Distance_Values)-1, 1);

for r = 1:length(Distance_Values)-1
    
    indices_pts_in_bin{r} = find(dist_site_grid > Distance_Values(r) & dist_site_grid < Distance_Values(r+1));
    
end




%% Pre-processing of isotropic distance bins 



%%% Loop over Bins 

mu_centered_in_bin = cell(1, length(Bins_vector));
delta_mu_centered = cell(1, length(Bins_vector));
sigma_mu_non_ergodic_in_bin = cell(1, length(Bins_vector));
num_sources_in_bin_vector = cell(1, length(Bins_vector));

exp_sum_med_vector = cell(1, length(Bins_vector));
exp_sum_med_squared_vector = cell(1, length(Bins_vector));
sum_sd_in_bin_vector = cell(1, length(Bins_vector));
sum_sd_in_bin_squared_vector = cell(1, length(Bins_vector));
point_of_TE_sigma_mu_vector = cell(1, length(Bins_vector));
sum_z_shift_vector = cell(1, length(Bins_vector));
sum_z_shift_squared_vector = cell(1, length(Bins_vector));
sumXY_vector = cell(1, length(Bins_vector));
TE_Matrix_vector = cell(1, length(Bins_vector));

point_of_TE_vec = log(z_vector)'; % in Log Scale 
exp_point_of_TE_vec = z_vector';

for bin_loop = Bins_vector


indices_in_bin = indices_pts_in_bin{bin_loop};
num_sources_in_bin = length(indices_in_bin);

num_sources_in_bin_vector{bin_loop} = num_sources_in_bin;

delta_mu_non_ergodic_in_bin = zeros(1, num_sources_in_bin);

    for i = 1:num_sources_in_bin
    

        delta_mu_non_ergodic_in_bin(i) = delta_mu_non_ergodic{polygon_loop}(indices_in_bin(i));
        sigma_mu_non_ergodic_in_bin{bin_loop}(i) = sigma_mu_non_ergodic{polygon_loop}(indices_in_bin(i));

    end


mean_delta_mu_non_ergodic_in_bin = mean(delta_mu_non_ergodic_in_bin);

% Takes reference median with minimum adjustement term, so that all the shifts from that reference are positive and to the right 

mu_centered_in_bin{bin_loop} = median_SA_ergodic{bin_loop} + mean_delta_mu_non_ergodic_in_bin;

% Shifts all the median adjustements by the mean median in the bin 

delta_mu_centered{bin_loop} = delta_mu_non_ergodic_in_bin - mean_delta_mu_non_ergodic_in_bin;


%%% Taylor Expansion Pre-Processing

exp_sum_med =  sum(exp(delta_mu_centered{bin_loop}));
exp_sum_med_squared = sum(exp(delta_mu_centered{bin_loop}).^2);

sum_sd_in_bin = sum(sigma_mu_non_ergodic_in_bin{bin_loop});
sum_sd_in_bin_squared = sum(sigma_mu_non_ergodic_in_bin{bin_loop}.^2);

point_of_TE_sigma_mu = mean(sigma_mu_non_ergodic_in_bin{bin_loop});

sum_z_shift = exp_point_of_TE_vec * exp_sum_med;
sum_z_shift_squared = exp_point_of_TE_vec.^2 * exp_sum_med_squared;


sumXY = exp_point_of_TE_vec.*(exp(delta_mu_centered{bin_loop})*sigma_mu_non_ergodic_in_bin{bin_loop}');

TE_Matrix = [ones(num_pts_z, 1), z_vector', point_of_TE_sigma_mu*ones(num_pts_z, 1), (z_vector)'.^2, (z_vector.*point_of_TE_sigma_mu)', point_of_TE_sigma_mu^2* ones(num_pts_z, 1)];



%%%

exp_sum_med_vector{bin_loop} =  exp_sum_med;
exp_sum_med_squared_vector{bin_loop} = exp_sum_med_squared;

sum_sd_in_bin_vector{bin_loop} = sum_sd_in_bin;
sum_sd_in_bin_squared_vector{bin_loop} = sum_sd_in_bin_squared;

point_of_TE_sigma_mu_vector{bin_loop} = point_of_TE_sigma_mu;

sum_z_shift_vector{bin_loop} = sum_z_shift;
sum_z_shift_squared_vector{bin_loop} = sum_z_shift_squared;


sumXY_vector{bin_loop} = sumXY;

TE_Matrix_vector{bin_loop} = TE_Matrix(:, 1: (1+orderTE));

%%%


end

%% PC Hazard Calculation 


Total_PC_Term_0 = zeros(1, length(z_vector));
Total_PC_Term_1 = zeros(1, length(z_vector));
Total_PC_Term_2 = zeros(1, length(z_vector));
Total_PC_Term_3 = zeros(1, length(z_vector));
Total_PC_Term_4 = zeros(1, length(z_vector));


tic

for bin_loop = Bins_vector

    num_sources_in_bin = num_sources_in_bin_vector{bin_loop};

    %%% Direct PC Hazard calculation with all sources in each distance bin

    PC_term_0_in_bin = zeros(num_sources_in_bin, length(z_vector));
    PC_term_1_in_bin = zeros(num_sources_in_bin, length(z_vector));
    PC_term_2_in_bin = zeros(num_sources_in_bin, length(z_vector));
    PC_term_3_in_bin = zeros(num_sources_in_bin, length(z_vector));
    PC_term_4_in_bin = zeros(num_sources_in_bin, length(z_vector));



    for i = 1:num_sources_in_bin

        a = -sigma_mu_non_ergodic_in_bin{bin_loop}(i)^2/(2*sigma_SA^2) - 1/2;

        b =  (log(z_vector) - (mu_centered_in_bin{bin_loop} + delta_mu_centered{bin_loop}(i))) * sigma_mu_non_ergodic_in_bin{bin_loop}(i)/(sigma_SA^2);

        c = - (log(z_vector) - (mu_centered_in_bin{bin_loop} + delta_mu_centered{bin_loop}(i))).^2/(2*sigma_SA^2);

        alpha = sigma_mu_non_ergodic_in_bin{bin_loop}(i)/(sigma_SA *2*sqrt(pi)) * exp(c - b.^2/(4*a));

        PC_term_0 = 1/1 * (1 - normcdf((log(z_vector) - (mu_centered_in_bin{bin_loop} + delta_mu_centered{bin_loop}(i)))/...
                                        sqrt(sigma_SA^2 + sigma_mu_non_ergodic_in_bin{bin_loop}(i)^2))...
                           );

        PC_term_1 = 1/1 * (1/sqrt(-a)) * alpha;
        PC_term_2 = 1/2 * alpha .*b/(2*(-a)^(3/2));
        PC_term_3 = 1/6 * alpha .*(-2*a*(1 + 2*a) + b.^2)/(4*(-a)^(5/2));
        PC_term_4 = 1/24 * alpha .* (-b).*(6*a*(1 + 2*a) - b.^2)/(8*(-a)^(7/2));


        PC_term_0_in_bin(i, :) = rate_bin * PC_term_0;
        PC_term_1_in_bin(i, :) = rate_bin * PC_term_1;
        PC_term_2_in_bin(i, :) = rate_bin * PC_term_2;
        PC_term_3_in_bin(i, :) = rate_bin * PC_term_3;
        PC_term_4_in_bin(i, :) = rate_bin * PC_term_4;


    end


PC_Term_0_in_bin_sum = sum(PC_term_0_in_bin, 1);
PC_Term_1_in_bin_sum = sum(PC_term_1_in_bin, 1);
PC_Term_2_in_bin_sum = sum(PC_term_2_in_bin, 1);
PC_Term_3_in_bin_sum = sum(PC_term_3_in_bin, 1);
PC_Term_4_in_bin_sum = sum(PC_term_4_in_bin, 1);



Total_PC_Term_0 = Total_PC_Term_0 + PC_Term_0_in_bin_sum;
Total_PC_Term_1 = Total_PC_Term_1 + PC_Term_1_in_bin_sum;
Total_PC_Term_2 = Total_PC_Term_2 + PC_Term_2_in_bin_sum;
Total_PC_Term_3 = Total_PC_Term_3 + PC_Term_3_in_bin_sum;
Total_PC_Term_4 = Total_PC_Term_4 + PC_Term_4_in_bin_sum;
    

end

time_PC = toc;



%% Direct Hazard calculation 

Hazard_Samples_MC_Total_Full = zeros(length(z_vector), N_Samples_Monte_Carlo);




tic

for bin_loop = Bins_vector
    
    num_sources_in_bin = num_sources_in_bin_vector{bin_loop};

    for i = 1:num_sources_in_bin
        for j = 1:N_Samples_Monte_Carlo
            
            Hazard_Samples_MC_Total_Full(:, j) = Hazard_Samples_MC_Total_Full(:, j) + rate_bin * (1 - normcdf((log(z_vector)' - (mu_centered_in_bin{bin_loop} + delta_mu_centered{bin_loop}(i) + sigma_mu_non_ergodic_in_bin{bin_loop}(i) * xi_real(j)))/...
                sqrt(sigma_SA^2)...
                ));
        end
        
        
       
    end
    
            
    
    disp(strcat('Finished Monte-Carlo sampling for distance bin ', {' '}, num2str(bin_loop), '/', num2str(Bins_vector(end))))

        
        
end

time_MC_full = toc;



%% Taylor Expansion Vectorized with PC Terms 


Total_PC_Term_0_TE = zeros(1, length(z_vector));
Total_PC_Term_1_TE = zeros(1, length(z_vector));
Total_PC_Term_2_TE = zeros(1, length(z_vector));
Total_PC_Term_3_TE = zeros(1, length(z_vector));
Total_PC_Term_4_TE = zeros(1, length(z_vector));




tic

for bin_loop = Bins_vector % loop over event/scenario


%%% TE Loop over z_vector 

num_sources_in_bin = num_sources_in_bin_vector{bin_loop};

exp_sum_med =  exp_sum_med_vector{bin_loop};
exp_sum_med_squared = exp_sum_med_squared_vector{bin_loop};

sum_sd_in_bin = sum_sd_in_bin_vector{bin_loop};
sum_sd_in_bin_squared = sum_sd_in_bin_squared_vector{bin_loop};

point_of_TE_sigma_mu = point_of_TE_sigma_mu_vector{bin_loop};

sum_z_shift = sum_z_shift_vector{bin_loop};
sum_z_shift_squared = sum_z_shift_squared_vector{bin_loop};


sumXY = sumXY_vector{bin_loop};

TE_Matrix = TE_Matrix_vector{bin_loop};

%%% Taylor Expansion Approximation 


% All PC Terms together

f_coeffs_bin_total = Total_Hazard_TE_Truncated_function(exp_point_of_TE_vec, point_of_TE_vec, mu_centered_in_bin{bin_loop}, sigma_SA, point_of_TE_sigma_mu, orderTE);
f_coeffs_bin_0 = reshape(f_coeffs_bin_total(1, :, :), [num_pts_z orderTE+1]);
f_coeffs_bin_1 = reshape(f_coeffs_bin_total(2, :, :), [num_pts_z orderTE+1]);
f_coeffs_bin_2 = reshape(f_coeffs_bin_total(3, :, :), [num_pts_z orderTE+1]);
f_coeffs_bin_3 = reshape(f_coeffs_bin_total(4, :, :), [num_pts_z orderTE+1]);
f_coeffs_bin_4 = reshape(f_coeffs_bin_total(5, :, :), [num_pts_z orderTE+1]);



% New Sum function with select orderTE

h_coeffs_0 = Total_Hazard_TE_Sum_new_function(sum_z_shift, sum_sd_in_bin, sum_z_shift_squared, sumXY, sum_sd_in_bin_squared, num_sources_in_bin, f_coeffs_bin_0);
h_coeffs_1 = Total_Hazard_TE_Sum_new_function(sum_z_shift, sum_sd_in_bin, sum_z_shift_squared, sumXY, sum_sd_in_bin_squared, num_sources_in_bin, f_coeffs_bin_1);
h_coeffs_2 = Total_Hazard_TE_Sum_new_function(sum_z_shift, sum_sd_in_bin, sum_z_shift_squared, sumXY, sum_sd_in_bin_squared, num_sources_in_bin, f_coeffs_bin_2);
h_coeffs_3 = Total_Hazard_TE_Sum_new_function(sum_z_shift, sum_sd_in_bin, sum_z_shift_squared, sumXY, sum_sd_in_bin_squared, num_sources_in_bin, f_coeffs_bin_3);
h_coeffs_4 = Total_Hazard_TE_Sum_new_function(sum_z_shift, sum_sd_in_bin, sum_z_shift_squared, sumXY, sum_sd_in_bin_squared, num_sources_in_bin, f_coeffs_bin_4);




PC_Term_0_in_bin_sum_TE = rate_bin * (sum(h_coeffs_0 .* TE_Matrix, 2)); % rate = rate for event
PC_Term_1_in_bin_sum_TE = rate_bin * (sum(h_coeffs_1 .* TE_Matrix, 2));
PC_Term_2_in_bin_sum_TE = rate_bin * (sum(h_coeffs_2 .* TE_Matrix, 2));
PC_Term_3_in_bin_sum_TE = rate_bin * (sum(h_coeffs_3 .* TE_Matrix, 2));
PC_Term_4_in_bin_sum_TE = rate_bin * (sum(h_coeffs_4 .* TE_Matrix, 2));


Total_PC_Term_0_TE = Total_PC_Term_0_TE + PC_Term_0_in_bin_sum_TE';
Total_PC_Term_1_TE = Total_PC_Term_1_TE + PC_Term_1_in_bin_sum_TE';
Total_PC_Term_2_TE = Total_PC_Term_2_TE + PC_Term_2_in_bin_sum_TE';
Total_PC_Term_3_TE = Total_PC_Term_3_TE + PC_Term_3_in_bin_sum_TE';
Total_PC_Term_4_TE = Total_PC_Term_4_TE + PC_Term_4_in_bin_sum_TE';


end


time_TE = toc;


%% Displays elapsed time for the three approaches

disp(['Time elapsed with MC and Full correlation and ', num2str(N_Samples_Monte_Carlo), ' ', 'samples: ' , num2str(time_MC_full), 's'])

disp(strcat('Time elapsed with PC only: ' , ' ', num2str(time_PC), 's'))

disp(strcat('Time elapsed with PC and Taylor Expansion: ' , num2str(time_TE), 's'))





%% Post-Processing 



Hermite_Samples = [ones(1, N_Samples_Monte_Carlo) ; ...
                   Hermite_Poly(xi_real, 1) ; ...
                   Hermite_Poly(xi_real, 2) ; ...
                   Hermite_Poly(xi_real, 3) ; ...
                   Hermite_Poly(xi_real, 4)];

% Direct PC Terms 

Total_Hazard_PC_Terms = [Total_PC_Term_0; ...
                        Total_PC_Term_1; ...
                        Total_PC_Term_2; ...
                        Total_PC_Term_3; ...
                        Total_PC_Term_4];
                    
Total_Hazard_PC_Terms_Samples = Total_Hazard_PC_Terms' * Hermite_Samples;

% Taylor Expansion 

Total_Hazard_PC_Terms_TE = [Total_PC_Term_0_TE; ...
                        Total_PC_Term_1_TE; ...
                        Total_PC_Term_2_TE; ...
                        Total_PC_Term_3_TE; ...
                        Total_PC_Term_4_TE];
                    
Total_Hazard_PC_Terms_TE_Samples = Total_Hazard_PC_Terms_TE' * Hermite_Samples;


%%% Percentiles 

percentile_lower = 10;
percentile_median = 50;
percentile_higher = 90;

%%% PC

percentile_lower_PC = zeros(1, length(z_vector));
percentile_median_PC = zeros(1, length(z_vector));
percentile_higher_PC = zeros(1, length(z_vector));


for i = 1:length(z_vector)
    
    percentile_lower_PC(1, i) = prctile(Total_Hazard_PC_Terms_Samples(i, :), percentile_lower, 2);
    percentile_median_PC(1, i) = prctile(Total_Hazard_PC_Terms_Samples(i, :), percentile_median, 2);
    percentile_higher_PC(1, i) = prctile(Total_Hazard_PC_Terms_Samples(i, :), percentile_higher, 2);    
    
end


%%% Taylor Expansion 

percentile_lower_TE = zeros(1, length(z_vector));
percentile_median_TE = zeros(1, length(z_vector));
percentile_higher_TE = zeros(1, length(z_vector));


for i = 1:length(z_vector)
    
    percentile_lower_TE(1, i) = prctile(Total_Hazard_PC_Terms_TE_Samples(i, :), percentile_lower, 2);
    percentile_median_TE(1, i) = prctile(Total_Hazard_PC_Terms_TE_Samples(i, :), percentile_median, 2);
    percentile_higher_TE(1, i) = prctile(Total_Hazard_PC_Terms_TE_Samples(i, :), percentile_higher, 2);    
    
end


%%% Monte-Carlo 

percentile_lower_MC = zeros(1, length(z_vector));
percentile_median_MC = zeros(1, length(z_vector));
percentile_higher_MC = zeros(1, length(z_vector));


for i = 1:length(z_vector)
    
    percentile_lower_MC(1, i) = prctile(Hazard_Samples_MC_Total_Full(i, :), percentile_lower, 2);
    percentile_median_MC(1, i) = prctile(Hazard_Samples_MC_Total_Full(i, :), percentile_median, 2);
    percentile_higher_MC(1, i) = prctile(Hazard_Samples_MC_Total_Full(i, :), percentile_higher, 2);    
    
end



%% Plot Direct vs TE 

fs = 17;
lw = 3;

figure

loglog(z_vector, percentile_lower_MC, '-.b', 'Linewidth', lw)
hold on
loglog(z_vector, percentile_median_MC, '--b', 'Linewidth', lw)
loglog(z_vector, percentile_higher_MC, '-.b', 'Linewidth', lw)
loglog(z_vector, mean(Hazard_Samples_MC_Total_Full, 2), '-b', 'Linewidth', lw)


loglog(z_vector, percentile_lower_PC, '+k', 'Linewidth', lw)
hold on
loglog(z_vector, percentile_median_PC, 'sk', 'Linewidth', lw)
loglog(z_vector, percentile_higher_PC, '^k', 'Linewidth', lw)
loglog(z_vector, mean(Total_Hazard_PC_Terms_Samples, 2), 'ok', 'Linewidth', lw)

loglog(z_vector, percentile_lower_TE, ':r', 'Linewidth', lw)
loglog(z_vector, percentile_median_TE, '-.r', 'Linewidth', lw)
loglog(z_vector, percentile_higher_TE, ':r', 'Linewidth', lw)
loglog(z_vector, mean(Total_Hazard_PC_Terms_TE_Samples, 2), '--r', 'Linewidth', lw)



xlabel('SA (g)')
ylabel('Total Hazard')
% title('Full Correlation: Monte-Carlo vs PC Approximation')
legend(strcat(num2str(percentile_lower), 'th Percentile of Total Hazard MC'), ...
       strcat(num2str(percentile_median), 'th Percentile of Total Hazard MC'), ...
       strcat(num2str(percentile_higher), 'th Percentile of Total Hazard MC'), ...
       'Mean Hazard MC', ...
       strcat(num2str(percentile_lower), 'th Percentile of Total Hazard PC'), ...
       strcat(num2str(percentile_median), 'th Percentile of Total Hazard PC'), ...
       strcat(num2str(percentile_higher), 'th Percentile of Total Hazard PC'), ...
       'Mean Hazard PC', ...
       strcat(num2str(percentile_lower), 'th Percentile of Total Hazard TE'), ...
       strcat(num2str(percentile_median), 'th Percentile of Total Hazard TE'), ...
       strcat(num2str(percentile_higher), 'th Percentile of Total Hazard TE'), ...
       'Mean Hazard TE')   
   
   
   
set(gca, 'Fontsize', fs)
% set(findall(gca, 'Type', 'Line'),'LineWidth',lw);

% ylim([1E-3 1E2])
ylim([max(mean(Total_Hazard_PC_Terms_Samples, 2))/1000 max(mean(Total_Hazard_PC_Terms_Samples, 2))])

