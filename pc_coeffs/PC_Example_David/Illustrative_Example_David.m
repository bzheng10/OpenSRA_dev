%% Hazard curve parameters

mean_of_mu = 0; % Mean of Median gorund motion

sigma_of_mu = 0.3; % Epistemic uncertainty sigma of the median ground motion

sigma_SA = 0.5; % Aleatory variability sigma 

rate = 1; % Activity rate


z_vector = linspace(1E-2, 3, 20); % Ground-motion values



%% Monte-Carlo sampling

N_Samples_Median = 10000;

xi_median = normrnd(0, 1, 1, N_Samples_Median);

Hazard_MC_samples = zeros(N_Samples_Median, length(log(z_vector)));

for i = 1:N_Samples_Median
    
    
    Hazard_MC_samples(i, :) = rate*...
                            (1 - normcdf((log(z_vector) - ...
                            (mean_of_mu + sigma_of_mu*xi_median(i)))/sigma_SA));
    
end



%% PC expansion

% PC_term_0_sum = some dimensions

% for i in scenarios

a = -sigma_of_mu^2/(2*sigma_SA^2) - 1/2;

b =  (log(z_vector) - mean_of_mu) * sigma_of_mu/(sigma_SA^2);

c = - (log(z_vector) - mean_of_mu).^2/(2*sigma_SA^2);

alpha = sigma_of_mu/(sigma_SA *2*sqrt(pi)) * exp(c - b.^2/(4*a));

			 
			
PC_term_0 = 1/1 * (1 - normcdf((log(z_vector) - mean_of_mu)/sqrt(sigma_of_mu^2 + sigma_SA^2)));


PC_term_1 = 1/1 * (1/sqrt(-a)) * alpha;


PC_term_2 =  1/2 * alpha .*b/(2*(-a)^(3/2));
				  
				  
PC_term_3 = 1/6 * alpha .*(-2*a*(1 + 2*a) + b.^2)/(4*(-a)^(5/2));


PC_term_4 = 1/24 * alpha .* (-b).*(6*a*(1 + 2*a) - b.^2)/(8*(-a)^(7/2));


PC_term_5 = 1/120 * alpha .* (12*a^2*(1+ 2*a)^2 - 12*a*(1 + 2*a)*b.^2 + b.^4)/(16*(-a)^(9/2));


PC_term_6 = 1/720 * alpha .* b.*(60*a^2*(1+ 2*a)^2 - 20*a*(1 + 2*a)*b.^2 + b.^4)/(32*(-a)^(11/2));


% PC_term_0_sum = PC_term_0_sum + rate_scenario * PC_term_0
% PC_term_1_sum = PC_term_0_sum + rate_scenario * PC_term_0
% PC_term_2_sum = PC_term_0_sum + rate_scenario * PC_term_0
% PC_term_3_sum = PC_term_0_sum + rate_scenario * PC_term_0
% PC_term_4_sum = PC_term_0_sum + rate_scenario * PC_term_0






%% Post-Processing

%%% PC

Hazard_PC_samples = zeros(length(z_vector), N_Samples_Median);

for i = 1:length(z_vector)
    
Hazard_PC_samples(i, :) =     PC_term_0(i) + ...
                              PC_term_1(i) * Hermite_Poly(xi_median, 1) + ...
                              PC_term_2(i) * Hermite_Poly(xi_median, 2) + ...
                              PC_term_3(i) * Hermite_Poly(xi_median, 3) + ...
                              PC_term_4(i) * Hermite_Poly(xi_median, 4) + ...
                              PC_term_5(i) * Hermite_Poly(xi_median, 5) + ...
                              PC_term_6(i) * Hermite_Poly(xi_median, 6);
                              
                              
                              
                          
end
    

percentile_10_PC_example = zeros(1, length(z_vector));
percentile_50_PC_example = zeros(1, length(z_vector));
percentile_90_PC_example = zeros(1, length(z_vector));




for i = 1:length(z_vector)
    percentile_10_PC_example(1, i) = prctile(Hazard_PC_samples(i, :), 5, 2);
    percentile_50_PC_example(1, i) = prctile(Hazard_PC_samples(i, :), 50, 2);
    percentile_90_PC_example(1, i) = prctile(Hazard_PC_samples(i, :), 95, 2);
    
end



% MC

percentile_10_MC_example = zeros(1, length(z_vector));
percentile_50_MC_example = zeros(1, length(z_vector));
percentile_90_MC_example = zeros(1, length(z_vector));



for i = 1:length(z_vector)
    percentile_10_MC_example(1, i) = prctile(Hazard_MC_samples(:, i), 5, 1);
    percentile_50_MC_example(1, i) = prctile(Hazard_MC_samples(:, i), 50, 1);
    percentile_90_MC_example(1, i) = prctile(Hazard_MC_samples(:, i), 95, 1);
    
end



%% Plots percentiles

lw = 5; fs = 25;

z_to_plot = 1:length(z_vector);

figure('Units','normalized','Position',[0 0 1 1])


loglog(z_vector(z_to_plot), percentile_10_MC_example(z_to_plot), '--k','LineWidth', lw/2)
hold on
loglog(z_vector(z_to_plot), percentile_50_MC_example(z_to_plot), '-.k','LineWidth',lw/2)
loglog(z_vector(z_to_plot), percentile_90_MC_example(z_to_plot), ':k','LineWidth',lw/2)
loglog(z_vector(z_to_plot), mean(Hazard_MC_samples(:, z_to_plot), 1), '-k','LineWidth',lw/2)

loglog(z_vector(z_to_plot), percentile_10_PC_example(z_to_plot), 'sk','LineWidth',lw/1.5)
% hold on
loglog(z_vector(z_to_plot), percentile_50_PC_example(z_to_plot), '^k','LineWidth',lw/1.5)
loglog(z_vector(z_to_plot), percentile_90_PC_example(z_to_plot), 'ok','LineWidth',lw/1.5)
% loglog(x_total(x_to_plot), PC_term_0((x_to_plot)), '+k','LineWidth',lw/2)
loglog(z_vector(z_to_plot), mean(Hazard_PC_samples((z_to_plot), :), 2), '*k','LineWidth',lw/2)


xlabel('SA (g)')
ylabel('Probability of Exceedance')
title('Probability of Exceedance: Monte-Carlo vs PC Approximation')
legend('5th Percentile MC', ...
       '50th Percentile MC', ...
       '95th Percentile MC', ...
       'Mean MC', ...
       '5th Percentile PC', ...
       '50th Percentile PC', ...
       '95th Percentile PC', ...
       'Mean PC', ...
       'Location', 'SouthWest')   
   
set(gca, 'Fontsize', fs)
hold off
   








%% Plots some histograms at some z-value

z_check = 1.5;

z_index = find(z_vector >= z_check, 1);
z_axis = logspace(-3, 0, 20);

figure('Units','normalized','Position',[0 0 1 1])

subplot(1, 2, 1)
h1 = histogram(Hazard_MC_samples(:, z_index),z_axis, 'FaceColor', 'k');
h1_bins = h1.BinEdges;

xlabel('Probability of Exceedance')
title(strcat('Monte-Carlo at z = ', num2str(z_check), 'g'))
set(gca, 'Fontsize', 30)
%ylim([0 1500])
set(gca,'XScale','log')

subplot(1, 2, 2)

h2 = histogram(Hazard_PC_samples(z_index, :), z_axis, 'FaceColor', 'k');
h2.BinEdges = h1_bins;
xlabel('Probability of Exceedance')
title(strcat('Polynomial Chaos at z = ', num2str(z_check), 'g'))
set(gca, 'Fontsize', 30)
%ylim([0 1500])
set(gca,'XScale','log')





