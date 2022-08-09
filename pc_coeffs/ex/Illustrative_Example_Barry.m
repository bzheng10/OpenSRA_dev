%% Octave packages

pkg load statistics

%% Hazard curve parameters

mean_of_mu_example = 0; % mean

sigma_of_mu_example = 0.2; % epistemic

sigma_SA_example = 0.5; % aleatory

rate = 1;

z_total = linspace(1E-2, 3, 20); % z score



%% Monte-Carlo sampling

N_real_Median = 10000;

xi_median = normrnd(0, 1, 1, N_real_Median);

Hazard_MC_samples = zeros(N_real_Median, length(log(z_total)));

for i = 1:N_real_Median
    
    
    Hazard_MC_samples(i, :) = rate*...
                            (1 - normcdf((log(z_total) - ...
                            (mean_of_mu_example + sigma_of_mu_example*xi_median(i)))/sigma_SA_example));
    
end



%% PC expansion

a = -sigma_of_mu_example^2/(2*sigma_SA_example^2) - 1/2;

b =  (log(z_total) - mean_of_mu_example) * sigma_of_mu_example/(sigma_SA_example^2);

c = - (log(z_total) - mean_of_mu_example).^2/(2*sigma_SA_example^2);

alpha = sigma_of_mu_example/(sigma_SA_example *2*sqrt(pi)) * exp(c - b.^2/(4*a));


PC_term_0 = 1/1 * (1 - normcdf((log(z_total) - mean_of_mu_example)/sqrt(sigma_of_mu_example^2 + sigma_SA_example^2)));


PC_term_1 = 1/1 * (1/sqrt(-a)) * alpha;


PC_term_2 =  1/2 * alpha .*b/(2*(-a)^(3/2));


PC_term_3 = 1/6 * alpha .*(-2*a*(1 + 2*a) + b.^2)/(4*(-a)^(5/2));


PC_term_4 = 1/24 * alpha .* (-b).*(6*a*(1 + 2*a) - b.^2)/(8*(-a)^(7/2));


PC_term_5 = 1/120 * alpha .* (12*a^2*(1+ 2*a)^2 - 12*a*(1 + 2*a)*b.^2 + b.^4)/(16*(-a)^(9/2));


PC_term_6 = 1/720 * alpha .* b.*(60*a^2*(1+ 2*a)^2 - 20*a*(1 + 2*a)*b.^2 + b.^4)/(32*(-a)^(11/2));




%% Post-Processing

%%% PC

Hazard_PC_samples = zeros(length(z_total), N_real_Median);

for i = 1:length(z_total)
    
Hazard_PC_samples(i, :) =     PC_term_0(i) + ...
                              PC_term_1(i) * Hermite_Proba(xi_median, 1) + ...
                              PC_term_2(i) * Hermite_Proba(xi_median, 2) + ...
                              PC_term_3(i) * Hermite_Proba(xi_median, 3) + ...
                              PC_term_4(i) * Hermite_Proba(xi_median, 4) + ...
                              PC_term_5(i) * Hermite_Proba(xi_median, 5);% + ...
%                               PC_term_6(i) * Hermite_Proba(xi_median, 6);% + ...
                              %PC_term_7(i) * Hermite_Proba(xi_median, 7) + ...
                              %PC_term_8(i) * Hermite_Proba(xi_median, 8);

end
    

percentile_10_PC_example = zeros(1, length(z_total));
percentile_50_PC_example = zeros(1, length(z_total));
percentile_90_PC_example = zeros(1, length(z_total));




for i = 1:length(z_total)
    percentile_10_PC_example(1, i) = prctile(Hazard_PC_samples(i, :), 5, 2);
    percentile_50_PC_example(1, i) = prctile(Hazard_PC_samples(i, :), 50, 2);
    percentile_90_PC_example(1, i) = prctile(Hazard_PC_samples(i, :), 95, 2);
    
end



% MC

percentile_10_MC_example = zeros(1, length(z_total));
percentile_50_MC_example = zeros(1, length(z_total));
percentile_90_MC_example = zeros(1, length(z_total));



for i = 1:length(z_total)
    percentile_10_MC_example(1, i) = prctile(Hazard_MC_samples(:, i), 5, 1);
    percentile_50_MC_example(1, i) = prctile(Hazard_MC_samples(:, i), 50, 1);
    percentile_90_MC_example(1, i) = prctile(Hazard_MC_samples(:, i), 95, 1);
    
end



%% Plots percentiles

lw = 5; fs = 25;

z_to_plot = 1:length(z_total);

figure

loglog(z_total(z_to_plot), percentile_10_MC_example(z_to_plot), '--k','LineWidth', lw/2)
hold on
loglog(z_total(z_to_plot), percentile_50_MC_example(z_to_plot), '-.k','LineWidth',lw/2)
loglog(z_total(z_to_plot), percentile_90_MC_example(z_to_plot), ':k','LineWidth',lw/2)
loglog(z_total(z_to_plot), mean(Hazard_MC_samples(:, z_to_plot), 1), '-k','LineWidth',lw/2)

loglog(z_total(z_to_plot), percentile_10_PC_example(z_to_plot), 'sk','LineWidth',lw/1.5)
% hold on
loglog(z_total(z_to_plot), percentile_50_PC_example(z_to_plot), '^k','LineWidth',lw/1.5)
loglog(z_total(z_to_plot), percentile_90_PC_example(z_to_plot), 'ok','LineWidth',lw/1.5)
% loglog(x_total(x_to_plot), PC_term_0((x_to_plot)), '+k','LineWidth',lw/2)
loglog(z_total(z_to_plot), mean(Hazard_PC_samples((z_to_plot), :), 2), '*k','LineWidth',lw/2)


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
%set(findall(gca, 'Type', 'Line'),'LineWidth',2);
hold off
   








%% Plots some histograms at some z-value

z_check = 1.5;

z_index = find(z_total >= z_check, 1);
z_axis = logspace(-3, 0, 20);

figure
subplot(1, 2, 1)
%h1 = histogram(Hazard_MC_samples(:, z_index),z_axis, 'FaceColor', 'k');
h1 = hist(Hazard_MC_samples(:, z_index),z_axis, 'FaceColor', 'k');
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




