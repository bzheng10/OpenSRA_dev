function [median_lnd_vector, sd_lnd_vector] = Bray_Macedo_2018(mean_ky, mean_Ts, SA_vector, M, mean_of_mu_SA, type)
%BRAY_MACEDO_2018 Summary of this function goes here
%   Detailed explanation goes here

if mean_Ts <0.05
  a1 = -0.22;
  a2 = 0.0;
else
  a1 = -1.10;
  a2 = 1.50;
end

% median_lnd_vector = a1-2.83*log(mean_ky)-0.333*(log(mean_ky))^2+0.566*log(mean_ky)*log(SA_vector)+3.04*log(SA_vector)-0.244*(log(SA_vector)).^2 + a2*mean_Ts+0.278*1*(M-7);            

if strcmp(type, 'linear')
    %median_lnd_vector = a1-2.83*log(mean_ky)-0.333*(log(mean_ky)).^2+0.566*log(mean_ky).*log(SA_vector)+3.04*log(SA_vector)-0.0*(log(SA_vector)).^2 + a2*mean_Ts+0.278*1*(M-7);
    
slope_disp = 0.566*log(mean_ky) + 3.04 - 2*0.244*mean_of_mu_SA;

intercept_disp = a1-2.83*log(mean_ky)-0.333*(log(mean_ky)).^2+0.566*log(mean_ky).*mean_of_mu_SA+3.04*mean_of_mu_SA-...
                    0.244*(mean_of_mu_SA).^2 + a2*mean_Ts+0.278*1*(M-7);    
                
                
median_lnd_vector =  slope_disp .* (log(SA_vector) - mean_of_mu_SA) + intercept_disp;

elseif strcmp(type, 'quadratic')
    median_lnd_vector = a1-2.83*log(mean_ky)-0.333*(log(mean_ky)).^2+0.566*log(mean_ky).*log(SA_vector)+3.04*log(SA_vector)-0.244*(log(SA_vector)).^2 + a2*mean_Ts+0.278*1*(M-7);
end


sd_lnd_vector = 0.67 * ones(length(mean_ky), size(SA_vector, 2));
% sd_lnd_vector = 0.5 * ones(length(mean_ky), length(SA_vector));


end

