function h_coeffs=Total_Hazard_TE_Sum_new_function(sumX, sumY, sumXsquared, sumXY,  ...
  sumYsquared, numBins, f_coeffs)

order_TE = size(f_coeffs, 2) - 1;


if order_TE == 0
    
    f0 = f_coeffs(:, 1);

    h_0=(1/2).*(2.*f0.*numBins);
    h_coeffs = h_0;

end


if order_TE == 1
    
    f0 = f_coeffs(:, 1);
    f1 = f_coeffs(:, 2);

    h_0=(1/2).*(2.*f0.*numBins+(-2).*f1.*sumX);
    h_1=f1.*numBins;
    h_coeffs = [h_0, h_1];

end

if order_TE == 2
    
    f0 = f_coeffs(:, 1);
    f1 = f_coeffs(:, 2);
    f2 = f_coeffs(:, 3);

    h_0=(1/2).*(2.*f0.*numBins+(-2).*f1.*sumX+ 2.*f2.*sumY);
    h_1=f1.*numBins;
    h_2=(-1).*f2.*numBins;
    h_coeffs = [h_0, h_1, h_2];

end


if order_TE == 3
    
    f0 = f_coeffs(:, 1);
    f1 = f_coeffs(:, 2);
    f2 = f_coeffs(:, 3);
    f3 = f_coeffs(:, 4);

    h_0=(1/2).*(2.*f0.*numBins+(-2).*f1.*sumX+f3.*sumXsquared+ 2.*f2.*sumY);
    h_1=f1.*numBins+(-1).*f3.*sumX;
    h_2=(-1).*f2.*numBins;
    h_3=(1/2).*f3.*numBins;
    h_coeffs = [h_0, h_1, h_2, h_3];

end


if order_TE == 4
    
    f0 = f_coeffs(:, 1);
    f1 = f_coeffs(:, 2);
    f2 = f_coeffs(:, 3);
    f3 = f_coeffs(:, 4);
    f4 = f_coeffs(:, 5);

    h_0=(1/2).*(2.*f0.*numBins+(-2).*f1.*sumX+f3.*sumXsquared+(-2).*f4.*sumXY+ ...
      2.*f2.*sumY);
    h_1=f1.*numBins+(-1).*f3.*sumX+f4.*sumY;
    h_2=(-1).*f2.*numBins+f4.*sumX;
    h_3=(1/2).*f3.*numBins;
    h_4=(-1).*f4.*numBins;
    h_coeffs = [h_0, h_1, h_2, h_3, h_4];

end


if order_TE == 5
    
    f0 = f_coeffs(:, 1);
    f1 = f_coeffs(:, 2);
    f2 = f_coeffs(:, 3);
    f3 = f_coeffs(:, 4);
    f4 = f_coeffs(:, 5);
    f5 = f_coeffs(:, 6);

    h_0=(1/2).*(2.*f0.*numBins+(-2).*f1.*sumX+f3.*sumXsquared+(-2).*f4.*sumXY+ ...
      2.*f2.*sumY+f5.*sumYsquared);
    h_1=f1.*numBins+(-1).*f3.*sumX+f4.*sumY;
    h_2=(-1).*f2.*numBins+f4.*sumX+(-1).*f5.*sumY;
    h_3=(1/2).*f3.*numBins;
    h_4=(-1).*f4.*numBins;
    h_5=(1/2).*f5.*numBins;
    h_coeffs = [h_0, h_1, h_2, h_3, h_4, h_5];

end