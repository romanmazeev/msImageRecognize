function [mu,sigma] = compute_fwhm(input, num_bins)
% ------------------------------------------------------------------------
% [mu, sigma] = compute_fwhm(input, num_bins) 
% 
% Compute the Full Width Half Maximum FWHM of a voxel intensity tissue
% distribution. 
%  -input: 1D matrix containing the voxel intensities of a tissue
%    distribution
%  -num_bins: Number of bins used to generate the histogram.
%  
%  The function returns the \mu and \sigma as extracted from the generated
%  histogram
%
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ------------------------------------------------------------------------
  
    
    %[h_frequencies_, h_centers_] = hist(input, num_bins);
    %h_frequencies = h_frequencies_(2:end);
    %h_centers = h_centers_(2:end);
    
    [h_frequencies, h_centers] = hist(input, num_bins);    
    h_frequencies(1) = 0;
        
    [maxbin, pos_maxbin] = max(h_frequencies);
    
    %[~, freq_pos] = sort(h_frequencies,'descend');
    
    %pos_maxbin = freq_pos(2)
    %maxbin = h_frequencies(pos_maxbin)
    
    % mu estimation: 
    % mu estimation appears different to ITK. Computed using both ways 
    mu = h_centers(pos_maxbin);
    %disp([num2str(pos_maxbin), ' - ', num2str(mu)]);
    
    % sigma estimation
    % lower band
    pos_hx1 = find(h_frequencies ./ maxbin > 0.5, 1,'first') -1;
    hx1 = h_frequencies(pos_hx1) / maxbin;
    hx1m1 = h_frequencies(pos_hx1 +1) / maxbin;
    
    mx1 = h_centers(pos_hx1);
    mx1m1 = h_centers(pos_hx1 +1);
    loband = mx1+(0.5-hx1)*(mx1m1-mx1)/(hx1m1-hx1);
    
    %disp([num2str(pos_hx1), ' - ', num2str(mx1), ' - ', num2str(mx1m1)]);
    %disp([num2str(pos_hx1), ' - ', num2str(hx1), ' - ', num2str(hx1m1)]);
    
    % higher band
    pos_hx2 = find((h_frequencies ./ maxbin) > 0.5, 1, 'last') +1;
    hx2 = h_frequencies(pos_hx2) / maxbin;
    hx2m1 = h_frequencies(pos_hx2 -1) / maxbin;
    
    mx2 = h_centers(pos_hx2);
    mx2m1 = h_centers(pos_hx2 -1);
    hiband = mx2m1 -(0.5-hx2)*(mx2m1-mx2)/(hx2m1-hx2);
    
    %disp([num2str(pos_hx2), ' - ', num2str(mx2), ' - ', num2str(mx2m1)]);
    %disp([num2str(pos_hx2), ' - ', num2str(hx2), ' - ', num2str(hx2m1)]);

    % sigma
    sigma=(hiband-loband)/(2*sqrt(2*log(2)));
   
 end