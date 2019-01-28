function [candidate_region_mask] = extract_candidates(norm_flair, t1seg_img, alpha)
% **************************************************************************************************
% EXTRACT CANDIDATES from FLAIR
%
%  input:
%   -> norm_flair = FLAIR image
%   -> t1_seg_img = t1 segmentation (3 class discrete)
%   -> alpha = alpha parameter
%
%  output:
%   -> candidate_region_mask = mask containing all the Flair hyperintense regions.
%
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ***************************************************************************************************

    if sum(nonzeros(norm_flair)) > 0
        norm_flair = normalize_scan(norm_flair);
        [m_gm, s_gm] = compute_fwhm(norm_flair(t1seg_img == 2),512);
        th = m_gm + (alpha*s_gm);
        candidate_region_mask = norm_flair > th;
    else
        candidate_region_mask = zeros(size(norm_flair));
    end
end    
    
    
    
        
