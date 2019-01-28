function [lesion_candidates, T1_refilled] = find_lesion_candidates(flair_img, ...
                                                      t1_image,...
                                                      seg_out, ...
                                                      pve_out,...
                                                      prior_csf,...
                                                      prior_gm,...
                                                      prior_wm,...
                                                      prior_struct,...
                                                      parameters)
    

% ***************************************************************************************************
%  Find lesion candidates
%  Algorithm to detect and refill each of the WM lesion candidates in T1-w images. 
%
%  -inputs:
%   -> flair_img: flair input image 
%   -> seg_out: Initial tissue segmentation discrete 3 classes
%   -> pve_out: Initial tissue segmentation 5 classes
%   -> csf_prior: csf prior image
%   -> gm_prior: gm prior image
%   -> wm_prior: wm prior image
%   -> prior_struct: morphological strucutural atlas.
%   -> parameters: input parameters
%      + alpha (default 3)
%      + connectivity (default 4)
%
% - outputs:
%   lesion_candidates = mask containing all filled candidates
%   t1_refilled: T1-w image where lesion candidates have been refilled
%
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ***************************************************************************************************

    %********************************************************************************
    % input parameters
    %********************************************************************************

    alpha = parameters.alpha;                    % alpha multiplier for the threshold
    connectivity = parameters.connectivity;
    DEBUG = parameters.debug;
    

    
    % wm prior which is part of struct is neglected
    prior_wm(prior_struct==8) = 1;
    prior_wm(prior_struct == 3) = 0;
    prior_wm(prior_struct == 4) = 0;
    prior_wm(prior_struct == 5) = 0;
    prior_wm(prior_struct == 6) = 0;

    % .............................................................
    % 1) Extract candidates from FLAIR if available.
    %    -  Regions with less than 3 voxels are removed 
    % .............................................................
    
    flair_hyper_regions = extract_candidates(double(flair_img), seg_out, alpha);

    % filter FLAIR regions that are lower than 3 voxels. 
    if sum(nonzeros(flair_hyper_regions)) > 0
        hyper_map = flair_hyper_regions > 0;
        CC = bwconncomp(hyper_map, 6);
        filter = cellfun(@(x) numel(x)>3, CC.PixelIdxList);
        CC.PixelIdxList(filter == 0) = [];
        CC.NumObjects = sum(filter);
        hyper_labels_map = labelmatrix(CC);
        hyper_map = hyper_labels_map > 0;
    else
        hyper_map = flair_hyper_regions;
    end

    % .............................................................
    % 2) Select WM outliers and reassign to WM in T1-w image 
    %    -  Regions with less than 3 voxels are removed 
    % .............................................................
    
    gmwm_mask = pve_out > 3;
    gmwm_mask(hyper_map > 0)=0;
    n_mask = ones(size(gmwm_mask));
    n_mask(hyper_map> 0) = 0;

    lesion_candidates = zeros(size(pve_out));
    removed_candidates = zeros(size(pve_out));
    cortex = prior_struct== 2;
    for class=4:-1:2
        seg_mask = pve_out == class;
        current_regions = conn2d(seg_mask, connectivity);
        wm_mask = (pve_out == 5) .* not(seg_mask);
        
        for c=1:current_regions.NumObjects
            cv  = current_regions.PixelIdxList{c};
            
            % RULE 1: if numel < 0 --> not consider the current region
            if numel(cv)<4
                continue;
            end
            
            m_prior_wm = mean(prior_wm(cv));
            neighbors = compute_neighborhoods(cv, size(seg_mask),2,2);
            ratio_neigh = numel(nonzeros(wm_mask(neighbors))) / (numel(neighbors) - numel(cv));
            
            [rows, cols, slices] =ind2sub(size(seg_mask), cv);
            current_slice = [ rows, cols, slices];
            prev_slice = [rows, cols, (slices -1)];
            next_slice = [rows, cols, (slices +1)];
            vox_3d = [prev_slice; current_slice; next_slice];
            touch_cortex = sum(cortex(sub2ind(size(seg_mask), ...
                                                  vox_3d(:,1), vox_3d(:,2), min(max(vox_3d(:,3),1),size(seg_mask,3))))) > 0;
            
            % RULE 2: The current region has a high probability to belong to WM, it is connected to WM
            % and it is not touching  the cortex
            
            if (m_prior_wm >= 0.6) && ~touch_cortex
                if ratio_neigh <= 0.1
                    removed_candidates(cv) = 1;
                    continue;
                end
                pve_out(cv) = 5;
                lesion_candidates(cv) = 1;
                continue;
            end

       
            % RULE 3: The current region is touching a hyper-intense candidate in FLAIR. Flair candidates are 3D based to
            % remove possible outliers. If half of the neighbors are connecte to the GMWM and WM classes, the entire candidate
            % flair candidate is reassigned to WM. 
            
            touch_hyper = sum(hyper_map(cv)) > 0;
            if touch_hyper

                hyper_voxels = find(hyper_labels_map == mode(nonzeros(hyper_labels_map(cv))));
                neigh_hyper = compute_neighborhoods(hyper_voxels, size(seg_mask), 1,3);
                                                    
                num_neigh = numel(nonzeros(n_mask(neigh_hyper)));
                gmwm_neigh  = numel(nonzeros(gmwm_mask(neigh_hyper)));
                ratio_neigh_hyper = gmwm_neigh /num_neigh;

                if ratio_neigh_hyper < 0.5
                    removed_candidates(hyper_voxels) = 2;
                    continue;
                end
                pve_out(hyper_voxels) = 5;
                lesion_candidates(hyper_voxels) = 1;
                continue;
            end

        end
    end

    % *******************************************************
    % 3 refill the original t1_image using the same strategy
    % proposed in S. Valverde, A. Oliver, X. Lladó. "A white matter
    % lesion-filling approach to improve brain tissue volume measurements".
    % NeuroImage: Clinical, (6), pp 86-92, 2014
    % *******************************************************
    tic;
    T1_refilled = t1_image;
    CC = conn2d(lesion_candidates, connectivity);
    for c=1:CC.NumObjects
        clear t1_slice;
        clear wm_slice;
        current_voxels = CC.PixelIdxList{c};
        [~,~,slice] = ind2sub(size(lesion_candidates), current_voxels);
        t1_slice(:,:) = t1_image(:,:,slice(1));
        wm_slice(:,:) = wm_mask(:,:,slice(1));
        m_wm = mean(t1_slice(wm_slice == 1));
        s_wm = std(t1_slice(wm_slice == 1));
        T1_refilled(current_voxels) = normrnd(m_wm, s_wm/2,size(current_voxels));
    end
end
