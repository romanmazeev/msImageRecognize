
function [seg_out, pve_out] = estimate_tissue(input_image, brainmask, csf_prior, gm_prior, wm_prior, st, c, options)

% ***************************************************************************************************
%  Tissue estimation

%  -inputs:
%   -> t1: T1 image 
%   -> brainmask: brainmask image
%   -> csf_prior: csf prior image
%   -> gm_prior: gm prior image
%   -> wm_prior: wm prior image
%   -> st: morphological strucutural atlas.
%   -> c: number of classes (5 by default)
%   -> options:
%             options.prior = gamma parameter controlling the amount of atlas information used (default 0.025)
%             options.weighting = fuzzy factor exponent in FCM (default 2)
%             options.maxiter = Number of maximum iterations during energy minimization FCM (default 200)
%             options.num_neigh = Radius of the neighborhood used in spatial contraint (default 1)
%             options.dim = Dimension of the neighborhood (default 2)
%             options.term = Maximum error in energy minimization (default 1E-3)
%             options.gpu = Use GPU (default 0)
%             options.info = Show information during tissue segmentation (default 0)
%             options.debug = Save intermediate files (default 0)
%
% - outputs:
%   seg_out = 3 class labelled segmentation (1) CSF, (2) GM and (3) WM.
%   seg_out = 5 class labelled segmentation (1) CSF, (2) CSFGM, (3) GM, (4) GMWM and (5) WM.
%
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ***************************************************************************************************


    m = options.weighting;             % fuzziness exponent
    max_iter = options.maxiter;      % Max. iteration
    n_neigh = options.num_neigh;       % Number of neighbors used in the penalized function.
    neigh_dim = options.dim;     % Dimension of the neighborhood
    beta = options.beta;          % Beta parameter controling the strenght of the penalized function
    term_thr = options.term;      % Termination threshold
    use_gpu = options.gpu;       % Use GPU for compputation
    display = options.info;       % Display info or not
    gamma = options.prior;         % Gamma parameter to control the amount of previous information used
    out_name = options.name;

   
    
    if m <= 1,
        error('The weighting exponent should be greater than 1!');
    end
    
    % input Data
    input_image(~brainmask) = 0;            % nonskull input image
    Y = (find(input_image > 0));            % Indexed positions of each voxel
    X = (input_image(Y));                   % 1D reshaped input image
    [n,d] = size(X);                        % vector size
    W = (zeros(c,numel(X)));                % Weighting vectors for each class
    WP = (zeros(c,numel(X)));               % Weighting penalty vectors for each class
    WP_notmembers = (zeros(c, numel(X)));   % Weighting penalty for non class members
    error_v = (zeros(max_iter, 1));         % Array for termination measure values
    XC= (repmat(X',c,1));                   % precomputed duplicated input data vector X to increase the speed of the distance function
    seg_out = zeros(size(input_image));

    % create new atlases CSFGM and GMWM from the original three CSF,GM and WM atlases. 
    min_prob = 0.4;
    pv1mask = ((csf_prior > min_prob) & (gm_prior > min_prob));
    csfgm_prior = ((csf_prior + gm_prior) ./ 2) .* pv1mask;
    pv2mask = ((gm_prior > min_prob) & (wm_prior > min_prob));
    gmwm_prior = ((gm_prior + wm_prior) ./ 2) .* pv2mask;
    csfgm_prior(csfgm_prior>0) = csfgm_prior(csfgm_prior>0) + min_prob;
    gmwm_prior(gmwm_prior>0) = gmwm_prior(gmwm_prior>0) + min_prob;
    gm_prior(pv1mask) = 0;
    gm_prior(pv2mask)= 0;
    wm_prior(pv2mask)= 0;


    PRIOR(1,:) = csf_prior(Y)';       
    PRIOR(3,:) = gm_prior(Y)';
    PRIOR(5,:) = wm_prior(Y)';
    PRIOR(2,:) = csfgm_prior(Y)';
    PRIOR(4,:) = gmwm_prior(Y)';

    
    % estimate beta automatically
    beta = estimate_beta(input_image, wm_prior);
    if display
        disp(['MSSEG: Automatic beta parametrization: ', num2str(beta)]);
    end

    %beta = 0.5;

    %W = PRIOR;
    if use_gpu
        % pass to CUDA arrays
        Y = gpuArray(double(Y));
        X = gpuArray((X));
        W = gpuArray(double(W));
        WP = gpuArray(double(WP));
        WP_notmembers = gpuArray(double(WP_notmembers));
        WPR = WP;
        WPR_NM = WP_notmembers;
        XC = gpuArray(double(XC));
        PRIOR = gpuArray(double(PRIOR));
        seg_out = gpuArray(seg_out);
        if display
            disp(['MSSEG: GPU based computation. Data have been transformed to GPUArrays']);
        end
    end

    % (3). Find the initial cluster centers. Based on the histogram. By default,
    % the number of bins is set to 512.

    C = initialize_centers(X,PRIOR);
    
    if display
        disp(['MSSEG: Initial centers: ', num2str(C')]);
    end
  
    % Beta parameter should be adjusted by cross-validation of the test set.
    % So far, the approximate range in the image appears to work well also.

    if display
       disp(['MSSEG: Initial Beta: ', num2str(beta)]);
    end

    % (4). Precompute neighbor voxel position indices.
    if display
        disp('MSSEG: Precomputing neighbor voxel positions.....');
    end
    neighbor_voxels = (compute_neighborhoods(Y, size(input_image), n_neigh, neigh_dim));
    neighbor_voxels2 = (compute_neighborhoods(Y, size(input_image), 2, neigh_dim));


    current_weight = (zeros(size(input_image)));
    class_vector = 1:c;
    not_member_class = repmat(class_vector,c,1) ~= repmat(class_vector',1,c);
    if use_gpu
        neighbor_voxels = gpuArray(neighbor_voxels);
        current_weight = gpuArray(current_weight);
    end


    % (5). Minimize the objective function
    if display
        disp('MSSEG: Minimizing the objective function......');
    end

    for class=1:c
        current_weight(Y) = PRIOR(class,:);
        WPR(class,:) = sum(current_weight(neighbor_voxels2),2);
    end

    % for each voxel, sum non-class member neighbors.
    for class=1:c
        not_members = class_vector(not_member_class(class,:));
        WPR_NM(class,:) = sum(WPR(not_members,:));
    end


    for i = 1:max_iter,

        % (5.1) penalty function: for each voxel and class compute the sum
        % of the weights of their neighbors
        for class=1:c
            current_weight(Y) = W(class,:);
            WP(class,:) = sum(current_weight(neighbor_voxels),2);
        end

        % for each voxel, sum non-class member neighbors.
        for class=1:c
             not_members = class_vector(not_member_class(class,:));
             WP_notmembers(class,:) = sum(WP(not_members,:));
        end
        
        % (5.2) new weights W
        dist = abs(repmat(C,1,n) - XC);
        denom = (dist + (beta.*(WP_notmembers.^m))+ (gamma.*(WPR_NM.^m))).^(-2/(m-1));
        W = denom./ (ones(c, 1)*(sum(denom)));

        % Correct the situation of "singularity" (one of the data points is
        % exactly the same as one of the cluster centers).
        if use_gpu
            si = gather(find (denom == Inf));
        else
            si = find (denom == Inf);
        end
        
        if si > 0
            W(si) = 1;
            if display
                disp('singularity');
            end
        end

        % Check constraint
        tmp = find ((sum (W) - ones (1, n)) > 0.0001);
        if (size(tmp,2) ~= 0)
            disp('MSSEG:  Warning: Constraint for U is not hold.');
        end

        % (5.3) calculate new centers C and update the error
        C_old = C;
        mf = W.^m;
        C = mf*X./((ones(d, 1)*sum(mf'))');

        error_v(i) = norm (C - C_old, 1);
        if display
            disp(['MSSEG: Iteration: ', num2str(i), ' Estimated error: ', num2str(error_v(i))]);
        end
        % check termination condition
        if error_v(i) <= term_thr, break; end,


    end

    iter_n = i;	% Actual number of iterations
    error_v(iter_n+1:max_iter) = [];

    % (8). compute binary segmentation. Final classification is performed following two
    % steps: First, each voxel is assigned to one of the 5 classes taking the maximum
    % probability. Then, partial volumes are reassigned into CSF, GM and WM depending
    % of their distance to ventricles, cortical GM, or local intensity of their neighbors.

    [C, index] = sort(C);
    [~, segmentation] = max(W(index,:));
    % maintain also tissue classification with 5 classes.

    seg_out(Y) = segmentation;
    % gather tissue segmentation from the GPU back to the CPU
    if use_gpu
        seg_out = gather(seg_out);
    end
    pve_out = seg_out;
       
    
    % 8.1 partial volume reassign.    
    % partial volume regions touching the ventricles are reassigned to CSF
    regions_2 = conn2d(seg_out ==2,4);
    for c=1:regions_2.NumObjects
        cv = regions_2.PixelIdxList{c};
        if sum((st(cv) == 1) + seg_out(cv)) > 0
            seg_out(cv) = 1;
        end
    end

    
    % partial volume regions touching the cortical GM are reassigned to GM
    % partial volume regions touching the pons and brainstem are reassigned to WM

    regions_4 = conn2d(seg_out == 4,4);
    for c=1:regions_4.NumObjects
        cv = regions_4.PixelIdxList{c};
        if sum((st(cv) == 2) + seg_out(cv)) > 0
            seg_out(cv) = 3; 
        end
        if sum((st(cv) == 4) + seg_out(cv)) > 0
            seg_out(cv) = 3; 
        end
        
        if sum((st(cv) == 8) & seg_out(cv)) > (numel(cv) /2)
            seg_out(cv) = 5; 
        end
    end
    

    PCG = find(seg_out == 2);
    PGW = find(seg_out == 4);
    partial_vols = [PCG; PGW];

    rad = 6;
    neigh_region_pv = compute_neighborhoods(partial_vols, size(seg_out), rad,2);


    % mean intensity of the neighboring voxels to CSF(1), GM(3) and WM(5);
    s1 = input_image(neigh_region_pv).*(seg_out(neigh_region_pv) == 1);
    region_mean_csf = sum(s1.*(s1>0),2) ./ sum(s1>0,2);
    s2 = input_image(neigh_region_pv).*(seg_out(neigh_region_pv) == 3);
    region_mean_gm = sum(s2.*(s2>0),2) ./ sum(s2>0,2);
    s3 = input_image(neigh_region_pv).*(seg_out(neigh_region_pv) == 5);
    region_mean_wm = sum(s3.*(s3>0),2) ./ sum(s3>0,2);


    % difference of mean intensity of neighboring voxels with respect to each of the
    % partial volume voxels.
    mean_regions = [region_mean_csf region_mean_gm region_mean_wm];
    dif_for_classes = abs(mean_regions - repmat(input_image(partial_vols),1,3));
    [~, pve_segmentation] = min(dif_for_classes');
    %seg_out(partial_vols) = pve_segmentation;


    % Pure CSF, GM and WM are also reassigned to a 3 class segmentation.
    seg_out(seg_out == 1) = 1;
    seg_out(seg_out == 3) = 2;
    seg_out(seg_out == 5) = 3;
    seg_out(partial_vols) = pve_segmentation;
end



function [initial_centers] = initialize_centers(X, PRIOR)
% ************************************************************************
% initialize the cluster centers based on the mean intensity according to
% the prior probability atlases
% ************************************************************************

    th_prob = PRIOR > 0.5;
    
    % not so elegant, but cannot run nonzeros function with
    % gpuarrays. 5 classes is not so much for a for loop :(
    for c=1:size(th_prob,1)
          initial_centers(c,1) = mean(X(th_prob(c,:)==1));
    end
    
end



function [beta] = estimate_beta(process_img, prior_wm)
% ************************************************************************
% Compute the beta parameter using a spline function. The spline function
% is based on the interpolation after running several BRAINWEB segmentations
% and manually tuned the beta parameter on them.
% ************************************************************************
    for i=1:size(process_img,3)
        slice(:,:) = process_img(:,:,i);
        slice = slice .* (slice > 0);
        n(i) = estimate_noise(slice);
    end
    n  = mean(nonzeros(n));

    X = [0.1,1,4,7,9];
    Y = [0.05, 0.1, 0.5, 1, 2];

    beta = spline(X,Y, n);
    beta = round(beta * 10) / 10;

end


function [Sigma] = estimate_noise(I)
% **********************************************************************
%  Noise is estimated using the method proposed by J. Immerkaer,
%  “Fast Noise Variance Estimation", 1996.
% **********************************************************************
   
    [H W]=size(I);
    I=double(I);

    % compute sum of absolute values of Laplacian
    M=[1 -2 1; -2 4 -2; 1 -2 1];
    Sigma=sum(sum(abs(conv2(I, M))));

    % scale sigma with proposed coefficients
    Sigma=Sigma*sqrt(0.5*pi)./(6*(W-2)*(H-2));
end
