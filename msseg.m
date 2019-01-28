function [seg_out, pve_out] = msseg(T1_path, brainmask_path, flair_path, options)
    
% ***************************************************************************************************
%  MSSEG tissue segmentation.
%  Main script.
%
%  -inputs:
%   -> t1_path: path to the T1-w image without extension
%   -> brainmask_path: path to the brainmask image without extension
%   -> flair_path: path to the FLAIR image without extension (OPTIONAL)
%   -> options:
%             options.prior = gamma parameter controlling the amount of atlas information used (default 0.025)
%             options.weighting = fuzzy factor exponent in FCM (default 2)
%             options.alpha =  parameter to regulate the minimum intensity considered in FLAIR candidates (default 3)
%             options.maxiter = Number of maximum iterations during energy minimization FCM (default 200)
%             options.num_neigh = Radius of the neighborhood used in spatial contraint (default 1)
%             options.dim = Dimension of the neighborhood (default 2)
%             options.term = Maximum error in energy minimization (default 1E-3)
%             options.gpu = Use GPU (default 0)
%             options.info = Show information during tissue segmentation (default 0)
%             options.debug = Save registered and intermediate files (default 0)
%
% - outputs:
%   seg_out = 3 class labelled segmentation (1) CSF, (2) GM and (3) WM.
%   pve_out = 5 class labelled segmentation (1) CSF, (2) CSFGM, (3) GM, (4) GMWM and (5) WM.
%
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ***************************************************************************************************


    % add nifti_tools to path
    [current_path,current_file] = fileparts(mfilename('fullpath'));
    addpath(fullfile(current_path, 'nifti_tools'));

    % parse OPTIONS
    options  = parse_options(options);
    
    % switch input_arguments. T1 / brainmask paths and options are mandatory. FLAIR is optional.
    switch nargin
      case 1
        error('Incorrect number of parameters');
      case 2
        error('Incorrect number of parameters');
      case 3
        flair_path = 'none';
        disp('MSSEG: No FLAIR image is provided')
      otherwise
        disp('MSSEG: Using FLAIR image to remove outliers');
    end

    % load data 
    [image_folder, image_name] = fileparts(T1_path);
    input_img = load_nifti(T1_path);
    input_image = double(input_img.img);
    brainmask_img = load_nifti((brainmask_path));
    brainmask = (double(brainmask_img.img)>0.1);
    % remove non-brain parts of the image
    input_image(brainmask == 0) = 0;

    
    % if flair is not passed as input, we use a mask with zeros instead.
    if strcmp(flair_path,'none')
        flair_scan = zeros(size(input_image));
    else
        flair_img = load_nifti((flair_path));
        flair_scan = flair_img.img;
        flair_scan(isnan(flair_scan)) = 0;
        flair_scan(brainmask==0) = 0;
    end

      
    % ********************************************************
    % (1) Register the input image into the standard MNI space
    % So far, we call the NIFTY REG procedure within MATLAB. 
    % ********************************************************

    % intermediate files and atlases are saved in a folder
    if ~exist(fullfile(image_folder,'.run'))
        mkdir(fullfile(image_folder,'.run'));
    end
 
    options_register.force_reg = ~options.debug;
    options_register.info = options.info;
    tic;
    [csf_prior, gm_prior, wm_prior, st] = register_priors(T1_path, options_register);
    t = toc;
    disp(['MSSEG: registering priors (', num2str(t),' secs.)']);

    % ********************************************************
    % (2) estimate tissue. Brain tissue is estimated into 5 different classes
    % using a FCM clustering approach with spatial constraints and morphological atlases.
    % ********************************************************
    
    c = options.c;
    out_name = options.name;
    display = options.info;
   
    t = tic;    
    [seg_out,pve_out] = estimate_tissue(input_image, brainmask, csf_prior, gm_prior, wm_prior, st, c, options);
    t = toc;
    disp(['MSSEG: estimate tissue (', num2str(t),' secs.)']);


    if options.debug == 1
        input_img.img = pve_out;
        save_nifti(input_img, fullfile(image_folder,  '.run', [image_name,'_',out_name,'_pve_it_1_debug']));    
        input_img.img = seg_out;
        save_nifti(input_img, fullfile(image_folder, '.run', [image_name,'_', out_name,'_seg_it_1_debug']));
    end

    
    % *****************************************************************************************
    % (3) estimate WM lesion candidates. Based on the initial tissue segmentation and
    % the morphological, anatomical atlases the WM outliers are detected and refilled into the
    % T1-w image. If the FLAIR image is available, a map of hyper-intense regions with possible
    % lesions is also computed.
    % *****************************************************************************************
    parameters.alpha = options.alpha;
    parameters.connectivity = 4;
    parameters.debug = options.debug;
    t = tic;
    [lesion_candidates, refilled_scan] = find_lesion_candidates(flair_scan,...
                                                      input_image, ...
                                                      seg_out,...
                                                      pve_out,...
                                                      csf_prior,...
                                                      gm_prior,...
                                                      wm_prior,...
                                                      st, ...
                                                      parameters);
    t = toc;
    disp(['MSSEG: find lesion candidates (', num2str(t),' secs.)']);

    % *************************************************************
    % (4) Reestimate tissue volume. The refilled T1-w image is
    % again tissue segmented. 
    % *************************************************************
    
    tic;
    [seg_out,pve_out] = estimate_tissue(refilled_scan, brainmask, csf_prior, gm_prior, wm_prior, st, c, options);
    t = toc;
    disp(['MSSEG: Re-estimate tissue  (', num2str(t),' secs.)']);
        
   
   % ******************************************* 
   % (5) save out the tissue masks
   % *******************************************

   if options.debug == 1
       input_img.img = lesion_candidates;
       save_nifti(input_img, fullfile(image_folder, '.run', [image_name,'_',out_name,'_refilled_candidates']));
   
       input_img.img = refilled_scan;
       save_nifti(input_img, fullfile(image_folder,'.run', [image_name,'_', out_name,'_t1_refilled']));
   end
   
   input_img.img = pve_out;
   save_nifti(input_img, fullfile(image_folder, '.run',[image_name,'_',out_name,'_pve']));
   
   input_img.img = seg_out;
   save_nifti(input_img, fullfile(image_folder, [image_name,'_', out_name,'_seg']));


   % remove all intermediate files
   if options.debug == 0
       rmdir(fullfile(image_folder,'.run'),'s');
   end
   
end



function options = parse_options(options)

% ********************************************************************************
% function to parse the mandatory options for the method
%
% ********************************************************************************

       % number of classes
       options.c = 5;
       % Beta parameter (afterwards it is updated)
       options.beta = 0.1;

       % gamma weighting controlling the amount of atlas information used
       if ~isfield(options,'prior')
           options.prior = 0.025;         
       end
       % Alpha parameter to regulate the minimum intensity considered in FLAIR candidates
       if ~isfield(options,'alpha')
           options.alpha = 3;         
       end
       % fuzzy factor exponent
       if ~isfield(options,'weighting')
           options.weighting = 2;        
       end
       % number of maximum iterations
       if ~isfield(options,'maxiter')
           options.maxiter = 200;         
       end
       % Number of neighbors in the spatial constraint
       if ~isfield(options,'num_neigh')
           options.num_neigh  = 1;         
       end
       % Number of dimensions of the neighborhood
       if ~isfield(options,'dim')
           options.dim  = 2;         
       end
       % Number of maximum iterations for the FCM clustering 
       if ~isfield(options,'term')
           options.term  = 1E-3;         
       end
       % Use GPU
       if ~isfield(options,'gpu')
           options.gpu = 0;         
       end
       % Show information 
       if ~isfield(options,'info')
           options.info = 0;         
       end
       % Debug mode: save also intermediate files
       if ~isfield(options,'debug')
           options.debug = 0;         
       end
       % option output name 
       if ~isfield(options,'name')
           options.name = 'MSSEG';         
       end

end


