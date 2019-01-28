function [seg_out, pve_out] = tissue_segmentation(T1_path, brainmask_path, flair_path, options)
    
    % ***************************************************************************************************
    % MSSEG tissue segmentation.
    % 
    %
    %
    % svalverde@eia.udg.edu 2015
    % ***************************************************************************************************


    % switch input_arguments 
    
    switch nargin
      case 1
        error('Incorrect number of parameters');
      case 2
        % T1 is already skull-stripped. No FLAIR image is provided
        brainmask_path = T1_path;
        flair_path = 'none';
      case 3
        % No FLAIR image is provided
        flair_path = 'none';
        disp('No FLAIR image is provided')
      otherwise
        disp('Using FLAIR image to remove outliers');
    end

    
    % load data ***********************************
    [image_folder, image_name] = fileparts(T1_path);

    brainmask_img = load_compressed_nii((brainmask_path));
    %brainmask_img = load_untouch_nii([brainmask_path,'.nii']);
    brainmask = (double(brainmask_img.img)>0.1);
    
    input_img = load_compressed_nii(T1_path);
    input_image = double(input_img.img);
    input_image(brainmask == 0) = 0;
    
    % if flair is not passed as input, we use a mask with zeros instead.
    if strcmp(flair_path,'none')
        flair_scan = zeros(size(input_image));
    else

        flair_img = load_compressed_nii((flair_path));
        flair_scan = flair_img.img;
        flair_scan(isnan(flair_scan)) = 0;
        flair_scan(brainmask==0) = 0;
    end
    

    % --------------------------------------------------------------------------------
    % p i p e l i n e
    % --------------------------------------------------------------------------------


    % ********************************************************
    % (1) Register the input image into the standard MNI space
    % ********************************************************

    options_register.force_reg = options.force_reg;
    options_register.info = options.info;
    
    tic;
    [csf_prior, gm_prior, wm_prior, st] = register_priors(T1_path, options_register);
    t = toc;
    disp(['------- registering priors (', num2str(t),')']);

    % ********************************************************
    % (2) estimate tissue 
    % ********************************************************
    
    c = options.c;
    out_name = options.name;
    display = options.info;
   
    t = tic;
    
    [seg_out,pve_out] = estimate_tissue(input_image, brainmask, csf_prior, gm_prior, wm_prior, st, c, options);
    t = toc;
    disp(['------- estimate tissue (', num2str(t),')']);

    
    input_img.img = pve_out;
    save_compressed_nii(input_img, fullfile(image_folder,  '.run', [image_name,'_',out_name,'_pve_it_1']));
    
    input_img.img = seg_out;
    save_compressed_nii(input_img, fullfile(image_folder, '.run', [image_name,'_', out_name,'_seg_it_1']));
   

    % ********************************************************
    % (3) estimate WM lesion candidates
    % ********************************************************
    parameters.alpha = options.alpha;
    parameters.prior_probability = 0.1;
    parameters.connectivity = 4;
    parameters.minsize = 2;
    parameters.debug = 1;
    parameters.image_folder = image_folder;
    t = tic;
    [lesion_candidates, refilled_scan] = find_lesion_candidates(input_img, pve_out, ...
                                                      csf_prior,...
                                                      gm_prior,...
                                                      wm_prior,...
                                                      st, ...
                                                      flair_scan, ...
                                                      input_image, ...
                                                      seg_out,...
                                                      parameters);
    t = toc;
    disp(['------- find lesion candidates (', num2str(t),')']);

    % *******************************************************
    % (4) Reestimate tissue volume
    % *******************************************************
    
    tic;
    [seg_out,pve_out] = estimate_tissue(refilled_scan, brainmask, csf_prior, gm_prior, wm_prior, st, c, options);
    t = toc;
    disp(['------- re-estimate tissue  (', num2str(t),')']);
        
   
   % ******************************************* 
   % (5) save_out the masks
   % *******************************************

   input_img.img = lesion_candidates;
   save_compressed_nii(input_img, fullfile(image_folder, '.run', [image_name,'_',out_name,'_lesion_candidates']));
   
   
   input_img.img = refilled_scan;
   save_compressed_nii(input_img, fullfile(image_folder,'.run', [image_name,'_', out_name,'_t1_refilled']));
  
   input_img.img = pve_out;
   save_compressed_nii(input_img, fullfile(image_folder, '.run',[image_name,'_',out_name,'_pve']));
   
   input_img.img = seg_out;
   save_compressed_nii(input_img, fullfile(image_folder, [image_name,'_', out_name,'_seg']));
   
end






