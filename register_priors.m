function [csf_prior, gm_prior, wm_prior, st] = register_priors(T1_path, options_register)
% ***************************************************************************************************
%  Register anatomical and morphological priors into the T1-w image space
%
%
%  NOTES:
%  - So far, registration is performed using the NIFTY REG package. The compiled binaries are called
%    via this function.
%  
%  -inputs:
%   -> t1_path: path to the T1-w image without extension
%   -> options_register:
%             options_register.info = Show information during registration (default 0)
%             options_register.force_reg = Force registration (default 0)
%  -outputs:
%      csf_prior = CSF prior probability atlas registered into the T1-w space
%      gm_prior = GM prior probability atlas registered into the T1-w space
%      wm_prior = WM prior probability atlas registered into the T1-w space
%      st = anatomical atlas registered into the T1-w space
%
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ***************************************************************************************************
 
    [image_folder, image_name] = fileparts(T1_path);
    [current_path,~] = fileparts(mfilename('fullpath'));
    
    force_registration = options_register.force_reg;
    display = options_register.info;
    
    
    moving = fullfile(current_path,'atlas','t1_atlas.nii');
    reference = T1_path;
   
    reg_aladin = [current_path,'/reg_aladin', ' -ref ', reference, ...
                ' -flo ', moving, ...
                ' -aff ', fullfile(image_folder,'.run','aff_transform.txt'), ...
                ' -res ', fullfile(image_folder,'.run','atlas_transf.nii')];

    reg_f3d = [current_path,'/reg_f3d', ' -ref ', reference, ...
                ' -flo ', moving, ...
                ' -aff ', fullfile(image_folder,'.run','aff_transform.txt'), ...
                ' -cpp ',  fullfile(image_folder,'.run','transf.cpp'), ...
                ' -res ', fullfile(image_folder,'.run','w_atlas_transf.nii')];


    % first we perform an affine registration followed by a deformable registration.
    if ~exist(fullfile(image_folder,'.run','atlas_transf.nii')) || force_registration
        t = system([reg_aladin,' -voff']);
        t = system([reg_f3d, ' -voff']);
    end
    if display
        disp('MSSEG: MNI template has been registered into the T1-w space');
    end

    % we use the transformation matrix to resample the anatomical and morphological atlases.
    reg_resample_aff_struct =  [current_path,'/reg_resample ',...
                    '-ref ', [reference,'.nii.gz'], ' ', ...
                        '-flo ', fullfile(current_path,'atlas','prior_struct.nii'), ' ',...
                     '-trans ', fullfile(image_folder,'.run','transf.cpp.nii'), ' ',...
                    '-res ',   fullfile(image_folder,'.run','r_struct.nii.gz'), ' ',...
                    ' -inter 0'];
    t = system([reg_resample_aff_struct, ' -voff']);


    reg_resample_aff_csf =  [current_path,'/reg_resample ',...
                    '-ref ', [reference,'.nii.gz'], ' ', ...
                     '-flo ', fullfile(current_path,'atlas','csf_prior.nii'), ' ',...
                    '-trans ', fullfile(image_folder,'.run','transf.cpp.nii'), ' ',...
                    '-res ',   fullfile(image_folder,'.run','r_csf_prior.nii'), ' ',...
                    ' -inter 1'];
    t = system([reg_resample_aff_csf, ' -voff']);

    reg_resample_aff_gm =  [current_path,'/reg_resample ',...
                    '-ref ', [reference,'.nii.gz'], ' ', ...
                     '-flo ', fullfile(current_path,'atlas','gm_prior.nii'), ' ',...
                    '-trans ', fullfile(image_folder,'.run','transf.cpp.nii'), ' ',...
                    '-res ',   fullfile(image_folder,'.run','r_gm_prior.nii'), ' ',...
                    ' -inter 1'];
    t = system([reg_resample_aff_gm, ' -voff']);


    reg_resample_aff_wm =  [current_path,'/reg_resample ',...
                     '-ref ', [reference,'.nii.gz'], ' ', ...
                     '-flo ', fullfile(current_path, 'atlas','wm_prior.nii'), ' ',...
                    '-trans ', fullfile(image_folder,'.run','transf.cpp.nii'), ' ',...3
                    '-res ',   fullfile(image_folder,'.run','r_wm_prior.nii'), ' ',...
                    ' -inter 1'];
    t = system([reg_resample_aff_wm, ' -voff']);


    % (2) load input_images and configure variables 
    gm_img = load_untouch_nii(fullfile(image_folder, '.run','r_gm_prior.nii'));
    wm_img = load_untouch_nii(fullfile(image_folder, '.run', 'r_wm_prior.nii'));
    csf_img = load_untouch_nii(fullfile(image_folder, '.run','r_csf_prior.nii'));
    st_img = load_compressed_nii(fullfile(image_folder, '.run','r_struct'));
    gm_prior = (double(gm_img.img));
    wm_prior = (double(wm_img.img));
    csf_prior = (double(csf_img.img));
    st = (double(st_img.img));

    % weird :( Don't know why registered atlas intensities starts at -32K~
    gm_prior = normalize_scan(gm_prior + abs(min(gm_prior(:))));
    wm_prior = normalize_scan(wm_prior + abs(min(wm_prior(:))));
    csf_prior = normalize_scan(csf_prior + abs(min(csf_prior(:))));
end

