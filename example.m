% ***************************************************************************************************
% Simple example showing the use of the MSSEG automated brain tissue segmentation
% method for MRI images containing lesions.
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ***************************************************************************************************

clc;
clear all;


% image location
t1_path = 'examples/T1';
brainmask_path = 'examples/brainMask';
flair_path = 'examples/T2_FLAIR';


% MSSEG OPTIONS
options.gpu = 0;          % use gpu
options.info = 1;         % Display info
options.debug = 1;        % save intermediate files.

% example using both T1 and FLAIR
%[segmentation, seg_pve] = msseg(t1_path, brainmask_path, flair_path, options);


% example using only T1 
[segmentation, seg_pve] = msseg(t1_path, brainmask_path, flair_path, options);

