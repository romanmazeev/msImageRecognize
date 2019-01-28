
function [conn] = conn2d(input_mask, connectivity)
% ***************************************************************************************************
%  Function to find the connected components in axial MRI. The function just visits each of the slices
%  and computes the connected component regions. 
%
%  -inputs:
%   -> input mask
%   -> connectivity
%
% - outputs:
%    -> conn structure containing:
%       + connectivity: connectivity type used for region search 
%       + ImageSize: Image size
%       + NumObjects: Number of connected regions
%       + PixelIdxList: A vector containing the voxels for each of the connected regions
%
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ***************************************************************************************************

   
   size_mask = size(input_mask);
   conn.Connectivity = connectivity;
   conn.ImageSize = size_mask;
   region = 1;
   for i=1:size_mask(3)
       m(:,:) = input_mask(:,:,i);
       cc = bwconncomp(m,connectivity);
       % transform indexed 2d -> 3d coordinates
       for c=1:cc.NumObjects
           current_voxels = cc.PixelIdxList{c};
           [row,col] = ind2sub(size(m), current_voxels);
           conn.PixelIdxList{region} = sub2ind(size_mask,row,col,repmat(i,size(row,1),1));
           region = region +1;
       end
   end
   conn.NumObjects = region -1;
end