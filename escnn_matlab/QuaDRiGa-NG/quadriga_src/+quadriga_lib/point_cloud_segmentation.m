% POINT_CLOUD_SEGMENTATION
%    Rearranges elements of a point cloud into smaller sub-clouds
%    
% Description:
%    This function processes a large 3D point cloud by clustering closely spaced points and recursively 
%    partitioning it into smaller sub-clouds, each below a specified size threshold. It minimizes the 
%    axis-aligned bounding box of each sub-cloud while striving to maintain a target number of points 
%    per cluster.
%    
%    Sub-clouds are aligned to a specified SIMD vector size (e.g., for AVX or CUDA), with padding applied 
%    as needed. The function outputs a reorganized version of the input points (pointsR), where points 
%    are grouped by sub-cloud, and provides forward and reverse index maps to track the reordering. This 
%    organization is particularly useful for optimizing spatial processing tasks such as bounding volume 
%    hierarchies or GPU batch execution.
%    
% Usage:
%    
%    [ points_out, sub_cloud_index, forward_index, reverse_index ] = ...
%        quadriga_lib.point_cloud_segmentation( points_in, target_size, vec_size );
%    
% Input Arguments:
%    
%    - points_in
%      Points in 3D-Cartesian space; Size: [ n_points_in, 3 ]
%    
%    - target_size (optional)
%      The target number of elements of each sub-cloud. Default value = 1024. For best performance, the
%      value should be around 10  sgrt( n_points_in )
%    
%    - vec_size (optional)
%      Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1.
%      For values > 1,the number of rows for each sub-cloud in the output is increased to a multiple
%      of vec_size. For padding, zero-sized triangles are placed at the center of the AABB of
%      the corresponding sub-cloud.
%    
% Output Arguments:
%    
%    - points_out
%      Points in 3D-Cartesian space; singe or double precision;  Size: [ n_points_out, 3 ]
%    
%    - sub_cloud_index
%      Start indices of the sub-clouds in 0-based notation. Type: uint32; Vector of length [ n_sub_cloud ]
%    
%    - forward_index
%      Indices for mapping elements of "points_in" to "points_out"; 1-based;
%      Length: [ n_points_out ]; For vec_size > 1, the added elements not contained in the input
%      are indicated by zeros.
%    
%    - reverse_index
%      Indices for mapping elements of "points_out" to "points_in"; 1-based; Length: [ n_points_in ]
%
%
% quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% All rights reserved.
%
% e-mail: info@quadriga-lib.org
%
% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
% in compliance with the License. You may obtain a copy of the License at
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software distributed under the License
% is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
% or implied. See the License for the specific language governing permissions and limitations under
% the License.
    