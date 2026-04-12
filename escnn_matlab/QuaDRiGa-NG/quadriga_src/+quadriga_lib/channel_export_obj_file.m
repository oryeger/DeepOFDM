% CHANNEL_EXPORT_OBJ_FILE
%    Export path data to a Wavefront OBJ file for visualization in Blender
%    
% Description:
%    This function exports path data to a Wavefront OBJ file, which can be used for visualization in 3D
%    software such as Blender. It supports various colormaps for color-coding the paths based on their
%    gain values. In addition, the function allows you to control the maximum number of paths displayed,
%    set gain thresholds for color-coding and selection.
%    
% Usage:
%    
%    quadriga_lib.channel_export_obj_file( fn, max_no_paths, gain_max, gain_min, colormap, radius_max,
%        radius_min, n_edges, rx_position, tx_position, no_interact, interact_coord, center_freq,
%        coeff_re, coeff_im, i_snap )
%    
% Input Arguments:
%    - fn
%      Filename of the OBJ file, string, required
%    
%    - max_no_paths (optional)
%      Maximum number of paths to be shown, optional, default: 0 = export all above gain_min
%    
%    - gain_max (optional)
%      Maximum path gain in dB (only for color-coding), optional, default = -60.0
%    
%    - gain_min (optional)
%      Minimum path gain in dB (for color-coding and path selection), optional, default = -140.0
%    
%    - colormap (optional)
%      Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
%      'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', optional, default = 'jet'
%    
%    - radius_max (optional)
%      Maximum tube radius in meters, optional, default = 0.05
%    
%    - radius_min (optional)
%      Minimum tube radius in meters, optional, default = 0.01
%    
%    - n_edges (optional)
%      Number of vertices in the circle building the tube, must be >= 3, optional, default = 5
%    
%    - rx_position
%      Receiver positions, required, size [3, n_snap] or [3, 1]
%    
%    - tx_position
%      Transmitter positions, required, size [3, n_snap] or [3, 1]
%    
%    - no_interact
%      Number interaction points of paths with the environment, required, uint32, Size [n_path, n_snap]
%    
%    - interact_coord
%      Interaction coordinates, required, Size [3, max(sum(no_interact)), n_snap]
%    
%    - center_freq
%      Center frequency in [Hz], required, Size [n_snap, 1] or scalar
%    
%    - coeff_re
%      Channel coefficients, real part, Size: [ n_rx, n_tx, n_path, n_snap ]
%    
%    - coeff_im
%      Channel coefficients, imaginary part, Size: [ n_rx, n_tx, n_path, n_snap ]
%    
%    - i_snap (optional)
%      Snapshot indices, optional, 1-based, range [1 ... n_snap]
%    
% Output Argument:
%    This function does not return a value. It writes the OBJ file directly to disk.
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
    