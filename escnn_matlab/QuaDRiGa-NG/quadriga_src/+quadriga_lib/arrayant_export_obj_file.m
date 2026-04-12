% ARRAYANT_EXPORT_OBJ_FILE
%    Creates a Wavefront OBJ file for visualizing the shape of the antenna pattern
%    
% Usage:
%    
%    quadriga_lib.arrayant_export_obj_file( fn, arrayant, directivity_range, colormap, ...
%                    object_radius, icosphere_n_div, i_element );
%    
% Input Arguments:
%    - fn [1]
%      Filename of the OBJ file, string
%    
%    - arrayant [2]
%      Struct containing the arrayant data with the following fields:
%      e_theta_re     | e-theta field component, real part                    | Size: [n_elevation, n_azimuth, n_elements]
%      e_theta_im     | e-theta field component, imaginary part               | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_re       | e-phi field component, real part                      | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_im       | e-phi field component, imaginary part                 | Size: [n_elevation, n_azimuth, n_elements]
%      azimuth_grid   | Azimuth angles in [rad], -pi to pi, sorted            | Size: [n_azimuth]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation]
%      element_pos    | Antenna element (x,y,z) positions, optional           | Size: [3, n_elements]
%      name           | Name of the array antenna object, optional            | String
%    
%    - directivity_range [3]
%      Directivity range of the antenna pattern visualization in dB
%    
%    - colormap [4]
%      Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
%      'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', Optional, default = 'jet'
%    
%    - object_radius [5]
%      Radius in meters of the exported object
%    
%    - icosphere_n_div [6]
%      Map pattern to an Icosphere with given number of subdivisions
%    
%    - element [7]
%      Antenna element indices, 1-based, empty = export all
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
    