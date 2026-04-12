% ARRAYANT_COPY_ELEMENT
%    Create copies of array antenna elements
%    
% Usage:
%    
%    arrayant_out = quadriga_lib.arrayant_copy_element(arrayant_in, source_element, dest_element);
%    
% Input Arguments:
%    - arrayant_in [1] (required)
%      Struct containing the arrayant data with the following fields:
%      e_theta_re     | e-theta field component, real part                    | Size: [n_elevation, n_azimuth, n_elements]
%      e_theta_im     | e-theta field component, imaginary part               | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_re       | e-phi field component, real part                      | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_im       | e-phi field component, imaginary part                 | Size: [n_elevation, n_azimuth, n_elements]
%      azimuth_grid   | Azimuth angles in [rad], -pi to pi, sorted            | Size: [n_azimuth]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation]
%      element_pos    | Antenna element (x,y,z) positions, optional           | Size: [3, n_elements]
%      coupling_re    | Coupling matrix, real part, optional                  | Size: [n_elements, n_ports]
%      coupling_im    | Coupling matrix, imaginary part, optional             | Size: [n_elements, n_ports]
%      center_freq    | Center frequency in [Hz], optional                    | Scalar
%      name           | Name of the array antenna object, optional            | String
%    
%    - source_element [2] (required)
%      Index of the source elements (1-based), scalar or vector
%    
%    - dest_element [3] (optional)
%      Index of the destination elements (1-based), either as a vector or as a scalar. If source_element
%      is also a vector, dest_element must have the same length.
%    
% Output Arguments:
%    - arrayant_out
%      Struct containing the arrayant data with the following fields:
%      e_theta_re     | e-theta field component, real part                    | Size: [n_elevation, n_azimuth, n_elements]
%      e_theta_im     | e-theta field component, imaginary part               | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_re       | e-phi field component, real part                      | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_im       | e-phi field component, imaginary part                 | Size: [n_elevation, n_azimuth, n_elements]
%      azimuth_grid   | Azimuth angles in [rad] -pi to pi, sorted             | Size: [n_azimuth]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation]
%      element_pos    | Antenna element (x,y,z) positions                     | Size: [3, n_elements]
%      coupling_re    | Coupling matrix, real part                            | Size: [n_elements, n_ports]
%      coupling_im    | Coupling matrix, imaginary part                       | Size: [n_elements, n_ports]
%      center_freq    | Center frequency in [Hz], default = 0.3 GHz           | Scalar
%      name           | Name of the array antenna object                      | String
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
    