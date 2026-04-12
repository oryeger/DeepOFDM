% ARRAYANT_QDANT_WRITE
%    Writes array antenna data to QDANT files
%    
% Description:
%    The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna pattern
%    data in XML. This function writes pattern data to the specified file.
%    
% Usage:
%    
%    % Arrayant as struct
%    id_in_file = quadriga_lib.arrayant_qdant_write( fn, arrayant, id, layout);
%    
%    % Arrayant as separate inputs
%    id_in_file = quadriga_lib.arrayant_qdant_write( fn, [], id, layout, e_theta_re, e_theta_im, e_phi_re,
%        e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name);
%    
% Caveat:
%    - Multiple array antennas can be stored in the same file using the id parameter.
%    - If writing to an exisiting file without specifying an id, the data gests appended at the end.
%      The output id_in_file identifies the location inside the file.
%    - An optional storage layout can be provided to organize data inside the file.
%    
% Input Arguments:
%    - fn [1]
%      Filename of the QDANT file, string
%    
%    - arrayant [2] (optional)
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
%      center_freq    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
%      name           | Name of the array antenna object, optional            | String
%      If an empty array is passed, array antenna data is provided as separate inputs (Inputs 5-15)
%    
%    - id [3] (optional)
%      ID of the antenna to be written to the file, optional, Default: Max-ID in existing file + 1
%    
%    - layout [4] (optional)
%      Layout of multiple array antennas. Must only contain element ids that are present in the file. optional
%    
% Output Argument:
%    - id_in_file
%      ID of the antenna in the file after writing
%    
% See also:
%    - [[arrayant_qdant_read]] (for reading QDANT data)
%    - QuaDRiGa Array Antenna Exchange Format  (QDANT)
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
    