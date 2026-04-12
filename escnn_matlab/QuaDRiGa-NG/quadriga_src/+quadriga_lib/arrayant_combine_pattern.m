% ARRAYANT_COMBINE_PATTERN
%    Calculate effective radiation patterns for array antennas
%    
% Description:
%    An array antenna consists of multiple individual elements. Each element occupies a specific position
%    relative to the array's phase-center, its local origin. Elements can also be inter-coupled,
%    represented by a coupling matrix. By integrating the element radiation patterns, their positions,
%    and the coupling weights, one can determine an effective radiation pattern observable by a receiver
%    in the antenna's far field. Leveraging these effective patterns is especially beneficial in antenna
%    design, beamforming applications such as in 5G systems, and in planning wireless communication
%    networks in complex environments like urban areas. This streamlined approach offers a significant
%    boost in computation speed when calculating MIMO channel coefficients, as it reduces the number of
%    necessary operations. The function arrayant_combine_pattern is designed to compute these effective
%    radiation patterns.
%    
% Usage:
%    
%    % Minimal example (input/output = struct)
%    arrayant_out = quadriga_lib.arrayant_combine_pattern(arrayant_in);
%    
%    % Optional inputs: freq, azimuth_grid, elevation_grid
%    arrayant_out = quadriga_lib.arrayant_combine_pattern(arrayant_in, freq, azimuth_grid, elevation_grid);
%    
%    % Separate outputs, struct input
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, freq, name] = quadriga_lib.arrayant_combine_pattern(arrayant_in);
%    
%    % Separate inputs
%    arrayant_out = quadriga_lib.arrayant_combine_pattern([], freq, azimuth_grid, elevation_grid, ...
%        e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, freq, name);
%    
% Examples:
%    
%    The following example creates a unified linear array of 4 dipoles, spaced at half-wavelength. The
%    elements are then coupled with each other (i.e., they receive the same signal). The effective pattern
%    is calculated using arrayant_combine_pattern.
%    
%    % Generate dipole pattern
%    ant = quadriga_lib.arrayant_generate('dipole');
%    
%    % Duplicate 4 times
%    ant.e_theta_re  = repmat(ant.e_theta_re, [1,1,4]);
%    ant.e_theta_im  = repmat(ant.e_theta_im, [1,1,4]);
%    ant.e_phi_re    = repmat(ant.e_phi_re, [1,1,4]);
%    ant.e_phi_im    = repmat(ant.e_phi_im, [1,1,4]);
%    ant.element_pos = repmat(ant.element_pos, [1,4]);
%    
%    % Set element positions and coupling matrix
%    ant.element_pos(2,:) = [ -0.75, -0.25, 0.25, 0.75];  % lambda, along y-axis
%    ant.coupling_re = [ 1 ; 1 ; 1 ; 1 ]/sqrt(4);
%    ant.coupling_im = [ 0 ; 0 ; 0 ; 0 ];
%    
%    % Calculate effective pattern
%    ant_c = quadriga_lib.arrayant_combine_pattern( ant );
%    
%    % Plot gain
%    plot( ant.azimuth_grid180/pi, [ 10log10( ant.e_theta_re(91,:,1).^2 ); 10log10( ant_c.e_theta_re(91,:).^2 ) ]);
%    axis([-180 180 -20 15]); ylabel('Gain (dBi)'); xlabel('Azimth angle (deg)'); legend('Dipole','Array')
%    
% Input Arguments:
%    - arrayant_in [1]
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
%      If an empty array is passed, array antenna data is provided as separate inputs (Inputs 5-15)
%    
%    - freq [2] (optional)
%      An alternative value for the center frequency. Overwrites the value given in arrayant_in. If
%      neither freq not arrayant_in["center_freq"] are given, an error is thrown.
%    
%    - azimuth_grid [3] (optional)
%      Alternative azimuth angles for the output in [rad], -pi to pi, sorted, Size: [n_azimuth_out],
%      If not given, arrayant_in.azimuth_grid is used instead.
%    
%    - elevation_grid [4] (optional)
%      Alternative elevation angles for the output in [rad], -pi/2 to pi/2, sorted, Size: [n_elevation_out],
%      If not given, arrayant_in.elevation_grid is used instead.
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
%      Can be returned as separate outputs.
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
    