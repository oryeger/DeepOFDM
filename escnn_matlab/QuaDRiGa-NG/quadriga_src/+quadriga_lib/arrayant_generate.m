% ARRAYANT_GENERATE
%    Generates predefined array antenna models
%    
% Description:
%    This functions can be used to generate a variety of pre-defined array antenna models, including 3GPP
%    array antennas used for 5G-NR simulations. The first argument is the array type. The following input
%    arguments are then specific to this type.
%    
% Usage:
%    
%    % Simple antenna models, output as struct
%    ant = quadriga_lib.arrayant_generate('omni', res);               % Isotropic radiator, v-pol
%    ant = quadriga_lib.arrayant_generate('dipole', res);             % Short dipole, v-pol
%    ant = quadriga_lib.arrayant_generate('half-wave-dipole', res);   % Half-wave dipole, v-pol
%    ant = quadriga_lib.arrayant_generate('xpol', res);               % Cross-polarized isotropic radiator
%    
%    % An antenna with a custom 3dB beam with (in degree)
%    ant = quadriga_lib.arrayant_generate('custom', res, freq, az_3dB, el_3db, rear_gain_lin);
%    
%    % A unified linear array with isotropic patterns
%    ant = quadriga_lib.arrayant_generate('ula', res, freq, [], [], [], 1, N, [], [], spacing);
%    
%    % Antenna model for the 3GPP-NR channel model with 3GPP default pattern
%    ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [],
%                                         M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh);
%    
%    % Antenna model for the 3GPP-NR channel model with a custom beam width
%    ant = quadriga_lib.arrayant_generate('3GPP', res, freq, az_3dB, el_3db, rear_gain_lin,
%                                         M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh);
%    
%    % Antenna model for the 3GPP-NR channel model with a custom antenna pattern
%    ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [],
%                                         M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh, pattern);
%    
%    % Multi-beam antenna (single beam serving multiple directions)
%    ant = quadriga_lib.arrayant_generate('multibeam', res, freq, az_3dB, el_3db, rear_gain_lin,
%                                         M, N, pol, beam_angles, spacing);
%    
%    % Multi-beam antenna (one beam per direction)
%    ant = quadriga_lib.arrayant_generate('multibeam_sep', res, freq, az_3dB, el_3db, rear_gain_lin,
%                                         M, N, pol, beam_angles, spacing);
%    
%    % Optional for all types: output as separate variables, (must have exactly 11 outputs)
%    [e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
%        coupling_re, coupling_im, freq, name] = quadriga_lib.arrayant_generate( ... );
%    
% Input Arguments:
%    - type [1]
%      Antenna model type, string
%    
%    - res [2]
%      Pattern resolution in [deg], scalar, default = 1 deg
%    
%    - freq [3]
%      The center frequency in [Hz], scalar, default = 299792458 Hz
%    
% Input arguments for type 'custom' and '3GPP' (custom beam width):
%    - az_3dB [4]
%      3dB beam width in azimuth direction in [deg], scalar,
%      default for custom = 90 deg, default for 3gpp = 67 deg
%    
%    - el_3db [5]
%      3dB beam width in elevation direction in [deg], scalar,
%      default for custom = 90 deg, default for 3gpp = 67 deg
%    
%    - rear_gain_lin [6]
%      Isotropic gain (linear scale) at the back of the antenna, scalar, default = 0.0
%    
% Input arguments for type '3GPP':
%    - M [7]
%      Number of vertically stacked elements, scalar, default = 1
%    
%    - N [8]
%      Number of horizontally stacked elements, scalar, default = 1
%    
%    - pol [9]
%      Polarization indicator to be applied for each of the M elements:
%      pol = 1 | vertical polarization (default value)
%      pol = 2 | H/V polarized elements, results in 2NM elements
%      pol = 3 | +/-45° polarized elements, results in 2NM elements
%      pol = 4 | vertical polarization, combines elements in vertical direction, results in N elements
%      pol = 5 | H/V polarization, combines elements in vertical direction, results in 2N elements
%      pol = 6 | +/-45° polarization, combines elements in vertical direction, results in 2N elements
%      Polarization indicator is ignored when a custom pattern is provided.
%    
%    - tilt [10]
%      The electric downtilt angle in [deg], Only relevant for pol = 4/5/6, scalar, default = 0
%    
%    - spacing [11]
%      Element spacing in [λ], scalar, default = 0.5
%    
%    - Mg [12]
%      Number of nested panels in a column, scalar, default = 1
%    
%    - Ng [13]
%      Number of nested panels in a row, scalar, default = 1
%    
%    - dgv [14]
%      Panel spacing in vertical direction in [λ], scalar, default = 0.5
%    
%    - dgh [15]
%      Panel spacing in horizontal direction in [λ], scalar, default = 0.5
%    
%    - pattern [16]
%      Struct containing a custom pattern (default = empty) with at least the following fields:
%      e_theta_re_c     | Real part of e-theta field component             | Size: [n_elevation, n_azimuth, n_elements_c]
%      e_theta_im_c     | Imaginary part of e-theta field component        | Size: [n_elevation, n_azimuth, n_elements_c]
%      e_phi_re_c       | Real part of e-phi field component               | Size: [n_elevation, n_azimuth, n_elements_c]
%      e_phi_im_c       | Imaginary part of e-phi field component          | Size: [n_elevation, n_azimuth, n_elements_c]
%      azimuth_grid_c   | Azimuth angles in [rad] -pi to pi, sorted        | Size: [n_azimuth]
%      elevation_grid_c | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: [n_elevation]
%    
%      If custom pattern data is not provided, the pattern is generated internally (either with a custom
%      beam width if az_3dB and el_3db are given or using the default 3GPP pattern).
%    
% Input arguments for type 'multibeam' and 'multibeam_sep':
%    - beam_angles [10]
%      Matrix containing the beam steering angles in [deg] for the multi-beam antenna, size = [3, n_beams],
%      Rows are: [azimuth_deg, elevation_deg, weight].
%    
% Output Arguments:
%    - ant
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
%      center_freq    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
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
    