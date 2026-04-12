% GET_CHANNELS_IRS
%    Calculate channel coefficients for intelligent reflective surfaces (IRS)
%    
% Description:
%    - Calculates MIMO channel coefficients and delays for IRS-assisted communication using two channel segments:
%      1. TX → IRS; 2. IRS → RX
%    - The IRS is modeled as a passive antenna array with phase shifts defined via its coupling matrix.
%    - IRS codebook entries can be selected via a port index (i_irs).
%    - Supports combining paths from both segments to form n_path_irs valid output paths, subject to a gain threshold.
%    - Optional second IRS array allows different antenna behavior for TX-IRS and IRS-RX directions.
%    
% Usage:
%    
%    [ coeff_re, coeff_im, delays, active_path, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_irs( ...
%        ant_tx, ant_rx, ant_irs, ...
%        fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
%        fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
%        tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, ...
%        i_irs, threshold_dB, center_freq, use_absolute_delays, active_path,  ant_irs_2 );
% Input Arguments:
%    - ant_tx [1] (required)
%      Struct containing the transmit (TX) arrayant data with the following fields:
%      e_theta_re     | e-theta field component, real part                    | Size: [n_elevation_tx, n_azimuth_tx, n_elements_tx]
%      e_theta_im     | e-theta field component, imaginary part               | Size: [n_elevation_tx, n_azimuth_tx, n_elements_tx]
%      e_phi_re       | e-phi field component, real part                      | Size: [n_elevation_tx, n_azimuth_tx, n_elements_tx]
%      e_phi_im       | e-phi field component, imaginary part                 | Size: [n_elevation_tx, n_azimuth_tx, n_elements_tx]
%      azimuth_grid   | Azimuth angles in [rad], -pi to pi, sorted            | Size: [n_azimuth_tx]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation_tx]
%      element_pos    | Antenna element (x,y,z) positions, optional           | Size: [3, n_elements_tx]
%      coupling_re    | Coupling matrix, real part, optional                  | Size: [n_elements_tx, n_ports_tx]
%      coupling_im    | Coupling matrix, imaginary part, optional             | Size: [n_elements_tx, n_ports_tx]
%    
%    - ant_rx [2] (required)
%      Struct containing the receive (RX) arrayant data with the following fields:
%      e_theta_re     | e-theta field component, real part                    | Size: [n_elevation_rx, n_azimuth_rx, n_elements_rx]
%      e_theta_im     | e-theta field component, imaginary part               | Size: [n_elevation_rx, n_azimuth_rx, n_elements_rx]
%      e_phi_re       | e-phi field component, real part                      | Size: [n_elevation_rx, n_azimuth_rx, n_elements_rx]
%      e_phi_im       | e-phi field component, imaginary part                 | Size: [n_elevation_rx, n_azimuth_rx, n_elements_rx]
%      azimuth_grid   | Azimuth angles in [rad], -pi to pi, sorted            | Size: [n_azimuth_rx]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation_rx]
%      element_pos    | Antenna element (x,y,z) positions, optional           | Size: [3, n_elements_rx]
%      coupling_re    | Coupling matrix, real part, optional                  | Size: [n_elements_rx, n_ports_rx]
%      coupling_im    | Coupling matrix, imaginary part, optional             | Size: [n_elements_rx, n_ports_rx]
%    
%    - ant_irs [3] (required)
%      Struct containing the intelligent reflective surface (IRS) model:
%      e_theta_re     | e-theta field component, real part                    | Size: [n_elevation_irs, n_azimuth_irs, n_elements_irs]
%      e_theta_im     | e-theta field component, imaginary part               | Size: [n_elevation_irs, n_azimuth_irs, n_elements_irs]
%      e_phi_re       | e-phi field component, real part                      | Size: [n_elevation_irs, n_azimuth_irs, n_elements_irs]
%      e_phi_im       | e-phi field component, imaginary part                 | Size: [n_elevation_irs, n_azimuth_irs, n_elements_irs]
%      azimuth_grid   | Azimuth angles in [rad], -pi to pi, sorted            | Size: [n_azimuth_irs]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation_irs]
%      element_pos    | Antenna element (x,y,z) positions, optional           | Size: [3, n_elements_irs]
%      coupling_re    | Coupling matrix, real part, optional                  | Size: [n_elements_irs, n_ports_irs]
%      coupling_im    | Coupling matrix, imaginary part, optional             | Size: [n_elements_irs, n_ports_irs]
%    
%    - fbs_pos_1 [4] (required)
%      First-bounce scatterer positions of TX → IRS paths, Size: [ 3, n_path_1 ].
%    
%    - lbs_pos_1 [5] (required)
%      Last-bounce scatterer positions of TX → IRS paths, Size [3, n_path_1].
%    
%    - path_gain_1 [6] (required)
%      Path gains (linear) for TX → IRS paths, Length n_path_1.
%    
%    - path_length_1 [7] (required)
%      Path lengths for TX → IRS paths, Length n_path_1.
%    
%    - M_1 [8] (required)
%      Polarization transfer matrix for TX → IRS paths, Size [8, n_path_1].
%    
%    - fbs_pos_2 [9] (required)
%      First-bounce scatterer positions of IRS → RX paths, Size: [ 3, n_path_2 ]
%    
%    - lbs_pos_2 [10] (required)
%      Last-bounce scatterer positions of IRS → RX paths, Size [3, n_path_2]
%    
%    - path_gain_2 [11] (required)
%      Path gains (linear) for IRS → RX paths, Length n_path_2.
%    
%    - path_length_2 [12] (required)
%      Path lengths for IRS → RX paths, Length n_path_2.
%    
%    - M_2 [13] (required)
%      Polarization transfer matrix for IRS → RX paths, Size [8, n_path_2].
%    
%    - tx_pos [14] (required)
%      Transmitter position in 3D Cartesian coordinates, Size: [3,1] or [1,3]
%    
%    - tx_orientation [15] (required)
%      3-element vector describing the orientation of the transmit antenna in Euler angles (bank, tilt, heading),
%      Size: [3,1] or [1,3]
%    
%    - rx_pos [16] (required)
%      Receiver position in 3D Cartesian coordinates, Size: [3,1] or [1,3]
%    
%    - rx_orientation [17] (required)
%      3-element vector describing the orientation of the receive antenna, Size: [3,1] or [1,3]
%    
%    - irs_pos [18] (required)
%      IRS position in 3D Cartesian coordinates, Size: [3,1] or [1,3]
%    
%    - irs_orientation [19] (required)
%      3-element (Euler) vector in Radians describing the orientation of the IRS, Size: [3,1] or [1,3]
%    
%    - i_irs [20] (optional)
%      Index of IRS codebook entry (port number), Scalar,  Default: 0.
%    
%    - threshold_dB [21] (optional)
%      Threshold (in dB) below which paths are discarded, Scalar, Default: -140.0.
%    
%    - center_freq [22] (optional)
%      Center frequency in [Hz]; optional; If the value is not provided or set to 0, phase calculation
%      in coefficients is disabled, i.e. that path length has not influence on the results. This can be
%      used to calculate the antenna response for a specific angle and polarization. Scalar value
%    
%    - use_absolute_delays [23] (optional)
%      If true, the LOS delay is included for all paths; Default is false, i.e. delays are normalized
%      to the LOS delay.
%    
%    - active_path [24] (optional)
%      Optional bitmask for selecting active TX-IRS and IRS-RX path pairs. Ignores threshold_dB when provided.
%    
%    - ant_irs_2 [25] (optional)
%      Optional second IRS array (TX side for IRS → RX paths) for asymmetric IRS behavior. Same structure as for ant_irs
%    
% Output Arguments:
%    - coeff_re
%      Channel coefficients, real part, Size: [ n_ports_rx, n_ports_tx, n_path ]
%    
%    - coeff_im
%      Channel coefficients, imaginary part, Size: [ n_ports_rx, n_ports_tx, n_path ]
%    
%    - delays
%      Propagation delay in seconds, Size: [ n_ports_rx, n_ports_tx, n_path ]
%    
%    - active_path (optional)
%      Boolean mask of length n_path_1  n_path_2, indicating which path combinations were used.
%    
%    - aod (optional)
%      Azimuth of Departure angles in [rad], Size: [ n_ports_rx, n_ports_tx, n_path ]
%    
%    - eod (optional)
%      Elevation of Departure angles in [rad], Size: [ n_ports_rx, n_ports_tx, n_path ]
%    
%    - aoa (optional)
%      Azimuth of Arrival angles in [rad], Size: [ n_ports_rx, n_ports_tx, n_path ]
%    
%    - eoa (optional)
%      Elevation of Arrival angles in [rad], Size: [ n_ports_rx, n_ports_tx, n_path ]
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
    