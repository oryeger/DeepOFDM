% GET_CHANNELS_PLANAR
%    Calculate channel coefficients for planar waves
%    
% Description:
%    In this function, the wireless propagation channel between a transmitter and a receiver is calculated,
%    based on a single transmit and receive position. Additionally, interaction points with the environment,
%    which are derived from either Ray Tracing or Geometric Stochastic Models such as QuaDRiGa, are
%    considered. The calculation is performed under the assumption of planar wave propagation. For accurate
%    execution of this process, several pieces of input data are required:
%    
%    - The 3D Cartesian (local) coordinates of both the transmitter and the receiver.
%    - The azimuth/elevation departure and arrval angles.
%    - The polarization transfer matrix for each propagation path.
%    - Antenna models for both the transmitter and the receiver.
%    - The orientations of the antennas.
%    
% Usage:
%    
%    [ coeff_re, coeff_im, delays, rx_Doppler ] = quadriga_lib.get_channels_planar( ant_tx, ant_rx, ...
%        aod, eod, aoa, eoa, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, ...
%        center_freq, use_absolute_delays, add_fake_los_path );
%    
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
%    - aod [3] (required)
%      Departure azimuth angles in [rad], Size: [ 1, n_path ] or [ n_path, 1 ]
%    
%    - eod [4] (required)
%      Departure elevation angles in [rad], Size: [ 1, n_path ] or [ n_path, 1 ]
%    
%    - aoa [5] (required)
%      Arrival azimuth angles in [rad], Size: [ 1, n_path ] or [ n_path, 1 ]
%    
%    - eoa [6] (required)
%      Arrival elevation angles in [rad], Size: [ 1, n_path ] or [ n_path, 1 ]
%    
%    - path_gain [7] (required)
%      Path gain (linear scale), Size: [ 1, n_path ] or [ n_path, 1 ]
%    
%    - path_length [8] (required)
%      Total path length in meters, Size: [ 1, n_path ] or [ n_path, 1 ]
%    
%    - M [9] (required)
%      Polarization transfer matrix, interleaved complex values (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH),
%      Size: [ 8, n_path ]
%    
%    - tx_pos [10] (required)
%      Transmitter position in 3D Cartesian coordinates; Size: [3,1] or [1,3]
%    
%    - tx_orientation [11] (required)
%      3-element vector describing the orientation of the transmit antenna in Euler angles (bank, tilt, heading),
%      Size: [3,1] or [1,3]
%    
%    - rx_pos [12] (required)
%      Receiver position in 3D Cartesian coordinates, Size: [3,1] or [1,3]
%    
%    - rx_orientation [13 (required)]
%      3-element vector describing the orientation of the receive antenna in Euler angles,
%      Size: [3,1] or [1,3]
%    
%    - center_freq [14] (optional)
%      Center frequency in [Hz]; optional; If the value is not provided or set to 0, phase calculation
%      in coefficients is disabled, i.e. that path length has not influence on the results. This can be
%      used to calculate the antenna response for a specific angle and polarization. Scalar value
%    
%    - use_absolute_delays [15] (optional)
%      If true, the LOS delay is included for all paths; Default is false, i.e. delays are normalized
%      to the LOS delay.
%    
%    - add_fake_los_path [16] (optional)
%      If true, adds a zero-power LOS path as the first path in case where no LOS path was present.
%      Default: false
%    
% Derived inputs:
%      n_azimuth_tx   | Number of azimuth angles in the TX antenna pattern
%      n_elevation_tx | Number of elevation angles in the TX antenna pattern
%      n_elements_tx  | Number of physical antenna elements in the TX array antenna
%      n_ports_tx     | Number of ports (after coupling) in the TX array antenna
%      n_azimuth_rx   | Number of azimuth angles in the RX antenna pattern
%      n_elevation_rx | Number of elevation angles in the RX antenna pattern
%      n_elements_rx  | Number of physical antenna elements in the RX array antenna
%      n_ports_rx     | Number of ports (after coupling) in the RX array antenna
%      n_path         | Number of propagation paths
%    
% Output Arguments:
%    - coeff_re
%      Channel coefficients, real part, Size: [ n_ports_tx, n_ports_rx, n_path ]
%    
%    - coeff_im
%      Channel coefficients, imaginary part, Size: [ n_ports_tx, n_ports_rx, n_path ]
%    
%    - delays
%      Propagation delay in seconds, Size: [ n_ports_tx, n_ports_rx, n_path ]
%    
%    - rx_Doppler
%      Doppler weights for moving RX, Size: [ 1, n_path ]
%    
% Caveat:
%    - Input data is directly accessed from MATLAB / Octave memory, without copying if it is provided in
%      double precision.
%    - Other formats (e.g. single precision inputs) will be converted to double automatically, causing
%      additional computation steps.
%    - To improve performance of repeated computations (e.g. in loops), consider preparing the data
%      accordingly to avoid unecessary computations.
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
    