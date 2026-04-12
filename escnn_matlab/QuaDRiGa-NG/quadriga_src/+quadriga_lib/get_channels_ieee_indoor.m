% GET_CHANNELS_IEEE_INDOOR
%    Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models
%    
% Description:
%    - Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions.
%    - 2D model: no elevation angles are used; azimuth angles and planar motion are considered.
%    - For 3D antenna models (default models from [[arrayant_generate]]), only the azimuth cut at elevation_grid = 0 is used
%    - Supports channel model types A, B, C, D, E, F (as defined by TGn) via ChannelType.
%    - Can generate MU-MIMO channels (n_users > 1) with per-user distances/floors and optional angle
%      offsets according to TGac.
%    - Optional time evolution via observation_time, update_rate, and mobility parameters.
%    
% Declaration:
%    chan = quadriga_lib.get_channels_ieee_indoor(ap_array, sta_array, ChannelType, CarrierFreq_Hz, ...
%       tap_spacing_s, n_users, observation_time, update_rate, speed_station_kmh, speed_env_kmh, ...
%       Dist_m, n_floors, uplink, offset_angles, n_subpath, Doppler_effect, seed, ...
%       KF_linear, XPR_NLOS_linear, SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m);
%    
% ap_array:
%    - ap_array [1]
%      Struct containing the access point array antenna with n_tx elements (= ports after element coupling)
%      e_theta_re     | Real part of e-theta field component             | Size: [n_elevation_ap, n_azimuth_ap, n_elements_ap]
%      e_theta_im     | Imaginary part of e-theta field component        | Size: [n_elevation_ap, n_azimuth_ap, n_elements_ap]
%      e_phi_re       | Real part of e-phi field component               | Size: [n_elevation_ap, n_azimuth_ap, n_elements_ap]
%      e_phi_im       | Imaginary part of e-phi field component          | Size: [n_elevation_ap, n_azimuth_ap, n_elements_ap]
%      azimuth_grid   | Azimuth angles in [rad] -pi to pi, sorted        | Size: [n_azimuth_ap]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: [n_elevation_ap]
%    
%    - sta_array [2]
%      Struct containing the mobile station array antenna with n_rx elements (= ports after element coupling)
%      e_theta_re     | Real part of e-theta field component             | Size: [n_elevation_sta, n_azimuth_sta, n_elements_sta]
%      e_theta_im     | Imaginary part of e-theta field component        | Size: [n_elevation_sta, n_azimuth_sta, n_elements_sta]
%      e_phi_re       | Real part of e-phi field component               | Size: [n_elevation_sta, n_azimuth_sta, n_elements_sta]
%      e_phi_im       | Imaginary part of e-phi field component          | Size: [n_elevation_sta, n_azimuth_sta, n_elements_sta]
%      azimuth_grid   | Azimuth angles in [rad] -pi to pi, sorted        | Size: [n_azimuth_sta]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: [n_elevation_sta]
%    
%    - ChannelType [3]
%      Channel model type as defined by TGn. String. Supported: A, B, C, D, E, F.
%    
%    - CarrierFreq_Hz = 5.25e9 [4] (optional)
%      Carrier frequency in Hz.
%    
%    - tap_spacing_s = 10e-9 [5] (optional)
%      Tap spacing in seconds. Must be equal to 10 ns / 2^k (TGn default = 10e-9).
%    
%    - n_users = 1 [6] (optional)
%      Number of users (only for TGac, TGah). Output struct array length equals n_users.
%    
%    - observation_time = 0 [7] (optional)
%      Channel observation time in seconds. 0 creates a static channel.
%    
%    - update_rate = 1e-3 [8] (optional)
%      Channel update interval in seconds (only relevant when observation_time > 0).
%    
%    - speed_station_kmh = 0 [9] (optional)
%      Station movement speed in km/h. Movement direction is AoA_offset. Only relevant when observation_time > 0.
%    
%    - speed_env_kmh = 1.2 [10] (optional)
%      Environment movement speed in km/h. Default 1.2 for TGn, use 0.089 for TGac. Only relevant when observation_time > 0.
%    
%    - vector Dist_m = [4.99] [11] (optional)
%      TX-to-RX distance(s) in meters. Length n_users or length 1 (same distance for all users).
%    
%    - vector n_floors = [0] [12] (optional)
%      Number of floors for TGah model (per user), up to 4 floors. Length n_users or length 1.
%    
%    - uplink = false [13] (optional)
%      Channel direction flag. Default is downlink; set to true to generate reverse (uplink) direction.
%    
%    - offset_angles = [] [14] (optional)
%      Offset angles in degree for MU-MIMO channels. Empty uses model defaults (TGac auto for n_users > 1).
%      Size [4, n_users] with rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS.
%    
%    - n_subpath = 20 [15] (optional)
%      Number of sub-paths per path/cluster used for Laplacian angular spread mapping.
%    
%    - Doppler_effect = 50 [16] (optional)
%      Special Doppler effects: models D, E (fluorescent lights, value = mains freq.) and F (moving vehicle speed in km/h).
%      Use 0 to disable.
%    
%    - seed = -1 [17] (optional)
%      Numeric seed for repeatability. -1 disables the fixed seed and uses the system random device.
%    
%    - KF_linear = [] [18] (optional)
%      Overwrites the model-specific KF-value. If this parameter is empty (default), NAN or negative, model defaults are used:
%      A/B/C (KF = 1 for d < dBP, 0 otherwise); D (KF = 2 for d < dBP, 0 otherwise); E/F (KF = 4 for d < dBP, 0 otherwise).
%      KF is applied to the first tap only. Breakpoint distance is ignored for KF_linear >= 0.
%    
%    - XPR_NLOS_linear = [] [19] (optional)
%      Overwrites the model-specific Cross-polarization ratio. If this parameter is empty (default), NAN or negative,
%      the model default of 2 (3 dB) is used. XPR is applied to all NLOS taps.
%    
%    - SF_std_dB_LOS = [] [20] (optional)
%      Overwrites the model-specific shadow fading for LOS channels. If this parameter is empty (default) or NAN,
%      the model default of 3 dB is used. SF_std_dB_LOS is applied to all LOS channels, where the
%      AP-STA distance d < dBP.
%    
%    - SF_std_dB_NLOS = [] [21] (optional)
%      Overwrites the model-specific shadow fading for LOS channels. If this parameter is empty (default) or NAN,
%      the model defaults are A/B: 4 dB, C/D: 5 dB, E/F: 6 dB. SF_std_dB_NLOS is applied to all NLOS channels,
%      where the AP-STA distance d >= dBP.
%    
%    - dBP_m = [] [22] (optional)
%      Overwrites the model-specific breakpoint distance. If this parameter is empty (default), NAN or negative,
%      the model defaults are A/B/C: 5 m, D: 10 m, E: 20 m, F: 30 m.
%    
% Returns:
%    - chan
%      Struct array of length n_users containing the channel data with the following fields.
%      name           | Channel name                                                             | String
%      tx_position    | Transmitter positions (AP for downlink, STA for uplink)                  | Size: [3, 1] or [3, n_snap]
%      rx_position    | Receiver positions (STA for downlink, AP for uplink)                     | Size: [3, 1] or [3, n_snap]
%      tx_orientation | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | Size: [3, 1] or [3, n_snap]
%      rx_orientation | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | Size: [3, 1] or [3, n_snap]
%      coeff_re       | Channel coefficients, real part                                          | Size: [n_rx, n_tx, n_path, n_snap]
%      coeff_im       | Channel coefficients, imaginary part                                     | Size: [n_rx, n_tx, n_path, n_snap]
%      delay          | Propagation delays in seconds                                            | Size: [n_rx, n_tx, n_path, n_snap]
%      path_gain      | Path gain before antenna, linear scale                                   | Size: [n_path, n_snap]
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
    