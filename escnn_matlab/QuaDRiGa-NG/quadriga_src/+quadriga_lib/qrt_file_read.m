% QRT_FILE_READ
%    Read ray-tracing data from a QRT file
%    
% Description:
%    - Reads channel impulse response (CIR) data from a QRT file for a specific snapshot and origin point.
%    - Supports both uplink and downlink directions by swapping TX/RX roles accordingly.
%    - All output arguments are optional; MATLAB only computes outputs that are requested.
%    - The normalize_M parameter controls how the polarization transfer matrix M and path gains are returned.
%    
% Usage:
%    [ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, ...
%      path_gain, path_length, M, aod, eod, aoa, eoa, path_coord ] = ...
%        qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M );
%    
% Input Arguments:
%    - fn
%      Path to the QRT file, string.
%    
%    - i_cir (optional)
%      Snapshot index (1-based), scalar. Default: 1
%    
%    - i_orig (optional)
%      Origin index (1-based). For downlink, origin corresponds to the transmitter, scalar. Default: 1
%    
%    - downlink (optional)
%      If true, origin is TX and destination is RX (downlink). If false, roles are swapped (uplink), 
%      logical scalar. Default: true
%    
%    - normalize_M (optional)
%      Normalization option for the polarization transfer matrix, scalar integer. Default: 1
%       0 | M as stored in QRT file, path_gain is -FSPL
%       1 | M has sum-column power of 2, path_gain is -FSPL minus material losses
%    
% Output Arguments:
%    - center_freq
%      Center frequency in Hz, double vector of size [n_freq, 1].
%    
%    - tx_pos
%      Transmitter position in Cartesian coordinates, double vector of size [3, 1].
%    
%    - tx_orientation
%      Transmitter orientation (bank, tilt, heading) in radians, double vector of size [3, 1].
%    
%    - rx_pos
%      Receiver position in Cartesian coordinates, double vector of size [3, 1].
%    
%    - rx_orientation
%      Receiver orientation (bank, tilt, heading) in radians, double vector of size [3, 1].
%    
%    - fbs_pos
%      First-bounce scatterer positions, double matrix of size [3, n_path].
%    
%    - lbs_pos
%      Last-bounce scatterer positions, double matrix of size [3, n_path].
%    
%    - path_gain
%      Path gain on linear scale, double matrix of size [n_path, n_freq].
%    
%    - path_length
%      Absolute path length from TX to RX phase center, double vector of size [n_path, 1].
%    
%    - M
%      Polarization transfer matrix, double array of size [8, n_path, n_freq] or [2, n_path, n_freq] 
%      for v6 files.
%    
%    - aod
%      Departure azimuth angles in radians, double vector of size [n_path, 1].
%    
%    - eod
%      Departure elevation angles in radians, double vector of size [n_path, 1].
%    
%    - aoa
%      Arrival azimuth angles in radians, double vector of size [n_path, 1].
%    
%    - eoa
%      Arrival elevation angles in radians, double vector of size [n_path, 1].
%    
%    - path_coord
%      Interaction coordinates per path, cell array of length n_path. Each cell contains a double matrix 
%      of size [3, n_interact + 2].
%    
% Example:
%    [center_freq, tx_pos, ~, rx_pos, ~, fbs_pos, lbs_pos, path_gain, path_length, M, ...
%        aod, eod, aoa, eoa] = qrt_file_read('scene.qrt', 1, 1, true, 1);
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
    