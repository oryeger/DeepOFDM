% calc_cross_polarization_ratio
%    Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases
%    
% Description:
%    - Computes the aggregate cross-polarization ratio (XPR) from the polarization transfer matrices
%      of all channel impulse responses (CIRs) using the total-power-ratio method.
%    - For each CIR, the total co-polarized and cross-polarized received powers are accumulated
%      across all qualifying paths, and the XPR is obtained as a single ratio of the totals.
%    - In addition to the linear V/H basis, the XPR is also computed in the circular LHCP/RHCP basis.
%    - The LOS path is identified by comparing each path's absolute length against the direct
%      TX-RX distance. All paths with path_length < dTR + window_size are excluded by default.
%    - If the total cross-polarized power is zero, the XPR is set to 0 (undefined).
%    
% Usage:
%    [ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos )
%    [ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, include_los )
%    [ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, include_los, window_size )
%    
% Input Arguments:
%    - powers (required)
%      Path powers in Watts. A 2D matrix of size [n_path_max, n_cir] where columns are zero-padded
%      if CIRs have different numbers of paths. Alternatively, for a single CIR, a column vector of
%      length [n_path].
%    
%    - M (required)
%      Polarization transfer matrices. A 3D array of size [8, n_path_max, n_cir] with interleaved
%      real/imaginary parts in column-major order.
%    
%    - path_length (required)
%      Absolute path length from TX to RX in meters. A 2D matrix of size [n_path_max, n_cir].
%    
%    - tx_pos (required)
%      Transmitter position in Cartesian coordinates. Size [3, 1] (fixed) or [3, n_cir] (mobile).
%    
%    - rx_pos (required)
%      Receiver position in Cartesian coordinates. Size [3, 1] (fixed) or [3, n_cir] (mobile).
%    
%    - include_los (optional)
%      Logical flag. If true, include LOS paths in XPR calculation. Default: false.
%    
%    - window_size (optional)
%      LOS window size in meters. Default: 0.01.
%    
% Output Arguments:
%    - xpr (optional)
%      Cross-polarization ratio in linear scale. Size [n_cir, 6] (double).
%      Columns: 1=aggregate linear, 2=V-XPR, 3=H-XPR, 4=aggregate circular, 5=LHCP, 6=RHCP.
%    
%    - pg (optional)
%      Total path gain over all paths (including LOS). Column vector of length [n_cir] (double).
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
    