% BASEBAND_FREQ_RESPONSE
%    Transforms the channel into frequency domain and returns the frequency response
%    
% Usage:
%    
%    [ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( coeff_re, coeff_im, delay, pilot_grid, bandwidth, i_snap );
%    
% Input Arguments:
%    - coeff_re
%      Channel coefficients, real part, Size: [ n_rx, n_tx, n_path, n_snap ]
%    
%    - coeff_im
%      Channel coefficients, imaginary part, Size: [ n_rx, n_tx, n_path, n_snap ]
%    
%    - delays
%      Propagation delay in seconds, Size: [ n_rx, n_tx, n_path, n_snap ] or [ 1, 1, n_path, n_snap ] 
%      or [ n_path, n_snap ]
%    
%    - pilot_grid
%      Sub-carrier positions relative to the bandwidth. The carrier positions are given relative to the
%      bandwidth where '0' is the begin of the spectrum (i.e., the center frequency f0) and '1' is
%      equal to f0+bandwidth. To obtain the channel frequency response centered around f0, the
%      input variable 'pilot_grid' must be set to '(-N/2:N/2)/N', where N is the number of sub-
%      carriers. Vector of length: [ n_carriers ]
%    
%    - bandwidth
%      The baseband bandwidth in [Hz], scalar
%    
%    - i_snap (optional)
%      Snapshot indices for which the frequency response should be generated (1-based index). If this
%      variable is not given, all snapshots are processed. Length: [ n_out ]
%    
% Output Argument:
%    - hmat_re
%      Freq. domain channel matrices (H), real part, Size [ n_rx, n_tx, n_carriers, n_out ]
%    
%    - hmat_im
%      Freq. domain channel matrices (H), imaginary part, Size [ n_rx, n_tx, n_carriers, n_out ]
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
    