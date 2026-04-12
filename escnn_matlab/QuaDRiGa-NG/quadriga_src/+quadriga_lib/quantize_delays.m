% quantize_delays
%    Fixes the path delays to a grid of delay bins
%    
% Description:
%    - For channel emulation with finite delay resolution, path delays must be mapped to a fixed grid
%      of delay bins (taps). This function approximates each path delay using two adjacent taps with
%      power-weighted coefficients, producing smooth transitions in the frequency domain.
%    - For a path at fractional offset &delta; between tap indices, two taps are created with complex
%      coefficients scaled by (1&minus;&delta;)^&alpha; and &delta;^&alpha;, where &alpha; is the power
%      exponent.
%    - Input delays may be per-antenna [n_rx, n_tx, n_path, n_snap] or shared [1, 1, n_path, n_snap].
%    
% Usage:
%    [ coeff_re_q, coeff_im_q, delay_q ] = quadriga_lib.quantize_delays( coeff_re, coeff_im, delay, ...
%        tap_spacing, max_no_taps, power_exponent, fix_taps );
%    
% Input Arguments:
%    - coeff_re (required)
%      Channel coefficients, real part. 4D array of size [n_rx, n_tx, n_path, n_snap] (double).
%    
%    - coeff_im (required)
%      Channel coefficients, imaginary part. 4D array of size [n_rx, n_tx, n_path, n_snap] (double).
%    
%    - delay (required)
%      Path delays in seconds. 4D array of size [n_rx, n_tx, n_path, n_snap] or
%      [1, 1, n_path, n_snap] (double).
%    
%    - tap_spacing (optional)
%      Spacing of the delay bins in seconds. Scalar double. Default: 5e-9
%    
%    - max_no_taps (optional)
%      Maximum number of output taps. Scalar integer. 0 = unlimited. Default: 48
%    
%    - power_exponent (optional)
%      Interpolation exponent. Scalar double. Default: 1.0
%    
%    - fix_taps (optional)
%      Delay sharing mode. Scalar integer (0-3). Default: 0
%      0 = per tx-rx pair and snapshot, 1 = single grid for all,
%      2 = per snapshot, 3 = per tx-rx pair.
%    
% Output Arguments:
%    - coeff_re_q
%      Output coefficients, real part. 4D array of size [n_rx, n_tx, n_taps, n_snap] (double).
%    
%    - coeff_im_q
%      Output coefficients, imaginary part. 4D array of size [n_rx, n_tx, n_taps, n_snap] (double).
%    
%    - delay_q
%      Output delays in seconds. 4D array of size [n_rx, n_tx, n_taps, n_snap] or
%      [1, 1, n_taps, n_snap] (double).
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
    