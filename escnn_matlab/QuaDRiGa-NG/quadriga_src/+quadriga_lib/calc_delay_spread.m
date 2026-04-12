% calc_delay_spread
%    Calculate the RMS delay spread in [s]
%    
% Description:
%    - Computes the root-mean-square (RMS) delay spread from a given set of delays and corresponding
%      linear-scale powers for each channel impulse response (CIR).
%    - An optional power threshold in [dB] relative to the strongest path can be applied. Paths with
%      power below p_max(dB) - threshold are excluded.
%    - An optional granularity parameter in [s] groups paths in the delay domain.
%    - Optionally returns the mean delay for each CIR.
%    
% Usage:
%    ds = quadriga_lib.calc_delay_spread( delays, powers );
%    ds = quadriga_lib.calc_delay_spread( delays, powers, threshold );
%    ds = quadriga_lib.calc_delay_spread( delays, powers, threshold, granularity );
%    [ ds, mean_delay ] = quadriga_lib.calc_delay_spread( delays, powers, threshold, granularity );
%    
% Arguments:
%    - delays (input)
%      Delays in [s]. A 2D matrix of size [n_path, n_cir]. Each row is one CIR. Rows may be
%      zero-padded if CIRs have different numbers of paths.
%    
%    - powers (input)
%      Path powers on a linear scale [W]. Same size as delays.
%    
%    - threshold = 100.0 (input)
%      Power threshold in [dB] relative to the strongest path. Default: 100 dB.
%    
%    - granularity = 0.0 (input)
%      Window size in [s] for grouping paths in the delay domain. Default: 0.
%    
% Returns:
%    - double ds (output)
%      RMS delay spread in [s] for each CIR. Size [n_cir, 1].
%    
%    - double mean_delay (optional output)
%      Mean delay in [s] for each CIR. Size [n_cir, 1].
%    
% Example:
%    delays = [0, 1e-6, 2e-6];
%    powers = [1.0, 0.5, 0.25];
%    [ds, mean_delay] = quadriga_lib.calc_delay_spread( delays, powers );
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
    