% calc_angular_spread
%    Calculate azimuth and elevation angular spread with spherical wrapping
%    
% Description:
%    - Calculates the RMS azimuth and elevation angular spread from a set of power-weighted angles.
%    - Inputs and outputs for angles and powers are provided as 2D matrices where each column
%      represents a CIR (internally converted to vectors of column vectors to allow variable path
%      counts per CIR when called from C++).
%    - Uses optional spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
%      direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
%      on the equator before computing spreads.
%    - Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
%      despite energy being focused into a small solid angle). This method corrects for that.
%    - Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
%      elevation spread, corresponding to the principal axes of the angular power distribution.
%    - Setting wrapping to true enables spherical wrapping.
%    
% Usage:
%    [ as, es ] = quadriga_lib.calc_angular_spread( az, el, powers );
%    [ as, es, orientation, phi, theta ] = quadriga_lib.calc_angular_spread( az, el, powers, wrapping, calc_bank_angle, quantize );
%    
% Input Arguments:
%    - az
%      Azimuth angles in [rad], ranging from -pi to pi. Size [n_path, n_cir] (each column is one CIR).
%    
%    - el
%      Elevation angles in [rad], ranging from -pi/2 to pi/2. Size [n_path, n_cir].
%    
%    - powers
%      Path powers in [W]. Size [n_path, n_cir].
%    
%    - wrapping (optional)
%      Logical. If true, enable spherical rotation. Default: false (use raw angles)
%    
%    - calc_bank_angle (optional)
%      Logical. If true, compute the optimal bank angle analytically. Only used when
%      wrapping is true. Default: false.
%    
%    - quantize (optional)
%      Angular quantization step in [deg]. Default: 0 (no quantization).
%    
% Output Arguments:
%    - as
%      RMS azimuth angular spread in [rad]. Type: double. Size [n_cir, 1].
%    
%    - es
%      RMS elevation angular spread in [rad]. Type: double. Size [n_cir, 1].
%    
%    - orientation
%      Power-weighted mean-angle orientation: row 1 = bank angle, row 2 = tilt angle, row 3 = heading
%      angle, all in [rad]. Type: double. Size [3, n_cir].
%    
%    - phi
%      Rotated azimuth angles in [rad]. Type: double. Size [n_path, n_cir].
%    
%    - theta
%      Rotated elevation angles in [rad]. Type: double. Size [n_path, n_cir].
%    
% Example:
%    az = [0.1, 0.2; -0.1, -0.2; 0.05, 0.0];
%    el = [0.0, 0.05; 0.0, -0.05; 0.0, 0.0];
%    powers = [1.0, 2.0; 1.0, 1.0; 0.5, 1.5];
%    [as, es, orient] = quadriga_lib.calc_angular_spread(az, el, powers);
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
    