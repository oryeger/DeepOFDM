% FAST_SINCOS
%    Fast, approximate sine/cosine for MATLAB numeric arrays
%    
% Description:
%    Computes elementwise sine and/or cosine for input angles in radians.
%    - Works on vectors, matrices, and 3-D arrays
%    - Accepts any numeric input class; best performance with single precision
%    - Outputs are always single precision
%    - Results are approximate and may differ from MATLAB sin / cos
%    - For x in [-pi, pi], the maximum absolute error is 2^(-22.1), and larger otherwise
%    - For x in [-500, 500], the maximum absolute error is 2^(-16.0)
%    - Request one or two outputs to control which results are returned
%    - With one output, set the optional cosineOnly flag to true to return cosine instead of sine
%    
% Usage:
%    
%    [s, c] = arrayant_lib.fast_sincos(x);
%    s = arrayant_lib.fast_sincos(x);
%    c = arrayant_lib.fast_sincos(x, true);
%    
% Input Arguments:
%    - x (input)
%      Numeric array of angles in radians. Any size/shape.
%    
%    - cosineOnly = false (optional input)
%      Logical scalar. When requesting a single output, set to true to return cos(x); otherwise returns
%      sin(x).
%    
% Output Arguments:
%    - s
%      Single-precision sin(x). Same size as x.
%    
%    - c
%      Single-precision cos(x). Same size as x.
%    
% Examples:
%    
%    % Input as single for best performance
%    x = single(linspace(0, 2pi, 1000));
%    
%    % Compute sine and cosine
%    [s, c] = arrayant_lib.fast_sincos(x);
%    
%    % Compute only sine (single output)
%    s = arrayant_lib.fast_sincos(x);
%    
%    % Compute only cosine (single output with flag)
%    c = arrayant_lib.fast_sincos(x, true);
%    
%    % Double input is accepted; outputs remain single
%    xd = linspace(0, 2pi, 8);
%    s_only = arrayant_lib.fast_sincos(xd);        % class(s_only) == 'single'
%    c_only = arrayant_lib.fast_sincos(xd, true);  % class(c_only) == 'single'
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
    