% acdf
%    Calculate the empirical averaged cumulative distribution function (CDF)
%    
% Description:
%    - Calculates the empirical CDF from the given data matrix, where each column represents an
%      independent data set (e.g., repeated experiment runs).
%    - Individual CDFs are computed per column and an averaged CDF is obtained by interpolation in
%      quantile space.
%    - Inf and NaN values in the data are excluded from the computation.
%    - If bins is empty or not provided, 201 equally spaced bins spanning the data range are generated.
%    
% Usage:
%    [ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data );
%    [ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data, bins );
%    [ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data, bins, n_bins );
%    
% Input arguments:
%    - data (input)
%      Input data matrix. Size [n_samples, n_sets]. Each column is one data set.
%    
%    - bins = [] (optional input)
%      Bin centers for the histogram. Length [n_bins]. If empty, bins are auto-generated.
%    
%    - n_bins = 201 (optional input)
%      Number of bins to generate when bins are auto-generated. Must be at least 2. Ignored when
%      non-empty bins are provided.
%    
% Output arguments:
%    - double Sh (output)
%      Individual CDFs. Size [n_bins, n_sets].
%    
%    - double bins (output)
%      Bin centers. Length [n_bins].
%    
%    - double Sc (output)
%      Averaged CDF. Length [n_bins].
%    
%    - double mu (output)
%      Mean of the 0.1, 0.2, ..., 0.9 quantiles across data sets. Length [9].
%    
%    - double sig (output)
%      Standard deviation of the 0.1, 0.2, ..., 0.9 quantiles across data sets. Length [9].
%    
% Example:
%    data = randn(10000, 5);
%    [ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data );
%    % bins has 201 elements, Sh is [201, 5], Sc is [201, 1], mu and sig are [9, 1]
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
    