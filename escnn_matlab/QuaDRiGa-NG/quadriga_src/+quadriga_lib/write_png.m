% write_png
%    Write data to a PNG file
%    
% Description:
%    - Converts input data into a color-coded PNG file for visualization
%    - Support optional selection of a colormap, as well a minimum and maximum value limits
%    - Uses the LodePNG library for PNG writing
%    
% Declaration:
%    quadriga_lib.write_png( fn, data, colormap, min_val, max_val, log_transform )
%    
% Arguments:
%    - fn
%      Filename of the PNG file, string, required
%    
%    - data
%      Data matrix, required, size [N, M]
%    
%    - colormap (optional)
%      Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
%      'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', optional, default = 'jet'
%    
%    - min_val (optional)
%      Minimum value. Values below this value will have be encoded with the color of the smallest value.
%      If NAN is provided (default), the lowest values is determined from the data.
%    
%    - max_val (optional)
%      Maximum value. Values above this value will have be encoded with the color of the largest value.
%      If NAN is provided (default), the largest values is determined from the data.
%    
%    - log_transform (optional)
%      If enabled, the data values are transformed to the log-domain (10log10(data)) before processing.
%      Default: false (disabled)
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
    