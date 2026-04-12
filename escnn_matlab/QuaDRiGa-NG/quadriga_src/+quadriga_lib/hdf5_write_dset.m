% HDF5_WRITE_DSET
%    Writes unstructured data to a HDF5 file
%    
% Description:
%    Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition 
%    to structured datasets, the library facilitates the inclusion of extra datasets of various types 
%    and shapes. This feature is particularly beneficial for integrating descriptive data or analysis 
%    results. The function quadriga_lib.hdf5_write_dset writes a single unstructured dataset. 
%    
% Usage:
%    
%    storage_dims = quadriga_lib.hdf5_write_dset( fn, location, name, data );
%    
% Input Arguments:
%    - fn
%      Filename of the HDF5 file, string
%    
%    - location (optional)
%      Storage location inside the file; 1-based; vector with 1-4 elements, i.e. [ix], [ix, iy], 
%      [ix,iy,iz] or [ix,iy,iz,iw]; Default: ix = iy = iz = iw = 1
%    
%    - name
%      Name of the dataset; String
%    
%    - data
%      Data to be written
%    
% Output Argument:
%    - storage_dims
%      Size of the dimensions of the storage space, vector with 4 elements, i.e. [nx,ny,nz,nw].
%    
% Caveat:
%    - Throws an error if dataset already exists at this location
%    - Throws an error if file does not exist (use hdf5_create_file)
%    - Supported types: string, double, float, (u)int32, (u)int64
%    - Supported size: up to 3 dimensions
%    - Storage order is maintained
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
    