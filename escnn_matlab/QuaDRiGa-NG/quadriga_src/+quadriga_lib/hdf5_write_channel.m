% HDF5_WRITE_CHANNEL
%    Writes channel data to HDF5 files
%    
% Description:
%    Quadriga-Lib provides an HDF5-based solution for storing and organizing channel data. This function
%    can be used to write structured and unstructured data to an HDF5 file. 
%    
% Usage:
%    
%    storage_dims = quadriga_lib.hdf5_write_channel( fn, location, par, rx_position, tx_position, ...
%       coeff_re, coeff_im, delay, center_freq, name, initial_pos, path_gain, path_length, ...
%       path_polarization, path_angles, path_fbs_pos, path_lbs_pos, no_interact, interact_coord, ...
%       rx_orientation, tx_orientation )
%    
% Input Arguments:
%    - fn
%      Filename of the HDF5 file, string
%    
%    - location (optional)
%      Storage location inside the file; 1-based; vector with 1-4 elements, i.e. [ix], [ix, iy], 
%      [ix,iy,iz] or [ix,iy,iz,iw]; Default: ix = iy = iz = iw = 1
%    
%    - par
%      Unstructured data as struct, can be empty if no unstructured data should be written
%    
%    - Structured data: (inputs 4-21, single or double precision)
%      rx_position    | Receiver positions                                       | [3, n_snap] or [3, 1]
%      tx_position    | Transmitter positions                                    | [3, n_snap] or [3, 1]
%      coeff_re       | Channel coefficients, real part                          | [n_rx, n_tx, n_path, n_snap]
%      coeff_im       | Channel coefficients, imaginary part                     | [n_rx, n_tx, n_path, n_snap]
%      delay          | Propagation delays in seconds                            | [n_rx, n_tx, n_path, n_snap] or [1, 1, n_path, n_snap]
%      center_freq    | Center frequency in [Hz]                                 | [n_snap, 1] or scalar
%      name           | Name of the channel                                      | String
%      initial_pos    | Index of reference position, 1-based                     | uint32, scalar
%      path_gain      | Path gain before antenna, linear scale                   | [n_path, n_snap]
%      path_length    | Path length from TX to RX phase center in m              | [n_path, n_snap]
%      polarization   | Polarization transfer function, interleaved complex      | [8, n_path, n_snap]
%      path_angles    | Departure and arrival angles {AOD, EOD, AOA, EOA} in rad | [n_path, 4, n_snap]
%      path_fbs_pos   | First-bounce scatterer positions                         | [3, n_path, n_snap]
%      path_lbs_pos   | Last-bounce scatterer positions                          | [3, n_path, n_snap]
%      no_interact    | Number interaction points of paths with the environment  | uint32, [n_path, n_snap]
%      interact_coord | Interaction coordinates                                  | [3, max(sum(no_interact)), n_snap]
%      rx_orientation | Transmitter orientation                                  | [3, n_snap] or [3, 1]
%      tx_orientation | Receiver orientation                                     | [3, n_snap] or [3, 1]
%    
% Output Arguments:
%    - storage_dims
%      Size of the dimensions of the storage space, vector with 4 elements, i.e. [nx,ny,nz,nw].
%    
% Caveat:
%    - If the file exists already, the new data is added to the exisiting file
%    - If a new file is created, a storage layout is created to store the location of datasets in the file
%    - For location = [ix] storage layout is [65536,1,1,1] or [ix,1,1,1] if (ix > 65536)
%    - For location = [ix,iy] storage layout is [1024,64,1,1]
%    - For location = [ix,iy,iz] storage layout is [256,16,16,1]
%    - For location = [ix,iy,iz,iw] storage layout is [128,8,8,8]
%    - You can create a custom storage layout by creating the file first using "hdf5_create_file"
%    - You can reshape the storage layout by using "hdf5_reshape_storage", but the total number of elements must not change
%    - Inputs can be empty or missing.
%    - All structured data is written in single precision (but can can be provided as single or double)
%    - Unstructured datatypes are maintained in the HDF file
%    - Supported unstructured types: string, double, float, (u)int32, (u)int64
%    - Supported unstructured size: up to 3 dimensions
%    - Storage order of the unstructured data is maintained
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
    