% QRT_FILE_PARSE
%    Read metadata from a QRT file
%    
% Description:
%    - Parses a QRT file and extracts metadata such as the number of snapshots, origins, destinations, and frequencies.
%    - All output arguments are optional; MATLAB only computes outputs that are requested.
%    - Can also retrieve CIR offsets per destination, human-readable names for origins and destinations, and the file version.
%    
% Usage:
%    [ no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, ...
%      fGHz, cir_pos, cir_orientation, orig_pos, orig_orientation ] = qrt_file_parse( fn );
%    
% Input Argument:
%    - fn
%      Path to the QRT file, string.
%    
% Output Arguments:
%    - no_cir
%      Number of channel snapshots per origin point, scalar.
%    
%    - no_orig
%      Number of origin points (TX), scalar.
%    
%    - no_dest
%      Number of destinations (RX), scalar.
%    
%    - no_freq
%      Number of frequency bands, scalar.
%    
%    - cir_offset
%      CIR offset for each destination, uint64 vector of size [no_dest, 1].
%    
%    - orig_names
%      Names of the origin points (TXs), cell array of strings with no_orig entries.
%    
%    - dest_names
%      Names of the destination points (RXs), cell array of strings with no_dest entries.
%      
%    - version
%      QRT file version number, scalar integer.
%    
%    - fGHz
%      Center frequency in GHz, float vector of size [no_freq, 1].
%    
%    - cir_pos
%      CIR positions in Cartesian coordinates, float matrix of size [no_cir, 3].
%    
%    - cir_orientation
%      CIR orientation in Euler angles in rad, float matrix of size [no_cir, 3].
%    
%    - orig_pos
%      Origin (TX) positions in Cartesian coordinates, float matrix of size [no_orig, 3].
%    
%    - orig_orientation
%      Origin (TX) orientations in Euler angles in rad, float matrix of size [no_orig, 3].
%    
% Example:
%    
%    [no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, ...
%        fGHz, cir_pos, cir_orientation, orig_pos, orig_orientation] = ...
%        qrt_file_parse('scene.qrt');
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
    