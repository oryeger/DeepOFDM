% ARRAYANT_INTERPOLATE
%    Interpolate array antenna field patterns
%    
% Description:
%    This function interpolates polarimetric antenna field patterns for a given set of azimuth and
%    elevation angles.
%    
% Usage:
%    
%    % Arrayant as struct
%    [V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma] = ...
%        quadriga_lib.arrayant_interpolate( arrayant, azimuth, elevation, element, orientation, element_pos );
%    
%    % Arrayant as separate inputs
%    [V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma] = ...
%        quadriga_lib.arrayant_interpolate( [], azimuth, elevation, element, orientation, element_pos,
%        e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid );
%    
% Input Arguments:
%    - arrayant [1] (optional)
%      Struct containing the arrayant data with the following fields:
%      e_theta_re     | e-theta field component, real part                    | Size: [n_elevation, n_azimuth, n_elements]
%      e_theta_im     | e-theta field component, imaginary part               | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_re       | e-phi field component, real part                      | Size: [n_elevation, n_azimuth, n_elements]
%      e_phi_im       | e-phi field component, imaginary part                 | Size: [n_elevation, n_azimuth, n_elements]
%      azimuth_grid   | Azimuth angles in [rad], -pi to pi, sorted            | Size: [n_azimuth]
%      elevation_grid | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: [n_elevation]
%      element_pos    | Antenna element (x,y,z) positions, optional           | Size: [3, n_elements]
%      If an empty array is passed, array antenna data is provided as separate inputs (Inputs 7-12)
%    
%    - azimuth [2] (required)
%      Azimuth angles in [rad] for which the field pattern should be interpolated. Values must be
%      between -pi and pi, single or double precision.
%      Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
%                 | Size: [1, n_ang]
%      Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
%                 | Size: [n_out, n_ang]
%    
%    - elevation [3] (required)
%      Elevation angles in [rad] for which the field pattern should be interpolated. Values must be
%      between -pi/2 and pi/2, single or double precision.
%      Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
%                 | Size: [1, n_ang]
%      Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
%                 | Size: [n_out, n_ang]
%    
%    - element [4] (optional)
%      The element indices for which the interpolation should be done. Optional parameter. Values must
%      be between 1 and n_elements. It is possible to duplicate elements, i.e. by passing [1,1,2].
%      If this parameter is not provided (or an empty array is passed), i_element is initialized
%      to [1:n_elements]. In this case, n_out = n_elements. Allowed types: uint32 or double.
%      Size: [1, n_out] or [n_out, 1] or empty []
%    
%    - orientation [5] (optional)
%      This (optional) 3-element vector describes the orientation of the array antenna or of individual
%      array elements. The The first value describes the ”bank angle”, the second value describes the
%      ”tilt angle”, (positive values point upwards), the third value describes the bearing or ”heading
%      angle”, in mathematic sense. Values must be given in [rad]. East corresponds to 0, and the
%      angles increase counter-clockwise, so north is pi/2, south is -pi/2, and west is equal to pi. By
%      default, the orientation is [0,0,0]', i.e. the broadside of the antenna points at the horizon
%      towards the East. Single or double precision
%      Size: [3, 1] or [3, n_out] or [3, 1, n_ang] or [3, n_out, n_ang] or empty []
%    
%    - element_pos [6] (optional)
%      Alternative positions of the array antenna elements in local cartesian coordinates (using units of [m]).
%      If this parameter is not given, element positions arrayant are used. If the arrayant has no
%      positions, they are initialzed to [0,0,0]. For example, when duplicating the fist element by setting
%      element = [1,1], different element positions can be set for the  two elements in the output.
%      Size: [3, n_out] or empty []
%    
% Derived inputs:
%      n_azimuth      | Number of azimuth angles in the filed pattern
%      n_elevation    | Number of elevation angles in the filed pattern
%      n_elements     | Number of antenna elements filed pattern of the array antenna
%      n_ang          | Number of interpolation angles
%      n_out          | Number of antenna elements in the generated output (may differ from n_elements)
%    
% Output Arguments:
%    - vr
%      Real part of the interpolated e-theta (vertical) field component. Size [n_out, n_ang]
%    
%    - vi
%      Imaginary part of the interpolated e-theta (vertical) field component. Size [n_out, n_ang]
%    
%    - hr
%      Real part of the interpolated e-phi (horizontal) field component. Size [n_out, n_ang]
%    
%    - hi
%      Imaginary part of the interpolated e-phi (horizontal) field component. Size [n_out, n_ang]
%    
%    - dist (optional)
%      The effective distances between the antenna elements when seen from the direction of the
%      incident path. The distance is calculated by an projection of the array positions on the normal
%      plane of the incident path. This is needed for calculating the phase of the antenna response.
%      Size [n_out, n_ang]
%    
%    - azimuth_loc (optional)
%      The azimuth angles in [rad] for the local antenna coordinate system, i.e., after applying the
%      'orientation'. If no orientation vector is given, these angles are identical to the input
%      azimuth angles. Size [n_out, n_ang]
%    
%    - elevation_loc (optional)
%      The elevation angles in [rad] for the local antenna coordinate system, i.e., after applying the
%      'orientation'. If no orientation vector is given, these angles are identical to the input
%      elevation angles. Size [n_out, n_ang]
%    
%    - gamma (optional)
%      Polarization rotation angles in [rad], Size [n_out, n_ang]
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
    