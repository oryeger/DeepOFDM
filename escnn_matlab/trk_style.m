function [color, line_style, marker] = trk_style(trk_mode, custom_colors)
% TRK_STYLE  Consistent color/linestyle/marker for a given trk= tracking
% mode, shared by plot_bler_trk_set.m and plot_mi_trk_set.m so the same trk
% mode looks identical across both subplots of master_plot_trk_mi_bler.m.
%
%   trk_mode      : e.g. 'ekf', 'sgdbce', 'sgdbcei', 'sgdsyn', 'notrack', or
%                   the special pseudo-mode 'baseline' for the un-augmented
%                   reference curve (S.bler_no_aug/S.mi_no_aug -- read from a
%                   .mat file's S.trk_mode field, or passed as 'baseline'
%                   explicitly by the set-plotters for that 5th curve).
%   custom_colors : containers.Map(trk_mode -> [r g b]) overriding/extending
%                   the default palette below, or [] to use only the default.
%
%   Encoding: every trk mode (including 'notrack') is solid; only 'baseline'
%   (dash-dot, black) differs -- it's the un-augmented detector output with
%   no ESCNN involved at all, a different reference axis than 'notrack'
%   (which is still an augmented/aug curve, just without online tracking).

trk_mode = lower(strtrim(trk_mode));

default_colors = containers.Map( ...
    {'ekf', 'sgdbce', 'sgdbceep', 'sgdbcei', 'sgdbceicsg', 'sgdsyn', 'sgdsynep', 'notrack', 'baseline'}, ...
    { ...
      [0.00, 0.45, 0.70], ...   % ekf        - blue
      [0.85, 0.33, 0.10], ...   % sgdbce     - vermillion
      [0.90, 0.60, 0.00], ...   % sgdbceep   - orange
      [0.47, 0.67, 0.19], ...   % sgdbcei    - green
      [0.00, 0.60, 0.50], ...   % sgdbceicsg - teal
      [0.49, 0.18, 0.56], ...   % sgdsyn     - purple
      [0.80, 0.47, 0.65], ...   % sgdsynep   - pink
      [0.93, 0.69, 0.13], ...   % notrack    - goldenrod (was grey, too close to baseline's black)
      [0.00, 0.00, 0.00] ...    % baseline   - black
    });

default_markers = containers.Map( ...
    {'ekf', 'sgdbce', 'sgdbceep', 'sgdbcei', 'sgdbceicsg', 'sgdsyn', 'sgdsynep', 'notrack', 'baseline'}, ...
    {'^', 's', 's', 'o', 'o', 'd', 'd', 'x', '+'});

if nargin > 1 && ~isempty(custom_colors) && isKey(custom_colors, trk_mode)
    color = custom_colors(trk_mode);
elseif isKey(default_colors, trk_mode)
    color = default_colors(trk_mode);
else
    % Unknown trk mode (a new one added later): pick a stable-but-arbitrary
    % color from MATLAB's lines() palette instead of erroring out, so a new
    % track_mode still plots instead of blocking the whole comparison.
    palette = lines(16);
    idx = mod(sum(double(trk_mode)), size(palette, 1)) + 1;
    color = palette(idx, :);
end

if isKey(default_markers, trk_mode)
    marker = default_markers(trk_mode);
else
    marker = 'p';
end

if strcmp(trk_mode, 'baseline')
    line_style = '-.';
else
    line_style = '-';
end
end
