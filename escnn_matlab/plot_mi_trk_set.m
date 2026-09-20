function [handles, labels] = plot_mi_trk_set( ...
    dir_path, trk_colors, ...
    snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts)
% PLOT_MI_TRK_SET  Plot one MI curve per trk= tracking mode found in
% dir_path. Mirrors plot_bler_trk_set.m -- see its header for the folder
% convention and why only the augmented curve is plotted per mode.
%
%   Encoding: solid = every trk mode, dashed = trk=notrack (see trk_style.m).

if nargin < 2, trk_colors = []; end
if nargin < 3 || isempty(snr_pad_left_db),   snr_pad_left_db   = 0; end
if nargin < 4 || isempty(snr_cut_right_pts), snr_cut_right_pts = 0; end
if nargin < 5 || isempty(snr_cut_left_pts),  snr_cut_left_pts  = 0; end

% dir_path is a folder mat_target() creates specifically for one trk sweep,
% so it only ever holds these mode .mat files -- safe to glob all of them.
files = dir(fullfile(dir_path, '*.mat'));
if isempty(files)
    error('plot_mi_trk_set: no .mat files found in %s', dir_path);
end

handles = [];
labels  = {};

for i = 1:numel(files)
    mat_file = fullfile(dir_path, files(i).name);
    try
        S = load(mat_file);
    catch ME
        warning('plot_mi_trk_set: could not load %s: %s', mat_file, ME.message);
        continue;
    end

    if ~isfield(S, 'trk_mode')
        warning('plot_mi_trk_set: %s has no trk_mode field, skipping', mat_file);
        continue;
    end
    trk_mode = strtrim(S.trk_mode);
    [color, line_style, marker] = trk_style(trk_mode, trk_colors);

    if isfield(S, 'mi_aug_1') && any(isfinite(S.mi_aug_1) & S.mi_aug_1 ~= 0)
        mi_vec = S.mi_aug_1;
    elseif isfield(S, 'mi_aug') && any(isfinite(S.mi_aug) & S.mi_aug ~= 0)
        mi_vec = S.mi_aug;
    elseif isfield(S, 'mi_escnn') && any(isfinite(S.mi_escnn) & S.mi_escnn ~= 0)
        mi_vec = S.mi_escnn;
    else
        warning('plot_mi_trk_set: %s has no non-zero mi_aug_1/mi_aug/mi_escnn, skipping', mat_file);
        continue;
    end

    % MI has its own SNR grid if present; otherwise fall back to snrs.
    if isfield(S, 'mi_snrs') && ~isempty(S.mi_snrs)
        snrs_src = S.mi_snrs(:)';
    else
        snrs_src = S.snrs(:)';
    end

    snrs = snrs_src;
    snrs = snrs(1 + snr_cut_left_pts : end - snr_cut_right_pts);
    if snr_pad_left_db > 0
        snr_step  = snrs(2) - snrs(1);
        snr_start = snrs(1) - snr_pad_left_db;
        snrs_pad  = snr_start : snr_step : snrs(1) - snr_step;
        snrs_plot = [snrs_pad, snrs];
    else
        snrs_plot = snrs;
    end
    n_pad = numel(snrs_plot) - numel(snrs);

    n_snrs_orig = numel(snrs_src);
    left_idx    = 1 + snr_cut_left_pts;
    right_idx   = n_snrs_orig - snr_cut_right_pts;
    trim_mi     = @(b) reshape(b(left_idx:right_idx), 1, []);
    pad_mi      = @(b) [zeros(1, n_pad), trim_mi(b)];

    mk_indices = 1 : 1 : numel(snrs_plot);

    h = plot(snrs_plot, pad_mi(mi_vec), ...
        'Color',           color, ...
        'LineStyle',       line_style, ...
        'Marker',          marker, ...
        'MarkerIndices',   mk_indices, ...
        'MarkerFaceColor', color, ...
        'MarkerSize',      5, ...
        'LineWidth',       1.4);

    handles(end+1) = h; %#ok<AGROW>
    labels{end+1}  = trk_mode; %#ok<AGROW>
end

if exist('snrs_plot', 'var') && ~isempty(snrs_plot)
    xlim([min(snrs_plot), max(snrs_plot)]);
end
end
