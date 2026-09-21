function [handles, labels] = plot_bler_trk_set( ...
    dir_path, trk_colors, add_snr_target, ...
    snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts)
% PLOT_BLER_TRK_SET  Plot one BLER curve per trk= tracking mode found in
% dir_path (one folder = one fixed underlying config, produced by
% build_analysis.py's mat_target() for a trk= sweep: ekf/sgdbce/sgdbcei/
% sgdsyn/notrack/... each saved as its own <mode>.mat inside it, e.g.
% ekf.mat, sgdbcei.mat -- the mode name is also read back from each file's
% S.trk_mode field, not parsed from the filename), plus one un-augmented
% baseline curve (e.g. plain LMMSE, no ESCNN involved at all -- read from
% S.bler_no_aug/S.aug_type, identical across every file in dir_path since
% trk doesn't affect it, so it's only plotted once).
%
%   trk_colors        : containers.Map(trk_mode -> [r g b]), or [] for the
%                        trk_style.m default palette.
%   add_snr_target     : append SNR@10% to each legend label.
%   snr_pad_left_db    : extend SNR axis left by this many dB with BLER=1 padding
%   snr_cut_right_pts  : remove this many points from the right of SNR/BLER vectors
%   snr_cut_left_pts   : remove this many points from the left  of SNR/BLER vectors
%
%   Encoding: solid = every trk mode, dashed = trk=notrack, black dash-dot =
%   the baseline curve (see trk_style.m).
%   Plots each mode's augmented (escnn) curve -- bler_aug_1, falling back to
%   bler_aug -- since trk only changes how that augmentation is tracked
%   online.

if nargin < 2, trk_colors = []; end
if nargin < 3 || isempty(add_snr_target),    add_snr_target    = false; end
if nargin < 4 || isempty(snr_pad_left_db),   snr_pad_left_db   = 0;     end
if nargin < 5 || isempty(snr_cut_right_pts), snr_cut_right_pts = 0;     end
if nargin < 6 || isempty(snr_cut_left_pts),  snr_cut_left_pts  = 0;     end

% dir_path is a folder mat_target() creates specifically for one trk sweep,
% so it only ever holds these mode .mat files -- safe to glob all of them.
files = dir(fullfile(dir_path, '*.mat'));
if isempty(files)
    error('plot_bler_trk_set: no .mat files found in %s', dir_path);
end

handles = [];
labels  = {};
S_baseline = [];
S_baseline_is_ekf = false;

for i = 1:numel(files)
    mat_file = fullfile(dir_path, files(i).name);
    try
        S = load(mat_file);
    catch ME
        warning('plot_bler_trk_set: could not load %s: %s', mat_file, ME.message);
        continue;
    end

    % Baseline (bler_no_aug/aug_type) is identical across every file in
    % dir_path -- it doesn't depend on trk -- so prefer pulling it from the
    % ekf file specifically for a deterministic choice; only fall back to
    % whichever other file has it (first one found) if ekf's is missing.
    if ~S_baseline_is_ekf && isfield(S, 'bler_no_aug') && isfield(S, 'aug_type')
        if isfield(S, 'trk_mode') && strcmpi(strtrim(S.trk_mode), 'ekf')
            S_baseline = S;
            S_baseline_is_ekf = true;
        elseif isempty(S_baseline)
            S_baseline = S;
        end
    end

    if ~isfield(S, 'trk_mode')
        warning('plot_bler_trk_set: %s has no trk_mode field, skipping', mat_file);
        continue;
    end
    trk_mode = strtrim(S.trk_mode);
    [color, line_style, marker] = trk_style(trk_mode, trk_colors);

    if isfield(S, 'bler_aug_1')
        bler_vec = S.bler_aug_1;
    elseif isfield(S, 'bler_aug')
        bler_vec = S.bler_aug;
    else
        warning('plot_bler_trk_set: %s has neither bler_aug_1 nor bler_aug, skipping', mat_file);
        continue;
    end

    [snrs_plot, pad_bler] = local_snr_grid(S, snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts);
    mk_indices = 1 : 1 : numel(snrs_plot);

    if add_snr_target && isfield(S, 'snr_target_aug_1')
        lbl = sprintf('aug %s, SNR@10%%=%g', trk_mode, S.snr_target_aug_1);
    else
        lbl = ['aug ', trk_mode];
    end

    h = semilogy(snrs_plot, pad_bler(bler_vec), ...
        'Color',           color, ...
        'LineStyle',       line_style, ...
        'Marker',          marker, ...
        'MarkerIndices',   mk_indices, ...
        'MarkerFaceColor', color, ...
        'MarkerSize',      5, ...
        'LineWidth',       1.4);

    handles(end+1) = h; %#ok<AGROW>
    labels{end+1}  = lbl; %#ok<AGROW>
end

% ---- Baseline (un-augmented) reference curve, plotted once ----
if ~isempty(S_baseline)
    [snrs_plot, pad_bler] = local_snr_grid(S_baseline, snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts);
    mk_indices = 1 : 1 : numel(snrs_plot);
    [color, line_style, marker] = trk_style('baseline', trk_colors);

    if add_snr_target && isfield(S_baseline, 'snr_target_no_aug')
        lbl = sprintf('no aug, SNR@10%%=%g', S_baseline.snr_target_no_aug);
    else
        lbl = 'no aug';
    end

    h = semilogy(snrs_plot, pad_bler(S_baseline.bler_no_aug), ...
        'Color',           color, ...
        'LineStyle',       line_style, ...
        'Marker',          marker, ...
        'MarkerIndices',   mk_indices, ...
        'MarkerFaceColor', color, ...
        'MarkerSize',      5, ...
        'LineWidth',       1.4);

    handles(end+1) = h; %#ok<AGROW>
    labels{end+1}  = lbl; %#ok<AGROW>
else
    warning('plot_bler_trk_set: no file in %s had bler_no_aug+aug_type, baseline curve skipped', dir_path);
end

if exist('snrs_plot', 'var') && ~isempty(snrs_plot)
    xlim([min(snrs_plot), max(snrs_plot)]);
end
end

function [snrs_plot, pad_fn] = local_snr_grid(S, snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts)
% Shared SNR trim/pad logic (BLER pads with 1, i.e. certain error).
snrs = S.snrs(:)';
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

n_snrs_orig = numel(S.snrs);
left_idx    = 1 + snr_cut_left_pts;
right_idx   = n_snrs_orig - snr_cut_right_pts;
trim_bler   = @(b) reshape(b(left_idx:right_idx), 1, []);
pad_fn      = @(b) [ones(1, n_pad), trim_bler(b)];
end
