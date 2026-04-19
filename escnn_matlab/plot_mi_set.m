function [handles, labels] = plot_mi_set( ...
    dir_path, ...
    algs_to_plot, alg_colors, alg_names, alg_files, ...
    markers_no_aug, markers_aug, fillable_algs, ...
    plot_aug_iter_2, snr_pad_left_db, snr_cut_right_pts)
% PLOT_MI_SET  Plot one set of MI curves from a single directory.
%
%   snr_pad_left_db   : extend SNR axis left by this many dB with MI=0 padding
%   snr_cut_right_pts : remove this many points from the right of SNR/MI vectors
%
%   Encoding:
%     Dashed + hollow marker = no aug
%     Solid  + filled marker = aug

% Extract code rate label from directory path
token = regexp(dir_path, '_(0\.\d+)', 'tokens', 'once');
if ~isempty(token)
    cr_str = [', r=', token{1}];
else
    cr_str = '';
end

handles = [];
labels  = {};

for alg = algs_to_plot
    mat_file = fullfile(dir_path, [alg_files{alg}, '.mat']);

    if ~isfile(mat_file)
        warning('plot_mi_set: file not found, skipping: %s', mat_file);
        continue;
    end

    try
        S = load(mat_file);
    catch ME
        warning('plot_mi_set: could not load %s: %s', mat_file, ME.message);
        continue;
    end

    alg_name    = alg_names{alg};
    alg_color   = alg_colors(alg,:);
    is_fillable = ismember(alg, fillable_algs);

    % Resolve MI vectors. Prefer aug/no_aug fields (newer .mat); fall back
    % to per-alg fields (older .mat) so legacy files still plot.
    no_aug_field = ['mi_', alg_files{alg}];
    mi_no_aug_vec = [];
    mi_aug_1_vec  = [];
    mi_aug_2_vec  = [];

    if isfield(S, 'mi_no_aug') && any(isfinite(S.mi_no_aug) & S.mi_no_aug ~= 0)
        mi_no_aug_vec = S.mi_no_aug;
    elseif isfield(S, no_aug_field) && any(isfinite(S.(no_aug_field)) & S.(no_aug_field) ~= 0)
        mi_no_aug_vec = S.(no_aug_field);
    end

    if isfield(S, 'mi_aug_1') && any(isfinite(S.mi_aug_1) & S.mi_aug_1 ~= 0)
        mi_aug_1_vec = S.mi_aug_1;
    elseif isfield(S, 'mi_aug') && any(isfinite(S.mi_aug) & S.mi_aug ~= 0)
        mi_aug_1_vec = S.mi_aug;
    elseif isfield(S, 'mi_escnn') && any(isfinite(S.mi_escnn) & S.mi_escnn ~= 0)
        mi_aug_1_vec = S.mi_escnn;
    end

    if plot_aug_iter_2
        if isfield(S, 'mi_aug_2') && any(isfinite(S.mi_aug_2) & S.mi_aug_2 ~= 0)
            mi_aug_2_vec = S.mi_aug_2;
        elseif isfield(S, 'mi_escnn_2') && any(isfinite(S.mi_escnn_2) & S.mi_escnn_2 ~= 0)
            mi_aug_2_vec = S.mi_escnn_2;
        end
    end

    if isempty(mi_no_aug_vec) && isempty(mi_aug_1_vec) && isempty(mi_aug_2_vec)
        warning('plot_mi_set: no usable MI fields in %s, skipping.', mat_file);
        continue;
    end

    % MI has its own SNR grid if present; otherwise fall back to snrs
    if isfield(S, 'mi_snrs') && ~isempty(S.mi_snrs)
        snrs_src = S.mi_snrs(:)';
    else
        snrs_src = S.snrs(:)';
    end

    snrs = snrs_src;
    if snr_cut_right_pts > 0
        snrs = snrs(1 : end - snr_cut_right_pts);
    end
    if snr_pad_left_db > 0
        snr_step   = snrs(2) - snrs(1);
        snr_start  = snrs(1) - snr_pad_left_db;
        snrs_pad   = snr_start : snr_step : snrs(1) - snr_step;
        snrs_plot  = [snrs_pad, snrs];
    else
        snrs_plot  = snrs;
    end
    n_pad = numel(snrs_plot) - numel(snrs);

    % Helper: trim right, then prepend MI=0 padding
    n_snrs_orig = numel(snrs_src);
    if snr_cut_right_pts > 0
        trim_mi = @(b) b(1 : n_snrs_orig - snr_cut_right_pts);
    else
        trim_mi = @(b) b(:)';
    end
    pad_mi = @(b) [zeros(1, n_pad), trim_mi(b)];

    n_total     = numel(snrs_plot);
    mk_indices  = 1 : 1 : n_total;

    lbl_base     = [alg_name, cr_str];
    lbl_base_aug = [alg_name, ' aug', cr_str];

    mk_no_aug = markers_no_aug{alg};
    mk_aug    = markers_aug{alg};

    if is_fillable
        face_no_aug = 'none';
        face_aug    = alg_color;
    else
        face_no_aug = 'none';
        face_aug    = 'none';
    end

    % --- No-aug curve (dashed, hollow) ---
    if ~isempty(mi_no_aug_vec)
        h1 = plot(snrs_plot, pad_mi(mi_no_aug_vec), ...
            'Color',           alg_color, ...
            'LineStyle',       '--', ...
            'Marker',          mk_no_aug, ...
            'MarkerIndices',   mk_indices, ...
            'MarkerFaceColor', face_no_aug, ...
            'MarkerSize',      5, ...
            'LineWidth',       1.2);
        handles(end+1) = h1; %#ok<AGROW>
        labels{end+1}  = lbl_base; %#ok<AGROW>
    end

    % --- Aug iteration 1 curve (solid, filled) ---
    if ~isempty(mi_aug_1_vec)
        h2 = plot(snrs_plot, pad_mi(mi_aug_1_vec), ...
            'Color',           alg_color, ...
            'LineStyle',       '-', ...
            'Marker',          mk_aug, ...
            'MarkerIndices',   mk_indices, ...
            'MarkerFaceColor', face_aug, ...
            'MarkerSize',      5, ...
            'LineWidth',       1.2);
        handles(end+1) = h2; %#ok<AGROW>
        labels{end+1}  = lbl_base_aug; %#ok<AGROW>
    end

    % --- Aug iteration 2 curve (dotted, optional) ---
    if ~isempty(mi_aug_2_vec)
        h3 = plot(snrs_plot, pad_mi(mi_aug_2_vec), ...
            'Color',           alg_color, ...
            'LineStyle',       ':', ...
            'Marker',          mk_aug, ...
            'MarkerIndices',   mk_indices, ...
            'MarkerFaceColor', face_aug, ...
            'MarkerSize',      5, ...
            'LineWidth',       1.2);
        handles(end+1) = h3; %#ok<AGROW>
        labels{end+1}  = [lbl_base_aug, ' iter2']; %#ok<AGROW>
    end
end
end
