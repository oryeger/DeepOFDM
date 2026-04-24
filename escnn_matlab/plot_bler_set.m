function [handles, labels] = plot_bler_set( ...
    dir_path, ...
    algs_to_plot, alg_colors, alg_names, alg_files, ...
    markers_no_aug, markers_aug, fillable_algs, ...
    add_snr_target, plot_aug_iter_2, snr_pad_left_db, snr_cut_right_pts)
% PLOT_BLER_SET  Plot one set of BLER curves from a single directory.
%
%   snr_pad_left_db   : extend SNR axis left by this many dB with BLER=1 padding
%   snr_cut_right_pts : remove this many points from the right of SNR/BLER vectors
%
%   Encoding:
%     Dashed + hollow marker = no aug  (LMMSE/Sphere)
%     Solid  + filled marker = aug     (LMMSE/Sphere)
%     Dashed                 = no aug  (DeepRx/DeepSIC)
%     Solid                  = aug     (DeepRx/DeepSIC)

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
        warning('plot_bler_set: file not found, skipping: %s', mat_file);
        continue;
    end

    try
        S = load(mat_file);
    catch ME
        warning('plot_bler_set: could not load %s: %s', mat_file, ME.message);
        continue;
    end

    alg_name    = alg_names{alg};
    alg_color   = alg_colors(alg,:);
    is_fillable = ismember(alg, fillable_algs);

    % Build SNR vector with optional left padding of BLER=1 points
    snrs = S.snrs(:)';
    if snr_cut_right_pts > 0
        snrs = snrs(1 : end - snr_cut_right_pts);
    end
    if snr_pad_left_db > 0
        snr_step   = snrs(2) - snrs(1);           % infer step from data
        snr_start  = snrs(1) - snr_pad_left_db;
        snrs_pad   = snr_start : snr_step : snrs(1) - snr_step;
        snrs_plot  = [snrs_pad, snrs];
    else
        snrs_plot  = snrs;
    end
    n_pad = numel(snrs_plot) - numel(snrs);        % number of padded points

    % Helper: trim right, then prepend BLER=1 padding to a BLER vector
    n_snrs_orig = numel(S.snrs);
    if snr_cut_right_pts > 0
        trim_bler = @(b) b(1 : n_snrs_orig - snr_cut_right_pts);
    else
        trim_bler = @(b) b(:)';
    end
    pad_bler = @(b) [ones(1, n_pad), trim_bler(b)];

    % Marker spacing: one marker every ~4 points across full SNR range
    n_total     = numel(snrs_plot);
    mk_indices  = 1 : 1 : n_total;

    % Legend labels
    lbl_base     = [alg_name, cr_str];
    lbl_base_aug = [alg_name, ' aug', cr_str];

    % Marker shapes
    mk_no_aug = markers_no_aug{alg};
    mk_aug    = markers_aug{alg};

    % Fill colors
    if is_fillable
        face_no_aug = 'none';
        face_aug    = alg_color;
    else
        face_no_aug = 'none';
        face_aug    = 'none';
    end

    % --- No-aug curve (dashed, hollow) ---
    if add_snr_target && isfield(S, 'snr_target_no_aug')
        lbl_no_aug = [lbl_base, ', SNR@10%=', num2str(S.snr_target_no_aug)];
    else
        lbl_no_aug = lbl_base;
    end

    h1 = semilogy(snrs_plot, pad_bler(S.bler_no_aug), ...
        'Color',           alg_color, ...
        'LineStyle',       '--', ...
        'Marker',          mk_no_aug, ...
        'MarkerIndices',   mk_indices, ...
        'MarkerFaceColor', face_no_aug, ...
        'MarkerSize',      5, ...
        'LineWidth',       1.2);
    handles(end+1) = h1; %#ok<AGROW>
    labels{end+1}  = lbl_no_aug; %#ok<AGROW>

    % --- Aug iteration 1 curve (solid, filled) ---
    if ~isfield(S, 'bler_aug_1') && isfield(S, 'bler_aug')
        S.bler_aug_1 = S.bler_aug;
    end
    if isfield(S, 'bler_aug_1')
        if add_snr_target && isfield(S, 'snr_target_aug_1')
            lbl_aug1 = [lbl_base_aug, ', SNR@10%=', num2str(S.snr_target_aug_1)];
        else
            lbl_aug1 = lbl_base_aug;
        end

        h2 = semilogy(snrs_plot, pad_bler(S.bler_aug_1), ...
            'Color',           alg_color, ...
            'LineStyle',       '-', ...
            'Marker',          mk_aug, ...
            'MarkerIndices',   mk_indices, ...
            'MarkerFaceColor', face_aug, ...
            'MarkerSize',      5, ...
            'LineWidth',       1.2);
        handles(end+1) = h2; %#ok<AGROW>
        labels{end+1}  = lbl_aug1; %#ok<AGROW>
    end

    % --- Aug iteration 2 curve (dotted, optional) ---
    if plot_aug_iter_2 && isfield(S, 'bler_aug_2')
        if add_snr_target && isfield(S, 'snr_target_aug_2')
            lbl_aug2 = [lbl_base_aug, ' iter2, SNR@10%=', num2str(S.snr_target_aug_2)];
        else
            lbl_aug2 = [lbl_base_aug, ' iter2'];
        end

        h3 = semilogy(snrs_plot, pad_bler(S.bler_aug_2), ...
            'Color',           alg_color, ...
            'LineStyle',       ':', ...
            'Marker',          mk_aug, ...
            'MarkerIndices',   mk_indices, ...
            'MarkerFaceColor', face_aug, ...
            'MarkerSize',      5, ...
            'LineWidth',       1.2);
        handles(end+1) = h3; %#ok<AGROW>
        labels{end+1}  = lbl_aug2; %#ok<AGROW>
    end
end

% Pin x-axis to the full SNR grid so BLER=0 points (dropped by log scale)
% don't cause auto-scaling to clip the right edge.
if exist('snrs_plot', 'var') && ~isempty(snrs_plot)
    xlim([min(snrs_plot), max(snrs_plot)]);
end
end
