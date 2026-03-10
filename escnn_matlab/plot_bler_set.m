function [handles, labels] = plot_bler_set( ...
    dir_path, rc_idx, ...
    algs_to_plot, alg_colors, alg_names, alg_files, ...
    markers_no_aug, markers_aug, fillable_algs, ...
    add_snr_target, plot_aug_iter_2)
% PLOT_BLER_SET  Plot one set of BLER curves from a single directory.
%
%   rc_idx : 1 or 2, indexes into the marker tables for this code-rate set
%
%   Encoding:
%     Dashed + hollow marker = no aug  (LMMSE/Sphere)
%     Solid  + filled marker = aug     (LMMSE/Sphere)
%     Dashed                 = no aug  (DeepRx/DeepSIC, non-fillable markers)
%     Solid                  = aug     (DeepRx/DeepSIC, non-fillable markers)

% Extract code rate label from directory path
token = regexp(dir_path, '_(0\.\d+)', 'tokens', 'once');
if ~isempty(token)
    cr_str = [', r=', token{1}];   % e.g. ' r=0.46'
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

    alg_name  = alg_names{alg};
    alg_color = alg_colors(alg,:);
    is_fillable = ismember(alg, fillable_algs);

    % Legend labels
    lbl_base     = [alg_name, cr_str];
    lbl_base_aug = [alg_name, ' aug', cr_str];

    % Marker shapes for this alg + code rate set
    mk_no_aug = markers_no_aug{alg, rc_idx};
    mk_aug    = markers_aug{alg, rc_idx};

    % Face colors: hollow for no-aug, filled for aug (fillable algs only)
    if is_fillable
        face_no_aug = 'none';       % hollow
        face_aug    = alg_color;    % filled
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

    h1 = semilogy(S.snrs, S.bler_no_aug, ...
        'Color',           alg_color, ...
        'LineStyle',       '--', ...
        'Marker',          mk_no_aug, ...
        'MarkerFaceColor', face_no_aug, ...
        'LineWidth',       1.2);
    handles(end+1) = h1; %#ok<AGROW>
    labels{end+1}  = lbl_no_aug; %#ok<AGROW>

    % --- Aug iteration 1 curve (solid, filled) ---
    if isfield(S, 'bler_aug_1')
        if add_snr_target && isfield(S, 'snr_target_aug_1')
            lbl_aug1 = [lbl_base_aug, ', SNR@10%=', num2str(S.snr_target_aug_1)];
        else
            lbl_aug1 = lbl_base_aug;
        end

        h2 = semilogy(S.snrs, S.bler_aug_1, ...
            'Color',           alg_color, ...
            'LineStyle',       '-', ...
            'Marker',          mk_aug, ...
            'MarkerFaceColor', face_aug, ...
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

        h3 = semilogy(S.snrs, S.bler_aug_2, ...
            'Color',           alg_color, ...
            'LineStyle',       ':', ...
            'Marker',          mk_aug, ...
            'MarkerFaceColor', face_aug, ...
            'LineWidth',       1.2);
        handles(end+1) = h3; %#ok<AGROW>
        labels{end+1}  = lbl_aug2; %#ok<AGROW>
    end
end
end
