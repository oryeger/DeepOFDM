% =========================================================
% master_plot_bler.m  –  configure here, then run
% =========================================================

clear; clc;

% ---- User configuration ----
base_name        = 'Completion4x4';
extra_text       = '';             % e.g. '_transfer'
root_dir         = 'C:\Projects\Scratchpad\mat_files\';

algs_to_plot     = [1 2];         % 1=LMMSE, 2=RBSD, 3=DeepRx, 4=DeepSIC
add_snr_target   = false;         % append SNR@10% to legend labels
plot_aug_iter_2  = false;         % plot second aug iteration if available
snr_pad_left_db   = 0;            % extend SNR axis to the left by this many dB (0 = no padding)
snr_cut_right_pts = 5;            % cut this many SNR points from the right (0 = no cut)
% ----------------------------

% ---- Auto-detect directories: code-rate variants OR TDL-channel variants ----
% set_type is 'cr' (e.g. base_0.6 / base_0.7) or 'channel' (e.g. base_TDLB / base_TDLC).
dirs     = {};
set_keys = {};
set_type = '';

% First try code-rate suffix (_0.XX)
candidates = dir(fullfile(root_dir, [base_name, '_0.*']));
candidates = candidates([candidates.isdir]);
for i = 1:numel(candidates)
    fname = candidates(i).name;
    tok = regexp(fname, ...
        [regexptranslate('escape', base_name), '_(0\.\d+)', ...
         regexptranslate('escape', extra_text), '$'], 'tokens', 'once');
    if ~isempty(tok)
        dirs{end+1}     = fullfile(root_dir, fname); %#ok<SAGROW>
        set_keys{end+1} = tok{1};                    %#ok<SAGROW>
    end
end
if ~isempty(dirs)
    set_type = 'cr';
    cr_vals  = cellfun(@str2double, set_keys);
    [~, sort_idx] = sort(cr_vals);
    dirs     = dirs(sort_idx);
    set_keys = set_keys(sort_idx);
end

% Else try TDL-channel suffix (_TDLA / _TDLB / _TDLC / ...)
if isempty(dirs)
    candidates = dir(fullfile(root_dir, [base_name, '_TDL*']));
    candidates = candidates([candidates.isdir]);
    for i = 1:numel(candidates)
        fname = candidates(i).name;
        tok = regexp(fname, ...
            [regexptranslate('escape', base_name), '_(TDL[A-Z])', ...
             regexptranslate('escape', extra_text), '$'], 'tokens', 'once');
        if ~isempty(tok)
            dirs{end+1}     = fullfile(root_dir, fname); %#ok<SAGROW>
            set_keys{end+1} = tok{1};                    %#ok<SAGROW>
        end
    end
    if ~isempty(dirs)
        set_type = 'channel';
        [~, sort_idx] = sort(set_keys);   % alphabetical: TDLA, TDLB, TDLC
        dirs     = dirs(sort_idx);
        set_keys = set_keys(sort_idx);
    end
end

% Fall back to bare base_name if no suffixed dirs found
if isempty(dirs)
    candidate = fullfile(root_dir, [base_name, extra_text]);
    if isfolder(candidate)
        dirs     = {candidate};
        set_keys = {''};
        set_type = '';
    else
        error('No valid directory found for base_name="%s"', base_name);
    end
end

% Limit to 2 sets max
if numel(dirs) > 2
    warning('More than 2 directories found, using first 2 only.');
    dirs     = dirs(1:2);
    set_keys = set_keys(1:2);
end

% TDL channel -> default delay spread (ns), matches python_code/utils/config_singleton.py
tdl_ds_keys   = {'TDLA', 'TDLB', 'TDLC', 'TDLD', 'TDLE'};
tdl_ds_values = [   30,    100,    300,    300,    300];

fprintf('Found %d curve set(s):\n', numel(dirs));
for i = 1:numel(dirs)
    fprintf('  %s\n', dirs{i});
end
% -------------------------------------------

% ---- Shared style definitions ----
alg_colors = [0.00, 0.45, 0.70;   % LMMSE   - blue
              0.85, 0.33, 0.10;   % RBSD    - vermillion
              0.47, 0.67, 0.19;   % DeepRx  - green
              0.49, 0.18, 0.56];  % DeepSIC - purple

alg_names = {'LMMSE', 'RBSD', 'DeepRx', 'DeepSIC'};
alg_files = {'lmmse', 'sphere', 'deeprx', 'deepsic'};

markers_no_aug = {'^';    % LMMSE   (hollow)
                  's';    % RBSD    (hollow)
                  'o';    % DeepRx  (hollow)
                  'd'};   % DeepSIC (hollow)

markers_aug    = {'^';    % LMMSE   (filled)
                  's';    % RBSD    (filled)
                  'o';    % DeepRx  (filled)
                  'd'};   % DeepSIC (filled)

fillable_algs = [1:4];
% ----------------------------------

% ---- Create figure ----
n_dirs = numel(dirs);
fig = figure;

if n_dirs == 1
    % ---- Single code rate: compact figure with internal legend ----
    set(fig, 'Units', 'inches', 'Position', [0 0 5 3.5]);

    ax(1) = axes; %#ok<LAXES>
    hold on; grid on;

    [all_handles, all_labels] = plot_bler_set( ...
        dirs{1}, ...
        algs_to_plot, alg_colors, alg_names, alg_files, ...
        markers_no_aug, markers_aug, fillable_algs, ...
        add_snr_target, plot_aug_iter_2, snr_pad_left_db, snr_cut_right_pts);

    hold off;

    % Build title from set_type / set_keys
    sk = set_keys{1};
    if strcmp(set_type, 'cr')
        subplot_title = ['r = ', sk];
    elseif strcmp(set_type, 'channel')
        ds_idx = find(strcmp(tdl_ds_keys, sk), 1);
        if ~isempty(ds_idx)
            subplot_title = sprintf('TDL-%s, delay spread=%dns', sk(end), tdl_ds_values(ds_idx));
        else
            subplot_title = ['TDL-', sk(end)];
        end
    else
        subplot_title = '';
    end
    if ~isempty(subplot_title)
        t1 = title(ax(1), subplot_title);
        t1.Units = 'normalized';
        t1.Position(2) = t1.Position(2) + 0.03;
    end

    set(ax(1), 'YScale', 'log', 'YMinorTick', 'on', 'Box', 'on');
    xlabel(ax(1), 'SNR (dB)');
    ylabel(ax(1), 'BLER');

    % Reorder legend: all no-aug then all aug, in requested alg order.
    desired_order = {'LMMSE',     'RBSD',     'DeepSIC',     'DeepRx', ...
                     'LMMSE aug', 'RBSD aug', 'DeepSIC aug', 'DeepRx aug'};
    reorder_idx = [];
    for k = 1:numel(desired_order)
        idx = find(strcmp(all_labels, desired_order{k}), 1);
        if ~isempty(idx)
            reorder_idx(end+1) = idx; %#ok<AGROW>
        end
    end
    all_handles = all_handles(reorder_idx);
    all_labels  = all_labels(reorder_idx);

    % Internal legend, best location, vertical layout
    legend(ax(1), all_handles, all_labels, ...
        'Location',    'southwest', ...
        'Interpreter', 'none', ...
        'NumColumns',  1, ...
        'FontSize',    14);

    % Tight axes margins (matches old style)
    set(ax(1), 'LooseInset', get(ax(1), 'TightInset'));

else
    % ---- Two code rates: side-by-side subplots with shared legend below ----
    set(fig, 'Units', 'inches', 'Position', [0 0 14 7]);

    all_handles = [];
    all_labels  = {};
    ax = gobjects(1, n_dirs);

    for d = 1:n_dirs
        ax(d) = subplot(1, n_dirs, d);
        hold on; grid on;

        [h, lbl] = plot_bler_set( ...
            dirs{d}, ...
            algs_to_plot, alg_colors, alg_names, alg_files, ...
            markers_no_aug, markers_aug, fillable_algs, ...
            add_snr_target, plot_aug_iter_2, snr_pad_left_db, snr_cut_right_pts);

        hold off;

        % Collect legend handles only from first subplot, strip code rate suffix
        if d == 1
            all_handles = h;
            % Remove ', r=X.XX' from labels since subplot title shows code rate
            all_labels  = regexprep(lbl, ',?\s*r=0\.\d+', '');
        end

        % Subplot title from set_type / set_keys
        sk = set_keys{d};
        if strcmp(set_type, 'cr')
            subplot_title = ['r = ', sk];
        elseif strcmp(set_type, 'channel')
            ds_idx = find(strcmp(tdl_ds_keys, sk), 1);
            if ~isempty(ds_idx)
                subplot_title = sprintf('TDL-%s, delay spread=%dns', sk(end), tdl_ds_values(ds_idx));
            else
                subplot_title = ['TDL-', sk(end)];
            end
        else
            subplot_title = base_name;
        end
        t_sp = title(ax(d), subplot_title);
        t_sp.Units = 'normalized';
        t_sp.Position(2) = t_sp.Position(2) + 0.03;

        set(ax(d), 'YScale', 'log');
        xlabel(ax(d), 'SNR (dB)');

        if d == 1
            ylabel(ax(d), 'BLER');
            set(ax(d), 'YAxisLocation', 'left');
        else
            linkaxes([ax(1), ax(d)], 'y');
            ylabel(ax(d), 'BLER');
        end

        set(ax(d), 'YMinorTick', 'on', 'Box', 'on');
    end

    % Reorder legend: row 1 = all no-aug, row 2 = all aug, in requested alg order.
    desired_order = {'LMMSE',     'RBSD',     'DeepSIC',     'DeepRx', ...
                     'LMMSE aug', 'RBSD aug', 'DeepSIC aug', 'DeepRx aug'};
    reorder_idx = [];
    for k = 1:numel(desired_order)
        idx = find(strcmp(all_labels, desired_order{k}), 1);
        if ~isempty(idx)
            reorder_idx(end+1) = idx; %#ok<AGROW>
        end
    end
    all_handles = all_handles(reorder_idx);
    all_labels  = all_labels(reorder_idx);

    % Shared legend below the subplots
    lgd = legend(ax(1), all_handles, all_labels, ...
        'Interpreter', 'none', ...
        'Orientation', 'horizontal', ...
        'NumColumns',  4, ...
        'FontSize',    14);

    lgd.Units       = 'normalized';
    lgd.Position(1) = 0.5 - lgd.Position(3)/2;   % center horizontally
    lgd.Position(2) = 0.01;                        % near bottom

    % Shrink subplots to make room for the (larger) legend + xlabel at bottom
    for d = 1:n_dirs
        pos = ax(d).Position;
        ax(d).Position = [pos(1), pos(2)+0.18, pos(3), pos(4)-0.18];
    end
end
% ylim(ax(1), [0.0002, 1])

% ---- Export ----
out_name = fullfile(root_dir, [base_name, extra_text]);
print(fig, [out_name, '.eps'], '-depsc', '-painters');
print(fig, [out_name, '.png'], '-dpng',  '-r600');       % high-DPI raster for PPT
savefig([out_name, '.fig']);
