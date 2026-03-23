% =========================================================
% master_plot_bler.m  –  configure here, then run
% =========================================================

clear; clc; close all;

% ---- User configuration ----
base_name        = 'CLEAN_MED';
extra_text       = '';             % e.g. '_transfer'
root_dir         = 'C:\Projects\Scratchpad\mat_files\';

algs_to_plot     = [1 2 3 4];         % 1=LMMSE, 2=RBSD, 3=DeepRx, 4=DeepSIC
add_snr_target   = false;         % append SNR@10% to legend labels
plot_aug_iter_2  = false;         % plot second aug iteration if available
snr_pad_left_db   = 0;            % extend SNR axis to the left by this many dB (0 = no padding)
snr_cut_right_pts = 5;            % cut this many SNR points from the right (0 = no cut)
% ----------------------------

% ---- Auto-detect code rate directories ----
pattern    = fullfile(root_dir, [base_name, '_0.*']);
candidates = dir(pattern);
candidates = candidates([candidates.isdir]);

dirs = {};
for i = 1:numel(candidates)
    fname = candidates(i).name;
    expected_suffix = regexp(fname, ...
        [regexptranslate('escape', base_name), '_(0\.\d+)', ...
         regexptranslate('escape', extra_text), '$'], 'tokens', 'once');
    if ~isempty(expected_suffix)
        dirs{end+1} = fullfile(root_dir, fname); %#ok<SAGROW>
    end
end

% Sort by code rate ascending
cr_vals = zeros(1, numel(dirs));
for i = 1:numel(dirs)
    tok = regexp(dirs{i}, '_(0\.\d+)', 'tokens', 'once');
    cr_vals(i) = str2double(tok{1});
end
[~, sort_idx] = sort(cr_vals);
dirs = dirs(sort_idx);

% Fall back to bare base_name if no suffixed dirs found
if isempty(dirs)
    candidate = fullfile(root_dir, [base_name, extra_text]);
    if isfolder(candidate)
        dirs = {candidate};
    else
        error('No valid directory found for base_name="%s"', base_name);
    end
end

% Limit to 2 sets max
if numel(dirs) > 2
    warning('More than 2 code rate directories found, using first 2 only.');
    dirs = dirs(1:2);
end

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

alg_names = {'LMMSE', 'SPHERE', 'DeepRx', 'DeepSIC'};
alg_files = {'lmmse', 'sphere', 'deeprx', 'deepsic'};

markers_no_aug = {'^';    % LMMSE   (fillable, hollow)
                  's';    % RBSD    (fillable, hollow)
                  '*';    % DeepRx  (non-fillable)
                  '+'};   % DeepSIC (non-fillable)

markers_aug    = {'^';    % LMMSE   (fillable, filled)
                  's';    % RBSD    (fillable, filled)
                  'p';    % DeepRx  (non-fillable)
                  'x'};   % DeepSIC (non-fillable)

fillable_algs = [1, 2];
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

    % Extract code rate for title (if available)
    tok = regexp(dirs{1}, '_(0\.\d+)', 'tokens', 'once');
    if ~isempty(tok)
        title(ax(1), ['r = ', tok{1}]);
    end

    set(ax(1), 'YScale', 'log', 'YMinorTick', 'on', 'Box', 'on');
    xlabel(ax(1), 'SNR (dB)');
    ylabel(ax(1), 'BLER');

    % Internal legend, best location, vertical layout
    legend(ax(1), all_handles, all_labels, ...
        'Location',    'southwest', ...
        'Interpreter', 'none', ...
        'NumColumns',  1);

    % Tight axes margins (matches old style)
    set(ax(1), 'LooseInset', get(ax(1), 'TightInset'));

else
    % ---- Two code rates: side-by-side subplots with shared legend below ----
    set(fig, 'Units', 'inches', 'Position', [0 0 7 3.5]);

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

        % Extract code rate for subplot title
        tok = regexp(dirs{d}, '_(0\.\d+)', 'tokens', 'once');
        if ~isempty(tok)
            subplot_title = ['r = ', tok{1}];
        else
            subplot_title = base_name;
        end
        title(ax(d), subplot_title);

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

    % Shared legend below the subplots
    lgd = legend(ax(1), all_handles, all_labels, ...
        'Interpreter', 'none', ...
        'Orientation', 'horizontal', ...
        'NumColumns',  4);

    lgd.Units       = 'normalized';
    lgd.Position(1) = 0.5 - lgd.Position(3)/2;   % center horizontally
    lgd.Position(2) = 0.01;                        % near bottom

    % Shrink subplots slightly to make room for legend at bottom
    for d = 1:n_dirs
        pos = ax(d).Position;
        ax(d).Position = [pos(1), pos(2)+0.08, pos(3), pos(4)-0.08];
    end
end
% ylim(ax(1), [0.0002, 1])

% ---- Export ----
out_name = fullfile(root_dir, [base_name, extra_text]);
print(fig, [out_name, '.eps'], '-depsc', '-painters');
savefig([out_name, '.fig']);
