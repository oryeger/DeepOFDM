% =========================================================
% master_plot_mi_bler.m  -  MI (left) + BLER (right) side-by-side
%                           for a single run (one code rate)
% =========================================================

clear; clc;

% ---- User configuration ----
base_name        = 'CFO_MI';
extra_text       = '';             % e.g. '_transfer'
root_dir         = 'C:\Projects\Scratchpad\mat_files\';

algs_to_plot     = [1 2 3 4];     % 1=LMMSE, 2=SPHERE, 3=DeepRx, 4=DeepSIC
add_snr_target   = false;         % append SNR@10% to BLER legend labels
plot_aug_iter_2  = false;         % plot second aug iteration if available
snr_pad_left_db   = 0;            % extend SNR axis to the left by this many dB (0 = no padding)
snr_cut_right_pts = 0;            % cut this many SNR points from the right (0 = no cut)
snr_cut_left_pts  = 0;            % cut this many SNR points from the left  (0 = no cut)
output_target    = 'paper';       % 'paper' (compact, default legend) or 'ppt' (large, reordered legend, fontsize 14, PNG export)
% ----------------------------

is_ppt = strcmpi(output_target, 'ppt');

% ---- Auto-detect directories: code-rate variants OR TDL-channel variants ----
dirs     = {};
set_keys = {};
set_type = '';

% Code-rate suffix (_0.XX)
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
    [~, sort_idx] = sort(cellfun(@str2double, set_keys));
    dirs     = dirs(sort_idx);
    set_keys = set_keys(sort_idx);
end

% TDL-channel suffix (_TDLA / _TDLB / ...)
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
        [~, sort_idx] = sort(set_keys);
        dirs     = dirs(sort_idx);
        set_keys = set_keys(sort_idx);
    end
end

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

if numel(dirs) > 1
    warning('Multiple directories found; using first: %s', dirs{1});
end
dir_path = dirs{1};
sk_first = set_keys{1};

% TDL channel -> default delay spread (ns)
tdl_ds_keys   = {'TDLA', 'TDLB', 'TDLC', 'TDLD', 'TDLE'};
tdl_ds_values = [   30,    100,    300,    300,    300];

fprintf('Plotting MI + BLER from: %s\n', dir_path);
% -------------------------------------------

% ---- Shared style definitions ----
alg_colors = [0.00, 0.45, 0.70;   % LMMSE   - blue
              0.85, 0.33, 0.10;   % SPHERE  - vermillion
              0.47, 0.67, 0.19;   % DeepRx  - green
              0.49, 0.18, 0.56];  % DeepSIC - purple

alg_names = {'LMMSE', 'SPHERE', 'DeepRx', 'DeepSIC'};
alg_files = {'lmmse', 'sphere', 'deeprx', 'deepsic'};

markers_no_aug = {'^'; 's'; 'o'; 'd'};
markers_aug    = {'^'; 's'; 'o'; 'd'};

fillable_algs = [1:4];
% ----------------------------------

% ---- Extract code rate for titles ----
tok = regexp(dir_path, '_(0\.\d+)', 'tokens', 'once');
if ~isempty(tok)
    subplot_title = ['r = ', tok{1}];
else
    subplot_title = base_name;
end

% ---- Create figure: MI left, BLER right ----
fig = figure;
if is_ppt
    set(fig, 'Units', 'inches', 'Position', [0 0 14 7]);
else
    % Slightly taller than the BLER-only paper figure (3.5") because this
    % layout always carries a 2-row legend (no-aug row + aug row).
    set(fig, 'Units', 'inches', 'Position', [0 0 7 4.2]);
end

ax = gobjects(1, 2);

% ---- Left: MI (linear y) ----
ax(1) = subplot(1, 2, 1);
hold on; grid on;

[h_mi, lbl_mi] = plot_mi_set( ...
    dir_path, ...
    algs_to_plot, alg_colors, alg_names, alg_files, ...
    markers_no_aug, markers_aug, fillable_algs, ...
    plot_aug_iter_2, snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts);

hold off;
xlabel(ax(1), 'SNR (dB)');
ylabel(ax(1), 'MI');
ylim(ax(1), [0, 1]);
if is_ppt
    t1 = title(ax(1), 'Bit-wise Mutual Information');
    t1.Units = 'normalized';
    t1.Position(2) = t1.Position(2) + 0.03;
else
    title(ax(1), 'Bit-wise Mutual Information');
end
set(ax(1), 'YMinorTick', 'on', 'Box', 'on');

% ---- Right: BLER (log y) ----
ax(2) = subplot(1, 2, 2);
hold on; grid on;

[h_bler, lbl_bler] = plot_bler_set( ...
    dir_path, ...
    algs_to_plot, alg_colors, alg_names, alg_files, ...
    markers_no_aug, markers_aug, fillable_algs, ...
    add_snr_target, plot_aug_iter_2, snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts);

hold off;
xlabel(ax(2), 'SNR (dB)');
ylabel(ax(2), 'BLER');
if is_ppt
    t2 = title(ax(2), 'Block Error Rate');
    t2.Units = 'normalized';
    t2.Position(2) = t2.Position(2) + 0.03;
else
    title(ax(2), 'Block Error Rate');
end
set(ax(2), 'YScale', 'log', 'YMinorTick', 'on', 'Box', 'on');

% Match MI x-axis to BLER's full SNR span so both subplots line up at SNR=0.
xlim(ax(1), xlim(ax(2)));

% ---- Shared legend below (use BLER labels; strip code-rate suffix) ----
legend_labels = regexprep(lbl_bler, ',?\s*r=0\.\d+', '');

% Reorder so row 1 = all no-aug, row 2 = all aug, each with the requested alg order.
desired_order = {'LMMSE',     'SPHERE',     'DeepSIC',     'DeepRx', ...
                 'LMMSE aug', 'SPHERE aug', 'DeepSIC aug', 'DeepRx aug'};
reorder_idx = [];
for k = 1:numel(desired_order)
    idx = find(strcmp(legend_labels, desired_order{k}), 1);
    if ~isempty(idx)
        reorder_idx(end+1) = idx; %#ok<AGROW>
    end
end
h_bler        = h_bler(reorder_idx);
legend_labels = legend_labels(reorder_idx);

legend_args = {'Interpreter', 'none', 'Orientation', 'horizontal', 'NumColumns', 4};
if is_ppt
    legend_args = [legend_args, {'FontSize', 14}];
end
lgd = legend(ax(2), h_bler, legend_labels, legend_args{:});

lgd.Units       = 'normalized';
lgd.Position(1) = 0.5 - lgd.Position(3)/2;
lgd.Position(2) = 0.01;

% Shrink subplots to make room for the legend + xlabel
if is_ppt
    shrink_amt = 0.18;
else
    shrink_amt = 0.14;
end
for d = 1:2
    pos = ax(d).Position;
    ax(d).Position = [pos(1), pos(2)+shrink_amt, pos(3), pos(4)-shrink_amt];
end

% ---- Export ----
out_name = fullfile(root_dir, [base_name, extra_text, '_mi_bler']);
print(fig, [out_name, '.eps'], '-depsc', '-painters');
if is_ppt
    print(fig, [out_name, '.png'], '-dpng',  '-r600');   % high-DPI raster for PPT
end
savefig([out_name, '.fig']);
