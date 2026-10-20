% =========================================================
% master_plot_trk_bler_paper.m  -  BLER-only paper figure (IEEE single
%                                  column) comparing tracking modes,
%                                  seed-averaged.
%                                  Two folders -> two BLER subplots side by
%                                  side with one joint legend below
%                                  (layout as in master_plot_mi_bler.m).
% =========================================================
%
% Usage: run build_analysis.py <tag> first, then set analysis_dir below to the
% name of the Analysis folder it produced (e.g. 'itrknk', or 'itrknk_R=0.48_LMMSE' - found under
% its parent Analysis\itrknk\ - for a tag that build_analysis split per code rate / primary) and run.
%
% analysis_dir = 'idsseed'                -> one figure per table (unchanged behavior)
% analysis_dir = {'ilmseed', 'idsseed'}   -> side-by-side figure: left = 1st folder, right = 2nd,
%                                            subplot titles from `titles`. Table i of the 1st folder
%                                            is paired with table i of the 2nd (compare_avg_<i>.csv).
%                                            Saved to <1st folder>\paper\<dir1>_vs_<dir2>_<table>_paper.*
%
% For every seed-averaged comparison table build_analysis wrote in that folder
% (compare_avg_<i>.csv - one per configuration run with several tracking modes),
% one paper figure is saved to <analysis_dir>\paper\compare_avg_<i>_paper.{eps,pdf,png,fig}.
% If the folder has no seed-averaged tables (single-seed runs), the per-seed
% compare_<i>.csv tables are used instead.
%
% Only the modes listed in `modes` are drawn (by column name, in that order); the
% reference (primary receiver) is the table's 2nd column.
%
% Seeds (recompute = true, the default): build_analysis' compare_avg_<i>.csv averages only
% over seeds complete for ALL modes of that section, so a mode you don't draw (e.g. ekfbp)
% can cost seeds. With recompute = true, each table is re-averaged first by
% compare_track_modes.py for exactly `modes` (and `seeds`, [] = all), using the
% configuration recorded in the table's seeds note - so only the drawn modes decide which
% seeds are dropped. The re-averaged table and figure go to <analysis_dir>\paper\
% (compare_avg_<i>_s=<seeds|all>_paper.*). recompute = false plots the build_analysis
% tables as they are. The seeds note printed for each figure says which seeds were used.

clear; clc;

% ---- User configuration ----
analysis_dir = {'ilmseed', 'idsseed'};   % one folder (char) or two folders (cell) under C:\Projects\Scratchpad\Analysis (or full paths)
titles     = {'LMMSE', 'DeepSIC'};       % side-by-side only: subplot titles; {} = the reference name from each CSV
%analysis_dir = {'itrknk'};   % one folder (char) or two folders (cell) under C:\Projects\Scratchpad\Analysis (or full paths)
modes      = {'notrack', 'ekfcrc', 'ekfbp', 'ekfht', 'sgdht', 'ekfit'};   % legend order (ref is drawn first)
%seeds      = [];          % [] = every seed complete for `modes`; or e.g. [58 123 1011 1505]
seeds      = [17 123 1011 1505 3008];          % [] = every seed complete for `modes`; or e.g. [58 123 1011 1505]
recompute  = true;        % true = re-average for `modes`/`seeds` (see above); false = tables as is
zipfile    = '';          % recompute only: '' = loose result CSVs in Scratchpad, else the archive name
                          % (side-by-side: '' lets each folder find its own Scratchpad\<tag>.gz)
deepofdm_root = 'C:\Projects\DeepOFDM';
python_exe = 'python';
ref_label  = '';          % '' = take it from the CSV ('LMMSE' / 'DeepSIC'); or e.g. 'LMMSE'
fig_size   = [3.5, 2.5];  % inches (IEEE column = 3.5)
legend_loc = 'southwest'; % inside the axes; the low-SNR / low-BLER corner is empty in these plots
legend_cols = 1;
y_min      = 2e-3;        % bottom of the BLER axis; [] = automatic (the decade below the lowest point)
bler_min_drawn = 2e-3;    % points below this are not drawn - the curve drops off the axis at its
                          % first such point (like BLER = 0), so isolated one- or two-error points
                          % in the tail disappear. [] = draw every point (default: y_min)
clip_y     = true;        % true = apply y_min / bler_min_drawn above; false = automatic axis
                          % bottom and every point drawn (both treated as [])

% ---- Side-by-side layout (two folders only) - drawn like master_plot_mi_bler.m: 7" wide,
% scaled to \columnwidth in LaTeX (or used as-is in a figure* spanning both columns) ----
PAIR.fig_size         = [7, 3.6];   % inches
PAIR.font_size        = 10;         % axes / title / labels
PAIR.legend_font_size = 9;          % use 8.5 with 4 legend columns (they nearly fill the 7" width at 9)
PAIR.legend_cols      = 3;          % 3 rows - see PAIR.order
PAIR.ref_label        = 'No aug.';   % the gray reference curve (LMMSE on the left, DeepSIC on the right)
% Pair-only legend labels (mode -> label); modes not listed keep their STYLE label.
PAIR.labels = containers.Map( ...
    {'notrack',           'ekfbp',          'ekfcrc',         'ekfht',              'sgdht',              'ekfit'}, ...
    {'Aug., no tracking', 'Aug., EKF-syn.', 'Aug., EKF-CRC',  'Aug., EKF-CRC+syn.', 'Aug., SGD-CRC+syn.', 'Aug., EKF-genie'});
% Pair-only legend order ('ref' = the reference, '' = an empty slot; legend fills row by row):
% row 1 = references / bound, row 2 = the label-free EKF trackers (EKF-CRC+syn. in the center),
% row 3 = the SGD baseline, centered under its EKF counterpart. Modes drawn but not listed are
% appended; {} = `modes` order.
PAIR.order  = {'ref', 'notrack', 'ekfit', 'ekfbp', 'ekfht', 'ekfcrc', '', 'sgdht'};
PAIR.bold   = {'ekfht'};        % legend entries drawn in bold (the proposed method)
PAIR.marker_size      = 5;
PAIR.lw_scale         = 1.5;        % multiplies the STYLE line widths (figure is shrunk ~2x in the paper)
PAIR.shrink           = 0.18;       % room left under the axes for the legend + xlabel (0.14 in master_plot_mi_bler, 2-row legend)
PAIR.same_ylim        = true;       % both subplots get the same BLER range
% ----------------------------
if ~clip_y
    y_min = [];
    bler_min_drawn = [];
end

% mode -> {legend, color, line style, marker, line width}
STYLE = containers.Map();
STYLE('notrack')  = {'No adaptation',          [0.00 0.00 0.00], '-',  'o', 0.9};
STYLE('ekfcrc')   = {'EKF, CRC',               [0.00 0.45 0.74], '-',  's', 0.9};
STYLE('ekfbp')    = {'EKF, syn.',              [0.93 0.49 0.09], '-',  '^', 0.9};
STYLE('ekfht')    = {'EKF, CRC+syn.',          [0.80 0.10 0.10], '-',  'p', 1.4};
STYLE('ekfit')    = {'EKF, genie',             [0.13 0.55 0.13], '-',  'd', 0.9};
STYLE('ekfis')    = {'EKF, genie (causal)',    [0.13 0.55 0.13], ':',  'd', 0.9};
STYLE('ekfi')     = {'EKF, genie',             [0.13 0.55 0.13], '-',  'd', 0.9};   % old genie, if ekfit not run yet
STYLE('ekfp')     = {'EKF, pilots',            [0.49 0.18 0.56], '-',  'v', 0.9};
STYLE('sgdht')    = {'SGD, CRC+syn.',          [0.80 0.10 0.10], '--', 'p', 0.9};
STYLE('sgdbceit') = {'SGD, genie',             [0.13 0.55 0.13], '--', 'd', 0.9};
STYLE('sgdbcei')  = {'SGD, genie',             [0.13 0.55 0.13], '--', 'd', 0.9};
REF_STYLE         = {[0.50 0.50 0.50], '-.', 'none', 0.9};
MARKER_SIZE       = 3;    % points ('p' is drawn 1 pt larger - it looks small otherwise)

FLOOR = 1e-12;   % stand-in for BLER = 0: the line drops off the bottom of the axis there

% ---- Find the tables build_analysis wrote (per folder) ----
if ischar(analysis_dir) || isstring(analysis_dir)
    analysis_dir = cellstr(analysis_dir);
end
n_dirs = numel(analysis_dir);
if n_dirs < 1 || n_dirs > 2
    error('master_plot_trk_bler_paper: analysis_dir must be one folder or a cell of two folders.');
end
if n_dirs == 2 && ~isempty(titles) && numel(titles) ~= 2
    error('master_plot_trk_bler_paper: titles must have two entries (or be {}).');
end

dir_paths = cell(1, n_dirs);
tables    = cell(1, n_dirs);
out_dirs  = cell(1, n_dirs);
for k = 1:n_dirs
    dir_paths{k} = resolve_dir(analysis_dir{k});
    tables{k}    = find_tables(dir_paths{k});
    out_dirs{k}  = fullfile(dir_paths{k}, 'paper');
    if ~isfolder(out_dirs{k})
        mkdir(out_dirs{k});
    end
end

n_tab = numel(tables{1});
if n_dirs == 2
    if numel(tables{2}) ~= n_tab
        warning(['master_plot_trk_bler_paper: %s has %d tables, %s has %d - pairing the first %d ', ...
                 'by index.'], dir_paths{1}, n_tab, dir_paths{2}, numel(tables{2}), ...
                min(n_tab, numel(tables{2})));
    end
    n_tab = min(n_tab, numel(tables{2}));
end

if isempty(seeds)
    seed_txt = 'all';
    seed_arg = '';
else
    seed_txt = strjoin(arrayfun(@num2str, seeds, 'UniformOutput', false), '-');
    seed_arg = [' --seeds ', strjoin(arrayfun(@num2str, seeds, 'UniformOutput', false), ' ')];
end

for t = 1:n_tab
    csv_paths = cell(1, n_dirs);
    out_names = cell(1, n_dirs);
    for k = 1:n_dirs
        [csv_paths{k}, out_names{k}] = prepare_csv(dir_paths{k}, tables{k}{t}, out_dirs{k}, recompute, ...
            modes, seeds, seed_txt, seed_arg, zipfile, deepofdm_root, python_exe);
    end
    if n_dirs == 1
        plot_one(csv_paths{1}, out_names{1}, modes, STYLE, REF_STYLE, MARKER_SIZE, FLOOR, ref_label, ...
                 fig_size, legend_loc, legend_cols, y_min, bler_min_drawn);
    else
        leaf = @(p) regexprep(p, '.*[\\/]', '');
        [~, nm] = fileparts(out_names{1});
        pair_name = fullfile(out_dirs{1}, [leaf(dir_paths{1}), '_vs_', leaf(dir_paths{2}), '_', nm]);
        plot_pair(csv_paths, pair_name, titles, modes, STYLE, REF_STYLE, FLOOR, PAIR, ...
                  y_min, bler_min_drawn);
    end
end

% =========================================================
function dir_path = resolve_dir(analysis_dir)
% Folder name under C:\Projects\Scratchpad\Analysis (or one level deeper), or a full path.
if contains(analysis_dir, ':')
    dir_path = analysis_dir;
else
    dir_path = fullfile('C:\Projects\Scratchpad\Analysis', analysis_dir);
    if ~isfolder(dir_path)
        % per-rate/primary builds sit under a parent tag folder (e.g. icpilot\icpilot_R=0.59)
        % (dir on a folder pattern lists that folder's contents, so .folder is the match)
        hit = dir(fullfile('C:\Projects\Scratchpad\Analysis', '*', analysis_dir));
        if ~isempty(hit)
            dir_path = hit(1).folder;
        end
    end
end
if ~isfolder(dir_path)
    error('master_plot_trk_bler_paper: no such folder: %s (run build_analysis.py first)', dir_path);
end
end

function tables = find_tables(dir_path)
tables = list_tables(dir_path, '^compare_avg_\d+\.csv$');
if isempty(tables)
    tables = list_tables(dir_path, '^compare_\d+\.csv$');
    if ~isempty(tables)
        fprintf('No seed-averaged tables in %s - using the per-seed compare_<i>.csv tables.\n', dir_path);
    end
end
if isempty(tables)
    error(['master_plot_trk_bler_paper: no compare_*.csv tables in %s - build_analysis.py writes ', ...
           'them only for configurations run with at least two tracking modes.'], dir_path);
end
end

function [csv_path, out_name] = prepare_csv(dir_path, table_name, out_dir, recompute, modes, seeds, ...
                                            seed_txt, seed_arg, zipfile, deepofdm_root, python_exe)
% The CSV to plot for one table (re-averaged by compare_track_modes.py if recompute) and the
% figure base name.
csv_path = fullfile(dir_path, table_name);
base     = table_name(1:end-4);
out_name = fullfile(out_dir, [base, '_paper']);
fprintf('\n==== %s ====\n', csv_path);
if ~recompute
    return;
end
cfg = read_config(csv_path);
if isempty(cfg)
    % Older table (no config line): fall back to the tag as the only filter token.
    dir_leaf = regexprep(dir_path, '.*[\\/]', '');
    cfg = strtok(dir_leaf, '_');
    fprintf(['(no config line in %s''s seeds note - filtering on the tag ''%s'' only; ', ...
             'rerun build_analysis.py for an exact match)\n'], table_name, cfg);
end
% config tokens -> --filter (the per-seed tables' s= token is not a config token)
toks = strsplit(cfg, '_');
seed_tok = toks(startsWith(toks, 's='));
toks = toks(~startsWith(toks, 's='));
this_arg = seed_arg;
if ~isempty(seed_tok) && isempty(seeds)   % per-seed table: keep its own seed
    this_arg = [' --seeds ', extractAfter(seed_tok{1}, 's=')];
end
out_png = fullfile(out_dir, [base, '_s=', seed_txt, '.png']);
cmd = sprintf(['cd /d "%s" && %s python_code\\utils\\compare_track_modes.py --modes %s ', ...
               '--filter %s%s --table --no-open --out "%s"'], deepofdm_root, python_exe, ...
              strjoin(modes, ' '), strjoin(toks, ' '), this_arg, out_png);
% Data source: zipfile if set; else the loose result CSVs, or - when the tag's loose
% CSVs are gone (archived) - its archive Scratchpad\<tag>.gz.
zf = zipfile;
tag = strtok(regexprep(dir_path, '.*[\\/]', ''), '_');
if isempty(zf) && isempty(dir(fullfile('C:\Projects\Scratchpad', ['*_', tag, '_s=*_SNR=*.csv']))) ...
        && isfile(fullfile('C:\Projects\Scratchpad', [tag, '.gz']))
    zf = tag;
    fprintf('no loose CSVs for %s - reading Scratchpad\\%s.gz\n', tag, tag);
end
if ~isempty(zf)
    cmd = [cmd, ' --zipfile ', zf];
end
[status, out_txt] = system(cmd);
if status ~= 0
    fprintf('%s\n', out_txt);
    error('master_plot_trk_bler_paper: compare_track_modes.py failed for %s (output above).', csv_path);
end
csv_path = strrep(out_png, '.png', '.csv');
out_name = fullfile(out_dir, [base, '_s=', seed_txt, '_paper']);
fprintf('re-averaged for %s -> %s\n', strjoin(modes, ' '), csv_path);
end

function cfg = read_config(csv_path)
% The 'config: <key>' line build_analysis writes into <csv>_seeds.txt, or '' if absent.
cfg = '';
note = [csv_path(1:end-4), '_seeds.txt'];
if ~isfile(note)
    return;
end
lines = splitlines(fileread(note));
hit = lines(startsWith(lines, 'config: '));
if ~isempty(hit)
    cfg = strtrim(extractAfter(hit{1}, 'config: '));
end
end

function names = list_tables(d, pattern)
% Table file names in folder d matching the regexp pattern, sorted by their index.
f = dir(fullfile(d, '*.csv'));
names = {f.name};
names = names(~cellfun(@isempty, regexp(names, pattern, 'once')));
idx = cellfun(@(n) str2double(regexp(n, '\d+(?=\.csv$)', 'match', 'once')), names);
[~, order] = sort(idx);
names = names(order);
end

function [handles, keys, labels, csv_ref] = draw_bler_axes(ax, csv_path, modes, STYLE, REF_STYLE, ...
        marker_size, lw_scale, FLOOR, font_size, y_min, bler_min_drawn)
% Draws the reference + `modes` BLER curves of one table into ax and formats the axes.
% handles(1) / keys{1} = 'ref' is the reference curve; the rest are the modes found in the CSV.
% labels are the STYLE legend names ('' for the reference - the caller names it).
seeds_note = [csv_path(1:end-4), '_seeds.txt'];
% analysis folder the table belongs to (re-averaged tables sit in its paper\ subfolder)
[tbl_dir, ~] = fileparts(csv_path);
[up_dir, tbl_name] = fileparts(tbl_dir);
if strcmp(tbl_name, 'paper')
    [~, tbl_name] = fileparts(up_dir);
end
if isfile(seeds_note)
    fprintf('[%s]\n%s', tbl_name, fileread(seeds_note));
else
    fprintf('(no %s - table written before seed notes existed; rerun build_analysis.py)\n', seeds_note);
end
T = readtable(csv_path, 'VariableNamingRule', 'preserve');
vars = T.Properties.VariableNames;
snr  = T.SNR(:)';

% Reference (primary receiver) = 2nd column, '<name>_BLER' (see compare_track_modes.make_figure).
ref_col = vars{2};
csv_ref = erase(ref_col, '_BLER');

hold(ax, 'on');
set(ax, 'FontName', 'Times New Roman', 'FontSize', font_size, 'YScale', 'log', 'Box', 'on', ...
    'TickDir', 'in', 'XMinorTick', 'off', 'YMinorTick', 'on', 'LineWidth', 0.6);
grid(ax, 'on');

handles = gobjects(0);
keys    = {};
labels  = {};
all_pos = [];

[y, ~] = waterfall(T.(ref_col)(:)', FLOOR, bler_min_drawn);
all_pos = [all_pos, y(y > FLOOR)];
handles(end+1) = semilogy(ax, snr, y, 'Color', REF_STYLE{1}, 'LineStyle', REF_STYLE{2}, ...
    'Marker', REF_STYLE{3}, 'LineWidth', REF_STYLE{4} * lw_scale);
keys{end+1}   = 'ref';
labels{end+1} = '';

for i = 1:numel(modes)
    m = modes{i};
    col = [m, '_BLER'];
    if ~ismember(col, vars)
        warning('master_plot_trk_bler_paper: no column %s in %s - skipped', col, csv_path);
        continue;
    end
    if ~isKey(STYLE, m)
        warning('master_plot_trk_bler_paper: no STYLE entry for %s - skipped', m);
        continue;
    end
    st = STYLE(m);
    [y, mk] = waterfall(T.(col)(:)', FLOOR, bler_min_drawn);
    all_pos = [all_pos, y(y > FLOOR)]; %#ok<AGROW>
    handles(end+1) = semilogy(ax, snr, y, 'Color', st{2}, 'LineStyle', st{3}, 'Marker', st{4}, ...
        'MarkerIndices', mk, 'MarkerFaceColor', st{2}, ...
        'MarkerSize', marker_size + strcmp(st{4}, 'p'), 'LineWidth', st{5} * lw_scale); %#ok<AGROW>
    keys{end+1}   = m;     %#ok<AGROW>
    labels{end+1} = st{1}; %#ok<AGROW>
end
hold(ax, 'off');

xlabel(ax, 'SNR [dB]');
ylabel(ax, 'BLER');
xlim(ax, [min(snr), max(snr)]);
% y from the decade below the lowest drawn point up to 1 (whole decades, clean ticks)
if isempty(y_min)
    y_lo = 10 ^ floor(log10(min(all_pos)));
else
    y_lo = y_min;
end
set_bler_ylim(ax, y_lo);
set(ax, 'GridColor', [0.80 0.80 0.80], 'GridAlpha', 1, 'MinorGridColor', [0.90 0.90 0.90], ...
    'MinorGridAlpha', 1, 'MinorGridLineStyle', '-');
end

function set_bler_ylim(ax, y_lo)
ylim(ax, [y_lo, 1]);
set(ax, 'YTick', 10 .^ (ceil(log10(y_lo) - 1e-9):0));   % ticks on whole decades only (y_lo may be e.g. 2e-3)
end

function plot_one(csv_path, out_name, modes, STYLE, REF_STYLE, MARKER_SIZE, FLOOR, ref_label, ...
                  fig_size, legend_loc, legend_cols, y_min, bler_min_drawn)
fig = figure('Units', 'inches', 'Position', [1 1 fig_size]);
ax = axes(fig);
[h, ~, lbl, csv_ref] = draw_bler_axes(ax, csv_path, modes, STYLE, REF_STYLE, MARKER_SIZE, 1, FLOOR, ...
                                      8, y_min, bler_min_drawn);
if isempty(ref_label)
    ref_label = csv_ref;
end

% Blank legend row (no line/marker): the "<ref> + ESCNN:" group header is drawn over it after the
% legend is placed (right-aligned, with a separator line above it) - see below.
h_blank = line(ax, NaN, NaN, 'LineStyle', 'none', 'Marker', 'none');
handles = [h(1), h_blank, h(2:end)];
labels  = [{[ref_label, ' only'], ' '}, lbl(2:end)];
header_row = 2;

lgd = legend(ax, handles, labels, 'Interpreter', 'tex', 'Location', legend_loc, ...
    'NumColumns', legend_cols, 'FontSize', 6.5, 'Box', 'on', 'EdgeColor', [0.6 0.6 0.6]);
lgd.ItemTokenSize = [16, 6];

% Group header: "<ref> + ESCNN:" left-aligned in its (blank) row, at the row's bottom so a gap
% separates it from "<ref> only" above. Rows are equally spaced inside the legend box;
% legend_pad is the box's top/bottom padding as a fraction of its height.
drawnow;
lp = lgd.Position;                       % normalized figure units
n_rows = numel(labels);
legend_pad = 0.035;
row_h = lp(4) * (1 - 2 * legend_pad) / n_rows;
row_top = @(k) lp(2) + lp(4) * (1 - legend_pad) - (k - 1) * row_h;   % top edge of row k
y_row = row_top(header_row) - row_h;     % bottom edge of the header row
annotation(fig, 'textbox', [lp(1) + 0.015, y_row - 0.2 * row_h, lp(3) - 0.03, row_h], ...
    'String', [ref_label, ' + ESCNN:'], 'Interpreter', 'none', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'EdgeColor', 'none', ...
    'Margin', 0, 'FontName', 'Times New Roman', 'FontSize', 6.5, 'FontAngle', 'normal');

export_fig_set(fig, out_name, fig_size);
end

function plot_pair(csv_paths, out_name, titles, modes, STYLE, REF_STYLE, FLOOR, P, y_min, bler_min_drawn)
% Two BLER subplots side by side (left = csv_paths{1}, right = csv_paths{2}) with one joint
% legend centered below - the master_plot_mi_bler.m layout.
fig = figure('Units', 'inches', 'Position', [1 1 P.fig_size]);
ax = gobjects(1, 2);
H = containers.Map();      % key ('ref' or mode) -> first handle drawn with that key (for the legend)
L = containers.Map();      % key -> legend label
for k = 1:2
    ax(k) = subplot(1, 2, k);
    [h, keys, lbl, csv_ref] = draw_bler_axes(ax(k), csv_paths{k}, modes, STYLE, REF_STYLE, ...
        P.marker_size, P.lw_scale, FLOOR, P.font_size, y_min, bler_min_drawn);
    for j = 1:numel(keys)
        if ~isKey(H, keys{j})
            H(keys{j}) = h(j);
            if isKey(P.labels, keys{j})
                L(keys{j}) = P.labels(keys{j});
            else
                L(keys{j}) = lbl{j};
            end
        end
    end
    if isempty(titles) || isempty(titles{k})
        ttl = csv_ref;
    else
        ttl = titles{k};
    end
    title(ax(k), ttl, 'FontName', 'Times New Roman', 'FontSize', P.font_size);
end
L('ref') = P.ref_label;

if P.same_ylim
    y_lo = min(arrayfun(@(a) a.YLim(1), ax));
    for k = 1:2
        set_bler_ylim(ax(k), y_lo);
    end
end

% Joint legend in P.order (default: reference, then `modes`); a mode missing from one table
% still gets its entry from the other.
if isempty(P.order)
    order = [{'ref'}, modes];
else
    order = [P.order, setdiff(modes, P.order(~cellfun(@isempty, P.order)), 'stable')];
end
handles = gobjects(0);
labels  = {};
for k = 1:numel(order)
    if isempty(order{k})          % empty legend slot (keeps the column alignment)
        handles(end+1) = line(ax(2), NaN, NaN, 'LineStyle', 'none', 'Marker', 'none'); %#ok<AGROW>
        labels{end+1}  = ''; %#ok<AGROW>
    elseif isKey(H, order{k})
        handles(end+1) = H(order{k}); %#ok<AGROW>
        labels{end+1}  = L(order{k}); %#ok<AGROW>
        if ismember(order{k}, P.bold)
            labels{end} = ['\bf', labels{end}];   % tex interpreter: bold for this entry only
        end
    end
end
lgd = legend(ax(2), handles, labels, 'Interpreter', 'tex', 'Orientation', 'horizontal', ...
    'NumColumns', P.legend_cols, 'FontName', 'Times New Roman', 'FontSize', P.legend_font_size, ...
    'Box', 'on', 'EdgeColor', [0.6 0.6 0.6]);
lgd.Units       = 'normalized';
lgd.Position(1) = 0.5 - lgd.Position(3)/2;
lgd.Position(2) = 0.01;

% Shrink subplots to make room for the legend + xlabel
for d = 1:2
    pos = ax(d).Position;
    ax(d).Position = [pos(1), pos(2) + P.shrink, pos(3), pos(4) - P.shrink];
end

export_fig_set(fig, out_name, P.fig_size);
end

function export_fig_set(fig, out_name, fig_size)
% EPS (vector, -painters) as the older master_plot_* scripts do - what the paper includes;
% PDF also vector (handy for Overleaf), PNG for a quick look, FIG to tweak by hand.
set(fig, 'PaperUnits', 'inches', 'PaperPosition', [0 0 fig_size], 'PaperSize', fig_size);
print(fig, [out_name, '.eps'], '-depsc', '-painters');
exportgraphics(fig, [out_name, '.pdf'], 'ContentType', 'vector');
exportgraphics(fig, [out_name, '.png'], 'Resolution', 600);
savefig(fig, [out_name, '.fig']);
fprintf('Saved %s.{eps,pdf,png,fig}\n', out_name);
end

function [y, mk] = waterfall(b, floor_val, y_min)
% y_min (optional, non-empty): values below it count as 0 (not resolved on this axis).
if nargin > 2 && ~isempty(y_min)
    b(b < y_min) = 0;
end
% BLER = 0 (no block errors) can't be drawn on a log axis: the first zero after
% a non-zero point becomes floor_val (line drops straight off the axis, no
% marker there), further zeros and missing SNRs (NaN) leave the line broken.
y = b;
prev = NaN;
for k = 1:numel(b)
    if isnan(b(k))
        y(k) = NaN;
    elseif b(k) > 0
        y(k) = b(k);
    elseif ~isnan(prev) && prev > floor_val
        y(k) = floor_val;
    else
        y(k) = NaN;
    end
    prev = y(k);
end
mk = find(~isnan(y) & y > floor_val);
end
