% =========================================================
% master_plot_trk_bler_paper.m  -  BLER-only paper figure (IEEE single
%                                  column) comparing tracking modes,
%                                  seed-averaged.
% =========================================================
%
% Usage: run build_analysis.py <tag> first, then set analysis_dir below to the
% name of the Analysis folder it produced (e.g. 'itrknk', or 'itrknk_R=0.48_LMMSE' - found under
% its parent Analysis\itrknk\ - for a tag that build_analysis split per code rate / primary) and run.
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
analysis_dir = 'idsseed';      % folder name under C:\Projects\Scratchpad\Analysis (or a full path)
modes      = {'notrack', 'ekfcrc', 'ekfht', 'sgdht', 'ekfit'};   % legend order (ref is drawn first)
seeds      = [17 123 1011];          % [] = every seed complete for `modes`; or e.g. [58 123 1011 1505]
%seeds      = [];          % [] = every seed complete for `modes`; or e.g. [58 123 1011 1505]
recompute  = true;        % true = re-average for `modes`/`seeds` (see above); false = tables as is
zipfile    = '';          % recompute only: '' = loose result CSVs in Scratchpad, else the archive name
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

% ---- Find the tables build_analysis wrote ----
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
out_dir = fullfile(dir_path, 'paper');
if ~isfolder(out_dir)
    mkdir(out_dir);
end

if isempty(seeds)
    seed_txt = 'all';
    seed_arg = '';
else
    seed_txt = strjoin(arrayfun(@num2str, seeds, 'UniformOutput', false), '-');
    seed_arg = [' --seeds ', strjoin(arrayfun(@num2str, seeds, 'UniformOutput', false), ' ')];
end

for t = 1:numel(tables)
    csv_path = fullfile(dir_path, tables{t});
    base     = tables{t}(1:end-4);
    out_name = fullfile(out_dir, [base, '_paper']);
    fprintf('\n==== %s ====\n', csv_path);
    if recompute
        cfg = read_config(csv_path);
        if isempty(cfg)
            % Older table (no config line): fall back to the tag as the only filter token.
            dir_leaf = regexprep(dir_path, '.*[\\/]', '');
            cfg = strtok(dir_leaf, '_');
            fprintf(['(no config line in %s''s seeds note - filtering on the tag ''%s'' only; ', ...
                     'rerun build_analysis.py for an exact match)\n'], tables{t}, cfg);
        end
        if true
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
    end
    plot_one(csv_path, out_name, modes, STYLE, REF_STYLE, MARKER_SIZE, FLOOR, ref_label, ...
             fig_size, legend_loc, legend_cols, y_min, bler_min_drawn);
end

% =========================================================
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

function plot_one(csv_path, out_name, modes, STYLE, REF_STYLE, MARKER_SIZE, FLOOR, ref_label, ...
                  fig_size, legend_loc, legend_cols, y_min, bler_min_drawn)
seeds_note = [csv_path(1:end-4), '_seeds.txt'];
if isfile(seeds_note)
    fprintf('%s', fileread(seeds_note));
else
    fprintf('(no %s - table written before seed notes existed; rerun build_analysis.py)\n', seeds_note);
end
T = readtable(csv_path, 'VariableNamingRule', 'preserve');
vars = T.Properties.VariableNames;
snr  = T.SNR(:)';

% Reference (primary receiver) = 2nd column, '<name>_BLER' (see compare_track_modes.make_figure).
ref_col = vars{2};
if isempty(ref_label)
    ref_label = erase(ref_col, '_BLER');
end

fig = figure('Units', 'inches', 'Position', [1 1 fig_size]);
ax = axes(fig);
hold(ax, 'on');
set(ax, 'FontName', 'Times New Roman', 'FontSize', 8, 'YScale', 'log', 'Box', 'on', ...
    'TickDir', 'in', 'XMinorTick', 'off', 'YMinorTick', 'on', 'LineWidth', 0.6);
grid(ax, 'on');

handles = gobjects(0);
labels  = {};
all_pos = [];

[y, ~] = waterfall(T.(ref_col)(:)', FLOOR, bler_min_drawn);
all_pos = [all_pos, y(y > FLOOR)];
handles(end+1) = semilogy(ax, snr, y, 'Color', REF_STYLE{1}, 'LineStyle', REF_STYLE{2}, ...
    'Marker', REF_STYLE{3}, 'LineWidth', REF_STYLE{4});
labels{end+1} = [ref_label, ' only'];
% Blank legend row (no line/marker): the "<ref> + ESCNN:" group header is drawn over it after the
% legend is placed (right-aligned, with a separator line above it) - see below.
handles(end+1) = semilogy(ax, NaN, NaN, 'LineStyle', 'none', 'Marker', 'none');
labels{end+1} = ' ';
header_row = numel(labels);

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
        'MarkerSize', MARKER_SIZE + strcmp(st{4}, 'p'), 'LineWidth', st{5}); %#ok<AGROW>
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
ylim(ax, [y_lo, 1]);
set(ax, 'YTick', 10 .^ (ceil(log10(y_lo) - 1e-9):0));   % ticks on whole decades only (y_lo may be e.g. 2e-3)
set(ax, 'GridColor', [0.80 0.80 0.80], 'GridAlpha', 1, 'MinorGridColor', [0.90 0.90 0.90], ...
    'MinorGridAlpha', 1, 'MinorGridLineStyle', '-');

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

% ---- Export ----
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
