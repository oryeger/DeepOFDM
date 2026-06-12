% =========================================================
% master_plot_mi_bler.m  -  MI (left) + BLER (right) side-by-side
%                           for a single run (one code rate)
% =========================================================

clear; clc;

% ---- User configuration ----
base_name        = 'Clip_UMi_4UEs_goodmcs';
extra_text       = '';             % e.g. '_transfer'
root_dir         = 'C:\Projects\Scratchpad\mat_files\';

algs_to_plot     = [1 2 3 4];     % 1=LMMSE, 2=SPHERE, 3=DeepRx, 4=DeepSIC
add_snr_target   = false;         % append SNR@10% to BLER legend labels
plot_aug_iter_2  = false;         % plot second aug iteration if available
snr_pad_left_db   = 0;            % extend SNR axis to the left by this many dB (0 = no padding)
snr_cut_right_pts = 0;            % cut this many SNR points from the right (0 = no cut)
snr_cut_left_pts  = 0;            % cut this many SNR points from the left  (0 = no cut)
output_target    = 'paper';       % 'paper' (compact, default legend) or 'ppt' (large, reordered legend, fontsize 14, PNG export)
override_mi_only = false;         % if true: do not create a new figure; just redraw the MI subplot in the currently-open figure

% ---- MI zoom inset (left subplot only) ----
mi_zoom_enable   = false;          % draw a zoom inset on the MI subplot
mi_zoom_xlim     = [16, 24];      % SNR range (dB) shown inside the inset
mi_zoom_ylim     = [0.965, 1.00]; % MI  range shown inside the inset
mi_zoom_position = [0.41, 0.10, 0.50, 0.45]; % inset placement inside MI axes, normalized [left bottom width height]
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

% Completion runs label SPHERE as RBSD in the legend.
if contains(base_name, 'Completion')
    alg_names{2} = 'RBSD';
end

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

% ---- Override mode: redraw MI subplot in the existing open figure ----
if override_mi_only
    if isempty(findall(0, 'Type', 'figure'))
        error('override_mi_only=true requires an open figure.');
    end
    fig = gcf;

    % Find the MI subplot in the open figure. Prefer the axes with a linear
    % YScale (BLER uses log); fall back to the leftmost axes by position.
    all_axes = findobj(fig, 'Type', 'axes');
    ax_mi = gobjects(0);
    for k = 1:numel(all_axes)
        if strcmpi(get(all_axes(k), 'YScale'), 'linear')
            ax_mi = all_axes(k);
            break;
        end
    end
    if isempty(ax_mi)
        xs = arrayfun(@(a) get(a, 'Position') * [1;0;0;0], all_axes);
        [~, left_idx] = min(xs);
        ax_mi = all_axes(left_idx);
    end

    % Capture current xlim so we can keep MI x-axis aligned with BLER.
    prev_xlim = xlim(ax_mi);

    % Clear graphics children only -- DO NOT reset, otherwise we lose the
    % subplot's shrunk position, font sizes, etc. that match the old PPT
    % format (set by the original figure's shrink_amt + ppt config).
    cla(ax_mi);
    axes(ax_mi); %#ok<LAXES>
    hold on; grid on;

    [h_mi, lbl_mi] = plot_mi_set( ...
        dir_path, ...
        algs_to_plot, alg_colors, alg_names, alg_files, ...
        markers_no_aug, markers_aug, fillable_algs, ...
        plot_aug_iter_2, snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts);

    hold off;

    % Match master_plot_bler.m order: title first, then YScale, xlabel, ylabel,
    % then YMinorTick+Box. Use absolute Position (not '+0.03') in override mode
    % because title() preserves Position across calls, so the relative nudge
    % accumulates every time the script is re-run and pushes the title off-chart.
    if is_ppt
        t1 = title(ax_mi, 'Bit-wise Mutual Information');
        t1.Units = 'normalized';
        t1.Position(2) = 1.03;
    else
        t1 = title(ax_mi, 'Bit-wise Mutual Information');
        t1.Units = 'normalized';
        t1.Position(2) = 1.0;
    end
    set(ax_mi, 'YScale', 'linear');
    xlabel(ax_mi, 'SNR (dB)');
    ylabel(ax_mi, 'MI');
    ylim(ax_mi, [0, 1]);
    set(ax_mi, 'YMinorTick', 'on', 'Box', 'on');
    xlim(ax_mi, prev_xlim);

    % ---- Find BLER axes and retitle it to "Block Error Rate" ----
    ax_bler = gobjects(0);
    for k = 1:numel(all_axes)
        if strcmpi(get(all_axes(k), 'YScale'), 'log')
            ax_bler = all_axes(k);
            break;
        end
    end
    if ~isempty(ax_bler)
        if is_ppt
            t2 = title(ax_bler, 'Block Error Rate');
            t2.Units = 'normalized';
            t2.Position(2) = 1.03;
        else
            t2 = title(ax_bler, 'Block Error Rate');
            t2.Units = 'normalized';
            t2.Position(2) = 1.0;
        end
    end

    % ---- MI zoom inset ----
    if mi_zoom_enable
        add_mi_zoom_inset(ax_mi, mi_zoom_xlim, mi_zoom_ylim, mi_zoom_position);
    end

    % ---- Rebuild the legend with correct labels ----
    % Build the legend from the freshly-created MI handles+labels (h_mi /
    % lbl_mi). MI is always plotted for every alg in algs_to_plot, whereas
    % BLER may be missing some algorithms (e.g. DeepSIC at certain rates),
    % so using MI guarantees a complete legend. Marker / linestyle / color
    % are identical between plot_mi_set and plot_bler_set, so the legend
    % looks the same regardless of which axes' handles it points to.

    legend_labels = regexprep(lbl_mi, ',?\s*r=0\.\d+', '');

    desired_order = {alg_names{1},             alg_names{2},             alg_names{4},             alg_names{3}, ...
                     [alg_names{1}, ' aug'],   [alg_names{2}, ' aug'],   [alg_names{4}, ' aug'],   [alg_names{3}, ' aug']};
    ordered_handles = gobjects(0);
    ordered_labels  = {};
    for k = 1:numel(desired_order)
        idx = find(strcmp(legend_labels, desired_order{k}), 1);
        if ~isempty(idx)
            ordered_handles(end+1) = h_mi(idx);             %#ok<AGROW>
            ordered_labels{end+1}  = legend_labels{idx};    %#ok<AGROW>
        end
    end

    if ~isempty(ordered_handles)
        delete(findobj(fig, 'Type', 'legend'));

        legend_args = {'Interpreter', 'none', 'Orientation', 'horizontal', 'NumColumns', 4};
        if is_ppt
            legend_args = [legend_args, {'FontSize', 14}];
        end

        if ~isempty(ax_bler)
            lgd_parent = ax_bler;
        else
            lgd_parent = ax_mi;
        end
        lgd = legend(lgd_parent, ordered_handles, ordered_labels, legend_args{:});
        lgd.Units       = 'normalized';
        lgd.Position(1) = 0.5 - lgd.Position(3)/2;
        lgd.Position(2) = 0.01;
    end

    % ---- Export (same set of formats as normal mode) ----
    out_name = fullfile(root_dir, [base_name, extra_text, '_mi_bler']);
    print(fig, [out_name, '.eps'], '-depsc', '-painters');
    if is_ppt
        print(fig, [out_name, '.png'], '-dpng',  '-r600');
    end
    savefig(fig, [out_name, '.fig']);

    fprintf('Override mode: MI subplot updated in figure %d; saved %s.{eps,fig}.\n', ...
        fig.Number, out_name);
    return;
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

% Order matches master_plot_bler.m: title (with PPT nudge) -> YScale -> xlabel
% -> ylabel -> ylim -> YMinorTick+Box. Keeping this order is required so the
% +0.03 normalized title nudge sees the same baseline as the BLER-only script.
if is_ppt
    t1 = title(ax(1), 'Bit-wise Mutual Information');
    t1.Units = 'normalized';
    t1.Position(2) = t1.Position(2) + 0.03;
else
    title(ax(1), 'Bit-wise Mutual Information');
end
set(ax(1), 'YScale', 'linear');
xlabel(ax(1), 'SNR (dB)');
ylabel(ax(1), 'MI');
ylim(ax(1), [0, 1]);
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

if is_ppt
    t2 = title(ax(2), 'Block Error Rate');
    t2.Units = 'normalized';
    t2.Position(2) = t2.Position(2) + 0.03;
else
    title(ax(2), 'Block Error Rate');
end
set(ax(2), 'YScale', 'log');
xlabel(ax(2), 'SNR (dB)');
ylabel(ax(2), 'BLER');
set(ax(2), 'YMinorTick', 'on', 'Box', 'on');

% Match MI x-axis to BLER's full SNR span so both subplots line up at SNR=0.
xlim(ax(1), xlim(ax(2)));

% ---- Shared legend below (use BLER labels; strip code-rate suffix) ----
legend_labels = regexprep(lbl_bler, ',?\s*r=0\.\d+', '');

% Reorder so row 1 = all no-aug, row 2 = all aug, each with the requested alg order.
desired_order = {alg_names{1},             alg_names{2},             alg_names{4},             alg_names{3}, ...
                 [alg_names{1}, ' aug'],   [alg_names{2}, ' aug'],   [alg_names{4}, ' aug'],   [alg_names{3}, ' aug']};
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

% ---- MI zoom inset ----
if mi_zoom_enable
    add_mi_zoom_inset(ax(1), mi_zoom_xlim, mi_zoom_ylim, mi_zoom_position);
end

% ---- Export ----
out_name = fullfile(root_dir, [base_name, extra_text, '_mi_bler']);
print(fig, [out_name, '.eps'], '-depsc', '-painters');
if is_ppt
    print(fig, [out_name, '.png'], '-dpng',  '-r600');   % high-DPI raster for PPT
end
savefig([out_name, '.fig']);

% =========================================================
% Local functions
% =========================================================

function add_mi_zoom_inset(ax_mi, zoom_xlim, zoom_ylim, inset_pos_norm)
    % Add a magnified inset of a region of the MI subplot. The inset shows
    % the same MI curves restricted to (zoom_xlim, zoom_ylim). A dashed
    % rectangle is drawn on the main MI plot to mark the zoomed region,
    % and two black connector lines are drawn from the rectangle to the
    % inset so the reader can see which area was magnified.
    %
    % inset_pos_norm : [left bottom width height], normalized within ax_mi.

    parent_fig = ancestor(ax_mi, 'figure');

    % Remove any existing zoom inset / marker / connectors / overlay so
    % re-running override mode doesn't stack them on top of each other.
    delete(findobj(parent_fig, 'Tag', 'mi_zoom_inset'));
    delete(findobj(ax_mi,      'Tag', 'mi_zoom_rect'));
    delete(findall(parent_fig, 'Tag', 'mi_zoom_connector'));
    delete(findobj(parent_fig, 'Tag', 'mi_zoom_overlay'));

    % Convert inset position from MI-axes-relative to figure-relative.
    ax_pos = ax_mi.Position;
    inset_pos_fig = [ax_pos(1) + inset_pos_norm(1) * ax_pos(3), ...
                     ax_pos(2) + inset_pos_norm(2) * ax_pos(4), ...
                     inset_pos_norm(3) * ax_pos(3), ...
                     inset_pos_norm(4) * ax_pos(4)];

    inset_ax = axes('Parent', parent_fig, 'Position', inset_pos_fig);
    inset_ax.Tag = 'mi_zoom_inset';

    % Copy only the Line children from the MI axes (skip title/labels/etc.).
    mi_lines = findobj(ax_mi, 'Type', 'line');
    if ~isempty(mi_lines)
        copyobj(mi_lines, inset_ax);
    end

    xlim(inset_ax, zoom_xlim);
    ylim(inset_ax, zoom_ylim);
    set(inset_ax, 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on', ...
                  'Color', [1 1 1], 'FontSize', 8);
    grid(inset_ax, 'on');

    % Solid black rectangle on the main MI plot marking the zoomed region.
    rectangle(ax_mi, ...
        'Position',  [zoom_xlim(1), zoom_ylim(1), diff(zoom_xlim), diff(zoom_ylim)], ...
        'EdgeColor', 'k', ...
        'LineStyle', '-', ...
        'LineWidth', 0.8, ...
        'Tag',       'mi_zoom_rect');

    % ---- Connector lines from zoom rectangle on main axes to inset box ----
    % Force rendering so ax_mi.Position and inset_ax.Position reflect the
    % final laid-out positions before we convert coordinates.
    drawnow;

    ax_pos    = double(ax_mi.Position);
    inset_pos = double(inset_ax.Position);
    main_xlim = double(ax_mi.XLim);   % some datasets return XLim as int32 etc.
    main_ylim = double(ax_mi.YLim);
    zoom_xlim = double(zoom_xlim);
    zoom_ylim = double(zoom_ylim);

    % Convert main-axes data coords to figure-normalized using:
    %   x_norm = ax.Position(1) + (x_data - ax.XLim(1)) / diff(ax.XLim) * ax.Position(3)
    %   y_norm = ax.Position(2) + (y_data - ax.YLim(1)) / diff(ax.YLim) * ax.Position(4)
    zoom_tl_x = ax_pos(1) + (zoom_xlim(1) - main_xlim(1)) / diff(main_xlim) * ax_pos(3);
    zoom_tl_y = ax_pos(2) + (zoom_ylim(2) - main_ylim(1)) / diff(main_ylim) * ax_pos(4);
    zoom_br_x = ax_pos(1) + (zoom_xlim(2) - main_xlim(1)) / diff(main_xlim) * ax_pos(3);
    zoom_br_y = ax_pos(2) + (zoom_ylim(1) - main_ylim(1)) / diff(main_ylim) * ax_pos(4);

    % Inset box corners (Position is already in figure-normalized coords).
    inset_tl_x = inset_pos(1);
    inset_tl_y = inset_pos(2) + inset_pos(4);
    inset_br_x = inset_pos(1) + inset_pos(3);
    inset_br_y = inset_pos(2);

    fprintf('[mi_zoom] ax_pos     = [%.4f %.4f %.4f %.4f]\n', ax_pos);
    fprintf('[mi_zoom] inset_pos  = [%.4f %.4f %.4f %.4f]\n', inset_pos);
    fprintf('[mi_zoom] main_xlim  = [%.4f %.4f]    main_ylim = [%.4f %.4f]\n', main_xlim, main_ylim);
    fprintf('[mi_zoom] zoom_xlim  = [%.4f %.4f]    zoom_ylim = [%.4f %.4f]\n', zoom_xlim, zoom_ylim);

    % Step-by-step debug of the X-term computation
    dbg_num   = zoom_xlim(1) - main_xlim(1);
    dbg_den   = diff(main_xlim);
    dbg_w     = ax_pos(3);
    dbg_ratio = dbg_num / dbg_den;
    dbg_term  = dbg_ratio * dbg_w;
    fprintf('[mi_zoom] DEBUG x-term TL\n');
    fprintf('          class(zoom_xlim)=%s size=[%s]\n', class(zoom_xlim), num2str(size(zoom_xlim)));
    fprintf('          class(main_xlim)=%s size=[%s]\n', class(main_xlim), num2str(size(main_xlim)));
    fprintf('          class(ax_pos)   =%s size=[%s]\n', class(ax_pos),    num2str(size(ax_pos)));
    fprintf('          num   = zoom_xlim(1)-main_xlim(1) = %.10g  size=[%s] class=%s\n', dbg_num,   num2str(size(dbg_num)),   class(dbg_num));
    fprintf('          den   = diff(main_xlim)           = %.10g  size=[%s] class=%s\n', dbg_den,   num2str(size(dbg_den)),   class(dbg_den));
    fprintf('          w     = ax_pos(3)                 = %.10g  size=[%s] class=%s\n', dbg_w,     num2str(size(dbg_w)),     class(dbg_w));
    fprintf('          num/den                           = %.10g  size=[%s]\n',         dbg_ratio, num2str(size(dbg_ratio)));
    fprintf('          (num/den)*w                       = %.10g  size=[%s]\n',         dbg_term,  num2str(size(dbg_term)));

    fprintf('[mi_zoom] zoom  TL   = (%.4f, %.4f)   BR = (%.4f, %.4f)\n', ...
            zoom_tl_x, zoom_tl_y, zoom_br_x, zoom_br_y);
    fprintf('[mi_zoom] inset TL   = (%.4f, %.4f)   BR = (%.4f, %.4f)\n', ...
            inset_tl_x, inset_tl_y, inset_br_x, inset_br_y);

    % Make parent_fig the current figure so annotation() targets it.
    prev_fig = get(0, 'CurrentFigure');
    figure(parent_fig);
    cleanup_fig = onCleanup(@() restore_current_fig(prev_fig));

    % Top-left of zoom rect -> top-left of inset box.
    ann1 = annotation('line', [zoom_tl_x, inset_tl_x], [zoom_tl_y, inset_tl_y], ...
                      'Color', 'k', 'LineWidth', 1);
    ann1.Tag = 'mi_zoom_connector';

    % Bottom-right of zoom rect -> bottom-right of inset box.
    ann2 = annotation('line', [zoom_br_x, inset_br_x], [zoom_br_y, inset_br_y], ...
                      'Color', 'k', 'LineWidth', 1);
    ann2.Tag = 'mi_zoom_connector';
end

function restore_current_fig(prev_fig)
    if ~isempty(prev_fig) && isgraphics(prev_fig)
        set(0, 'CurrentFigure', prev_fig);
    end
end
