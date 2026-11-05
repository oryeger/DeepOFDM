% =========================================================
% master_plot_mi_bler_zip.m  -  MI (left) + BLER (right) side-by-side,
%                               read directly from a .gz archive of CSVs
%
% Same figure as master_plot_mi_bler.m (normal mode). The only difference is
% the data source:
%   archive (.tar.gz of result CSVs)
%     -> make_mats_from_archive.py (extracts, checks seeds/SNRs, then calls
%        plot_multiple_csvs.plot_csvs per detector -> lmmse/sphere/deeprx/deepsic.mat)
%     -> plot_mi_set / plot_bler_set (unchanged, from escnn_matlab)
% =========================================================

clear; clc;

% ---- User configuration ----
archive_path     = 'C:\Projects\Scratchpad\fig5.gz';
out_base_name    = 'CFO_MI';      % output: <work_dir>\<out_base_name>_mi_bler.{eps,fig}
work_dir         = 'C:\Projects\Scratchpad\fig_from_archive';   % extracted CSVs, .mat files and figures (outside the repo)
python_exe       = 'C:\Users\owner\anaconda3\python.exe';
mats_script      = 'C:\Projects\DeepOFDM\python_code\utils\make_mats_from_archive.py';
escnn_matlab_dir = 'C:\Projects\DeepOFDM\escnn_matlab';   % plot_mi_set / plot_bler_set

algs_to_plot     = [1 2 3 4];     % 1=LMMSE, 2=SPHERE, 3=DeepRx, 4=DeepSIC
sphere_label     = 'SPHERE';      % 'SPHERE' for full sphere (ip=0), 'RBSD' for reduced-bit sphere (ip=1)
add_snr_target   = false;         % append SNR@10% to BLER legend labels
plot_aug_iter_2  = false;         % plot second aug iteration if available
snr_pad_left_db   = 0;            % extend SNR axis to the left by this many dB (0 = no padding)
snr_cut_right_pts = 5;            % cut this many SNR points from the right (0 = no cut)
snr_cut_left_pts  = 0;            % cut this many SNR points from the left  (0 = no cut)
output_target    = 'paper';       % 'paper' (compact, default legend) or 'ppt' (large, fontsize 14, PNG export)
export_pdf       = false;         % also export a vector PDF (exportgraphics)

% ---- MI zoom inset (left subplot only) ----
mi_zoom_enable   = false;
mi_zoom_xlim     = [35, 40];
mi_zoom_ylim     = [0.81, 0.86];
mi_zoom_position = [0.11, 0.65, 0.26, 0.30];
% ----------------------------

is_ppt = strcmpi(output_target, 'ppt');
addpath(escnn_matlab_dir);

% ---- Build the .mat files from the archive ----
[~, arch_stem] = fileparts(archive_path);
csv_dir  = fullfile(work_dir, ['csv_', arch_stem]);
dir_path = fullfile(work_dir, ['mat_', arch_stem]);
out_dir  = work_dir;
if ~isfolder(work_dir)
    mkdir(work_dir);
end

cmd = sprintf('"%s" "%s" "%s" "%s" "%s"', python_exe, ...
    mats_script, archive_path, csv_dir, dir_path);
fprintf('Running: %s\n', cmd);
status = system(cmd);
if status ~= 0
    error('make_mats_from_archive.py failed (status %d).', status);
end

fprintf('Plotting MI + BLER from: %s\n', dir_path);

% ---- Shared style definitions (identical to master_plot_mi_bler.m) ----
alg_colors = [0.00, 0.45, 0.70;   % LMMSE   - blue
              0.85, 0.33, 0.10;   % SPHERE  - vermillion
              0.47, 0.67, 0.19;   % DeepRx  - green
              0.49, 0.18, 0.56];  % DeepSIC - purple

alg_names = {'LMMSE', 'SPHERE', 'DeepRx', 'DeepSIC'};
alg_files = {'lmmse', 'sphere', 'deeprx', 'deepsic'};
alg_names{2} = sphere_label;

markers_no_aug = {'^'; 's'; 'o'; 'd'};
markers_aug    = {'^'; 's'; 'o'; 'd'};

fillable_algs = [1:4];
% ----------------------------------

% ---- Create figure: MI left, BLER right ----
fig = figure;
if is_ppt
    set(fig, 'Units', 'inches', 'Position', [0 0 14 7]);
else
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

% Match MI x-axis to BLER's full SNR span so both subplots line up.
xlim(ax(1), xlim(ax(2)));

% ---- Shared legend below ----
legend_labels = regexprep(lbl_bler, ',?\s*r=0\.\d+', '');

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

if is_ppt
    shrink_amt = 0.18;
else
    shrink_amt = 0.14;
end
for d = 1:2
    pos = ax(d).Position;
    ax(d).Position = [pos(1), pos(2)+shrink_amt, pos(3), pos(4)-shrink_amt];
end

if mi_zoom_enable
    add_mi_zoom_inset(ax(1), mi_zoom_xlim, mi_zoom_ylim, mi_zoom_position);
end

% ---- Export ----
out_name = fullfile(out_dir, [out_base_name, '_mi_bler']);
print(fig, [out_name, '.eps'], '-depsc', '-painters');
if is_ppt
    print(fig, [out_name, '.png'], '-dpng',  '-r600');
end
if export_pdf
    exportgraphics(fig, [out_name, '.pdf'], 'ContentType', 'vector');
end
savefig(fig, [out_name, '.fig']);
fprintf('Saved %s.{eps,fig}\n', out_name);

% =========================================================
% Local functions (copied from master_plot_mi_bler.m, debug prints removed)
% =========================================================

function add_mi_zoom_inset(ax_mi, zoom_xlim, zoom_ylim, inset_pos_norm)
    parent_fig = ancestor(ax_mi, 'figure');
    delete(findobj(parent_fig, 'Tag', 'mi_zoom_inset'));
    delete(findobj(ax_mi,      'Tag', 'mi_zoom_rect'));
    delete(findall(parent_fig, 'Tag', 'mi_zoom_connector'));

    ax_pos = ax_mi.Position;
    inset_pos_fig = [ax_pos(1) + inset_pos_norm(1) * ax_pos(3), ...
                     ax_pos(2) + inset_pos_norm(2) * ax_pos(4), ...
                     inset_pos_norm(3) * ax_pos(3), ...
                     inset_pos_norm(4) * ax_pos(4)];
    inset_ax = axes('Parent', parent_fig, 'Position', inset_pos_fig);
    inset_ax.Tag = 'mi_zoom_inset';
    mi_lines = findobj(ax_mi, 'Type', 'line');
    if ~isempty(mi_lines)
        copyobj(mi_lines, inset_ax);
    end
    xlim(inset_ax, zoom_xlim);
    ylim(inset_ax, zoom_ylim);
    set(inset_ax, 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on', ...
                  'Color', [1 1 1], 'FontSize', 6);
    set(inset_ax, 'YTick', [zoom_ylim(1), zoom_ylim(2)]);
    grid(inset_ax, 'on');

    rectangle(ax_mi, ...
        'Position',  [zoom_xlim(1), zoom_ylim(1), diff(zoom_xlim), diff(zoom_ylim)], ...
        'EdgeColor', 'k', 'LineStyle', '-', 'LineWidth', 0.8, 'Tag', 'mi_zoom_rect');

    drawnow;
    ax_pos    = double(ax_mi.Position);
    inset_pos = double(inset_ax.Position);
    main_xlim = double(ax_mi.XLim);
    main_ylim = double(ax_mi.YLim);
    zoom_xlim = double(zoom_xlim);
    zoom_ylim = double(zoom_ylim);

    zoom_tl_x = ax_pos(1) + (zoom_xlim(1) - main_xlim(1)) / diff(main_xlim) * ax_pos(3);
    zoom_tl_y = ax_pos(2) + (zoom_ylim(2) - main_ylim(1)) / diff(main_ylim) * ax_pos(4);
    zoom_br_x = ax_pos(1) + (zoom_xlim(2) - main_xlim(1)) / diff(main_xlim) * ax_pos(3);
    zoom_br_y = ax_pos(2) + (zoom_ylim(1) - main_ylim(1)) / diff(main_ylim) * ax_pos(4);

    figure(parent_fig);
    ann1 = annotation('line', [zoom_tl_x, inset_pos(1)], [zoom_tl_y, inset_pos(2) + inset_pos(4)], ...
                      'Color', 'k', 'LineWidth', 1);
    ann1.Tag = 'mi_zoom_connector';
    ann2 = annotation('line', [zoom_br_x, inset_pos(1) + inset_pos(3)], [zoom_br_y, inset_pos(2)], ...
                      'Color', 'k', 'LineWidth', 1);
    ann2.Tag = 'mi_zoom_connector';
end
