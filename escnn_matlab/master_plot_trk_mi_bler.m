% =========================================================
% master_plot_trk_mi_bler.m  -  MI (left) + BLER (right) side-by-side,
%                                comparing tracking modes (trk=ekf/sgdbce/
%                                sgdbcei/sgdsyn/notrack/...) for one fixed
%                                underlying config.
% =========================================================
%
% Point dir_path at a folder produced by build_analysis.py's mat_target():
% one config's trk= sweep, saved as one <mode>.mat file per tracking mode
% (e.g. mat_files\<tag>\<rest-of-config>\ekf.mat, sgdbcei.mat, ...). Every
% mode plots solid except trk=notrack, which is dashed (see trk_style.m for
% the color/marker assignment).

clear; clc;

% ---- User configuration ----
dir_path          = 'C:\Projects\Scratchpad\mat_files\t2x2cfo\sp=0_QPSK_cfo=0.54_cfod=0.75_R=0.30_r=d6d6af_frz=f';
extra_text        = '';           % e.g. '_transfer'
add_snr_target    = false;        % append SNR@10% to BLER legend labels
snr_pad_left_db   = 0;            % extend SNR axis to the left by this many dB (0 = no padding)
snr_cut_right_pts = 0;            % cut this many SNR points from the right (0 = no cut)
snr_cut_left_pts  = 0;            % cut this many SNR points from the left  (0 = no cut)
output_target     = 'paper';      % 'paper' (compact, default legend) or 'ppt' (large, fontsize 14, PNG export)
trk_colors        = [];           % containers.Map(trk_mode -> [r g b]) to override trk_style.m defaults, or [] to use them as-is
% ----------------------------

is_ppt = strcmpi(output_target, 'ppt');

if ~isfolder(dir_path)
    error('master_plot_trk_mi_bler: dir_path not found: %s', dir_path);
end
fprintf('Plotting trk-mode MI + BLER comparison from: %s\n', dir_path);

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

[h_mi, lbl_mi] = plot_mi_trk_set( ...
    dir_path, trk_colors, snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts); %#ok<ASGLU>

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

[h_bler, lbl_bler] = plot_bler_trk_set( ...
    dir_path, trk_colors, add_snr_target, snr_pad_left_db, snr_cut_right_pts, snr_cut_left_pts);

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

% ---- Shared legend below (BLER labels; same trk modes as MI) ----
legend_args = {'Interpreter', 'none', 'Orientation', 'horizontal', 'NumColumns', max(1, min(numel(lbl_bler), 5))};
if is_ppt
    legend_args = [legend_args, {'FontSize', 14}];
end
lgd = legend(ax(2), h_bler, lbl_bler, legend_args{:});

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
% Not fileparts(dir_path): these folder names contain dots (cfo=0.54,
% R=0.30, ...), which fileparts misreads as a file extension and truncates.
dir_parts = strsplit(dir_path, filesep);
dir_parts = dir_parts(~cellfun(@isempty, dir_parts));
dir_name  = dir_parts{end};
out_name  = fullfile(dir_path, [dir_name, extra_text, '_trk_mi_bler']);
print(fig, [out_name, '.eps'], '-depsc', '-painters');
print(fig, [out_name, '.png'], '-dpng', '-r600');
savefig(fig, [out_name, '.fig']);

% ---- HTML export ----
% Self-contained (image is base64-embedded, no separate file to keep track
% of): opening this in a browser shows the plot filling the window, scaled
% by the browser rather than pasted at a fixed small size the way OneNote
% pastes images -- avoids that shrinking entirely, and Ctrl+scroll zooms it
% like any other web page.
fid = fopen([out_name, '.png'], 'r');
png_bytes = fread(fid, Inf, '*uint8');
fclose(fid);
png_b64 = matlab.net.base64encode(png_bytes);

fid = fopen([out_name, '.html'], 'w');
% max-width/max-height (not width:100%%): fits the whole plot inside the
% window on open -- width:100%% made it wider than the window's aspect
% ratio, pushing the legend below the fold so you had to scroll to see it.
fprintf(fid, ['<!DOCTYPE html>\n<html><head><meta charset="utf-8">' ...
    '<title>%s</title>\n<style>body{margin:0;background:#fff;' ...
    'display:flex;justify-content:center;align-items:center;height:100vh;}' ...
    'img{display:block;max-width:100%%;max-height:100vh;width:auto;height:auto;}' ...
    '</style>\n</head><body>\n' ...
    '<img src="data:image/png;base64,%s" alt="%s">\n</body></html>\n'], ...
    dir_name, png_b64, dir_name);
fclose(fid);

fprintf('Saved %s.{eps,fig,png,html}\n', out_name);
