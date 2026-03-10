% =========================================================
% master_plot_bler.m  –  configure here, then run
% =========================================================

clear; clc; close all;

% ---- User configuration ----
base_name    = 'Completion';   % will look for base_name_0.46 / base_name_0.75 or base_name
extra_text   = '';             % e.g. '_transfer'
root_dir     = 'C:\Projects\Scratchpad\mat_files\';

algs_to_plot     = [1, 2];     % 1=LMMSE, 2=Sphere, 3=DeepRx, 4=DeepSIC
add_snr_target   = false;       % append SNR@10% to legend labels
plot_aug_iter_2  = false;      % plot second aug iteration if available
% ----------------------------

% ---- Detect which directories exist ----
suffixes = {'_0.46', '_0.75'};
dirs     = {};

for i = 1:numel(suffixes)
    candidate = fullfile(root_dir, [base_name, suffixes{i}, extra_text]);
    if isfolder(candidate)
        dirs{end+1} = candidate; %#ok<SAGROW>
    end
end

if isempty(dirs)
    candidate = fullfile(root_dir, [base_name, extra_text]);
    if isfolder(candidate)
        dirs = {candidate};
    else
        error('No valid directory found for base_name="%s"', base_name);
    end
end

fprintf('Found %d curve set(s):\n', numel(dirs));
for i = 1:numel(dirs)
    fprintf('  %s\n', dirs{i});
end
% ----------------------------------------

% ---- Shared style definitions ----
alg_colors = [0.00, 0.45, 0.70;   % LMMSE   - blue
              0.85, 0.33, 0.10;   % Sphere  - vermillion
              0.47, 0.67, 0.19;   % DeepRx  - green
              0.49, 0.18, 0.56];  % DeepSIC - purple

alg_names = {'LMMSE', 'Sphere', 'DeepRx', 'DeepSIC'};
alg_files = {'lmmse', 'sphere', 'deeprx', 'deepsicsb'};

% Markers per algorithm per code-rate set (2 sets max):
%   LMMSE  and Sphere  use fillable markers - shape encodes code rate
%   DeepRx and DeepSIC use non-fillable markers - shape encodes aug state
%
% Layout: markers_no_aug{alg, rc_idx}, markers_aug{alg, rc_idx}
%
%                    rc_idx=1   rc_idx=2
markers_no_aug = {'^',  'v';    % LMMSE   (fillable, hollow)
                  's',  'd';    % Sphere  (fillable, hollow)
                  '*',  '*';    % DeepRx  (non-fillable, same shape both sets)
                  '+',  '+'};  % DeepSIC (non-fillable, same shape both sets)

markers_aug    = {'^',  'v';    % LMMSE   (fillable, filled)
                  's',  'd';    % Sphere  (fillable, filled)
                  'p',  'p';    % DeepRx  (non-fillable)
                  'x',  'x'};  % DeepSIC (non-fillable)

% Fill colors: hollow for no-aug, filled for aug
%   alg 1-2 (fillable): use 'none' vs line color
%   alg 3-4 (non-fillable): always 'none' (ignored by MATLAB anyway)
fillable_algs = [1, 2];   % LMMSE and Sphere
% ----------------------------------

% ---- Create figure ----
figure; hold on; grid on;
all_handles = [];
all_labels  = {};

% ---- Plot each directory set ----
for d = 1:numel(dirs)
    [h, lbl] = plot_bler_set( ...
        dirs{d}, d, ...
        algs_to_plot, alg_colors, alg_names, alg_files, ...
        markers_no_aug, markers_aug, fillable_algs, ...
        add_snr_target, plot_aug_iter_2);

    all_handles = [all_handles, h];
    all_labels  = [all_labels,  lbl];
end

hold off;

legend(all_handles, all_labels, 'Location', 'best', 'Interpreter', 'none');
xlabel('SNR (dB)');
ylabel('BLER');
set(gca, 'YScale', 'log');
set(gcf, 'Units', 'inches', 'Position', [0 0 4 3]);
set(gca, 'LooseInset', get(gca, 'TightInset'));

out_name = fullfile(root_dir, [base_name, extra_text]);
print(gcf, [out_name, '.eps'], '-depsc', '-painters');
savefig([out_name, '.fig']);
