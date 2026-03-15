% =========================================================
% master_plot_bler.m  –  configure here, then run
% =========================================================

clear; clc; close all;

% ---- User configuration ----
base_name        = 'Completion';
extra_text       = '';             % e.g. '_transfer'
root_dir         = 'C:\Projects\Scratchpad\mat_files\';

algs_to_plot     = [1 2];        % 1=LMMSE, 2=Sphere, 3=DeepRx, 4=DeepSIC
add_snr_target   = false;         % append SNR@10% to legend labels
plot_aug_iter_2  = false;         % plot second aug iteration if available
snr_pad_left_db  = 0;             % extend SNR axis to the left by this many dB (0 = no padding)
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
              0.85, 0.33, 0.10;   % Sphere  - vermillion
              0.47, 0.67, 0.19;   % DeepRx  - green
              0.49, 0.18, 0.56];  % DeepSIC - purple

alg_names = {'LMMSE', 'RBSD', 'DeepRx', 'DeepSIC'};
alg_files = {'lmmse', 'sphere', 'deeprx', 'deepsicsb'};

%                    rc_idx=1   rc_idx=2
markers_no_aug = {'^',  'v';    % LMMSE   (fillable, hollow)
                  's',  'd';    % Sphere  (fillable, hollow)
                  '*',  '*';    % DeepRx  (non-fillable)
                  '+',  '+'};  % DeepSIC (non-fillable)

markers_aug    = {'^',  'v';    % LMMSE   (fillable, filled)
                  's',  'd';    % Sphere  (fillable, filled)
                  'p',  'p';    % DeepRx  (non-fillable)
                  'x',  'x'};  % DeepSIC (non-fillable)

fillable_algs = [1, 2];
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
        add_snr_target, plot_aug_iter_2, snr_pad_left_db);

    all_handles = [all_handles, h];
    all_labels  = [all_labels,  lbl];
end

hold off;

legend(all_handles, all_labels, 'Location', 'best', 'Interpreter', 'none');
xlabel('SNR (dB)');
ylabel('BLER');
set(gca, 'YScale', 'log');
set(gcf, 'Units', 'inches', 'Position', [0 0 5 3.5]);
set(gca, 'LooseInset', get(gca, 'TightInset'));

out_name = fullfile(root_dir, [base_name, extra_text]);
print(gcf, [out_name, '.eps'], '-depsc', '-painters');
savefig([out_name, '.fig']);
