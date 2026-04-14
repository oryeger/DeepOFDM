% generate_quadriga_channels.m
%
% Generates QuaDRiGa time-domain channel CIR files (.mat) for all three
% scenario types (UMa, UMi, RMa) and each configured seed.
%
% Output files are saved to <quadriga_mat_path> (from config.yaml) as:
%   UMa_seed_<N>.mat, UMi_seed_<N>.mat, RMa_seed_<N>.mat
%
% These files are loaded by python_code/channel/Quadriga/quadriga_channel.py
% when channel_model is set to 'UMa', 'UMi', or 'RMa' in config.yaml.
%
% Reads common params (n_ants, n_users, carrier_frequency, seeds, quadriga_mat_path)
%   from config.yaml.
% Reads QuaDRiGa simulation params from quadriga_config.yaml.
%
% Usage:
%   Run this script from MATLAB before running Python evaluation whenever:
%     - n_ants or n_users is changed
%     - any quadriga_* config key in quadriga_config.yaml is changed
%     - new seeds are needed
%
% Output per scenario+seed:
%   coeff  - [n_rx_ant, n_users, n_paths]  complex double, per-cluster CIR coefficients
%   delay  - [n_users, n_paths]            double, excess path delays in seconds (τ_min subtracted)

clear; close all;

%% ---- Locate project root and add QuaDRiGa to path ----------------------
script_dir   = fileparts(mfilename('fullpath'));          % escnn_matlab/
project_root = fileparts(script_dir);                    % project root
quadriga_src = fullfile(script_dir, 'QuaDRiGa-NG', 'quadriga_src');
addpath(quadriga_src);
config_file          = fullfile(project_root, 'python_code', 'config.yaml');
quadriga_config_file = fullfile(project_root, 'python_code', 'quadriga_config.yaml');

%% ---- Parse config files -------------------------------------------------
cfg  = parse_yaml(config_file);           % common params
qcfg = parse_yaml(quadriga_config_file);  % QuaDRiGa simulation params

n_ants        = get_int(cfg, 'n_ants',        8);
n_users       = get_int(cfg, 'n_users',       4);
carrier_freq  = get_float(cfg, 'carrier_frequency', 2.6e9);
channel_seed  = get_int(cfg, 'channel_seed',  17);
pilot_seed    = get_int(cfg, 'pilot_channel_seed', -1);
mat_path_rel  = get_str(cfg,  'quadriga_mat_path', '../Scratchpad/quadriga_channels');

bs_height     = get_float(qcfg, 'quadriga_bs_height',       25.0);
dist_range    = get_float_vec(qcfg, 'quadriga_distance_range',  [50, 500]);
ue_h_range    = get_float_vec(qcfg, 'quadriga_ue_height_range', [1.5, 1.8]);
spread_angle  = get_float(qcfg, 'quadriga_ue_spread_angle', 120.0);
indoor_prob   = get_float(qcfg, 'quadriga_indoor_probability', 0.0);
n_paths       = get_int(qcfg,   'quadriga_num_paths',       12);

%% ---- Scenario type → QuaDRiGa scenario string mapping ------------------
scenario_types = {'UMa', 'UMi', 'RMa'};
scenario_map = containers.Map( ...
    {'UMa', 'UMi', 'RMa'}, ...
    {'3GPP_38.901_UMa_NLOS', '3GPP_38.901_UMi_NLOS', '3GPP_38.901_RMa_NLOS'});

% Resolve mat_path: absolute used as-is; relative resolved from project root
if ~isempty(regexp(mat_path_rel, '^[A-Za-z]:[/\\]', 'once')) || startsWith(mat_path_rel, '/')
    mat_path = mat_path_rel;   % already absolute
else
    % Resolve relative path and normalise .. components via Java
    mat_path = char(java.io.File(fullfile(project_root, mat_path_rel)).getCanonicalPath());
end
if ~exist(mat_path, 'dir')
    mkdir(mat_path);
    fprintf('Created output directory: %s\n', mat_path);
end

%% ---- Determine seeds to generate ----------------------------------------
seeds_to_gen = channel_seed;
if pilot_seed >= 0 && pilot_seed ~= channel_seed
    seeds_to_gen(end+1) = pilot_seed;
end
seeds_to_gen = unique(seeds_to_gen);

fprintf('Generating QuaDRiGa channels\n');
fprintf('  Scenarios  : UMa, UMi, RMa\n');
fprintf('  Seeds      : %s\n', num2str(seeds_to_gen));
fprintf('  n_ants     : %d\n', n_ants);
fprintf('  n_users    : %d\n', n_users);
fprintf('  n_paths    : %d\n', n_paths);
fprintf('  Carrier    : %.2f GHz\n', carrier_freq / 1e9);
fprintf('  Output dir : %s\n\n', mat_path);

%% ---- Build BS array (UPA: N_H=2 horiz x M_V vert, cross-pol ±45°) ------
assert(n_ants >= 4 && mod(n_ants, 4) == 0, ...
    sprintf('n_ants must be a multiple of 4 and >= 4, got %d. Layout: 2H x (n_ants/4)V x 2 cross-pol.', n_ants));

M_V = n_ants / 4;   % vertical elements per polarization branch
N_H = 2;            % horizontal elements (fixed)

% pol=3: cross-polarization ±45°, matching 3GPP TR 38.901
% Total ports = M_V * N_H * 2 polarizations = n_ants
bs_array = qd_arrayant('3gpp-3d', M_V, N_H, carrier_freq, 3);

%% ---- UE array (single omni antenna per user) ----------------------------
ue_array = qd_arrayant('omni');
ue_array.center_frequency = carrier_freq;

%% ---- Generate one channel file per scenario × seed ---------------------
total = numel(scenario_types) * numel(seeds_to_gen);
counter = 0;
for sc_idx = 1 : numel(scenario_types)
    sc_type    = scenario_types{sc_idx};
    scenario   = scenario_map(sc_type);

for seed_idx = 1 : numel(seeds_to_gen)
    seed = seeds_to_gen(seed_idx);
    out_file = fullfile(mat_path, sprintf('%s_seed_%d.mat', sc_type, seed));
    counter = counter + 1;

    fprintf('[%d/%d] %s seed %d -> %s\n', counter, total, sc_type, seed, out_file);

    rng(seed);   % deterministic QuaDRiGa output

    %% Set up simulation parameters
    s = qd_simulation_parameters;
    s.center_frequency = carrier_freq;
    s.show_progress_bars = 0;
    s.use_3GPP_baseline = 1;  % Use statistical 3GPP delay model (not scatterer geometry)

    %% Set up layout: BS = 1 TX, UEs = n_users RX (downlink convention)
    % Channel reciprocity means H is the same for uplink detection.
    l = qd_layout(s);
    l.no_tx = 1;          % BS as transmitter
    l.no_rx = n_users;    % UEs as receivers

    l.tx_array = bs_array;
    for u = 1 : n_users
        l.rx_array(1, u) = ue_array;
    end

    % BS at origin at configured height
    l.tx_position = [0; 0; bs_height];

    % Place UEs randomly within the angular spread and distance range
    for u = 1 : n_users
        % Angle uniformly distributed within spread_angle centred at 0 deg
        angle_u = (rand() - 0.5) * spread_angle * pi / 180;   % [rad]
        dist_u  = dist_range(1) + rand() * (dist_range(2) - dist_range(1));
        ue_h_u  = ue_h_range(1) + rand() * (ue_h_range(2) - ue_h_range(1));
        l.rx_track(1, u).initial_position = [dist_u * cos(angle_u); dist_u * sin(angle_u); ue_h_u];

        % Scenario and indoor state per UE
        is_indoor = rand() < indoor_prob;
        if is_indoor
            l.rx_track(1, u).scenario = [scenario, '_O2I'];
        else
            l.rx_track(1, u).scenario = scenario;
        end
    end

    %% Generate channels
    channels = l.get_channels();   % [n_rx=n_users, n_tx=1]

    %% Collapse per-antenna delays back to per-cluster format.
    % The merge() step inside l.get_channels() sets individual_delays=true,
    % which expands delay from [n_clusters, n_snap] to
    % [n_rx_ant, n_tx_ant, n_clusters, n_snap] by replicating each cluster's
    % delay for every antenna pair.  This must be collapsed before extraction
    % so that squeeze(delay) yields a vector, not a matrix.
    for u = 1 : n_users
        channels(u, 1).individual_delays = false;
    end

    %% Debug: print raw channel info for user 1
    ch_dbg = channels(1, 1);
    fprintf('   DEBUG: coeff size = [%s], delay size = [%s]\n', ...
        num2str(size(ch_dbg.coeff)), num2str(size(ch_dbg.delay)));
    d_raw = squeeze(ch_dbg.delay);
    fprintf('   DEBUG: User 1 raw delays (ns) = %s\n', num2str(d_raw(:)' * 1e9, '%.1f '));
    fprintf('   DEBUG: Distinct delays = %d\n', numel(unique(round(d_raw * 1e12))));

    %% Extract per-cluster CIR for each UE
    % After collapsing individual_delays:
    %   channels(u,1).coeff : [n_ue_ant=1, n_bs_ant=n_ants, n_clusters, n_snap=1]
    %   channels(u,1).delay : [n_clusters, n_snap=1]
    %
    % With use_3GPP_baseline=1, delays come from the 3GPP statistical model
    % (random exponential, same as Sionna), not from scatterer geometry.

    % Determine actual cluster count from the first channel
    n_clusters_actual = size(channels(1,1).coeff, 3);
    n_save = max(n_paths, n_clusters_actual);   % save all clusters
    coeff_all = zeros(n_ants, n_users, n_save, 'like', 1+1i);
    delay_all = zeros(n_users, n_save);

    for u = 1 : n_users
        ch_u    = channels(u, 1);
        c_u     = ch_u.coeff;    % [1, n_ants, n_cl, 1]
        d_u     = ch_u.delay;    % [n_cl, 1]

        p_u = size(c_u, 3);     % actual number of clusters
        p_use = min(p_u, n_save);

        % Sort paths by delay (ascending)
        [d_sorted, sort_idx] = sort(squeeze(d_u));
        c_sorted = squeeze(c_u(1, :, :, 1));   % [n_ants, n_cl]
        c_sorted = c_sorted(:, sort_idx);

        % Subtract minimum delay so paths start near τ=0 (excess delay only)
        d_min = d_sorted(1);
        d_sorted = d_sorted - d_min;

        coeff_all(:, u, 1:p_use) = c_sorted(:, 1:p_use);
        delay_all(u, 1:p_use)    = d_sorted(1:p_use);
    end

    fprintf('   Clusters from QuaDRiGa: %d, saving: %d\n', n_clusters_actual, n_save);

    %% Save
    coeff = coeff_all;   % [n_rx_ant, n_users, n_save]  complex
    delay = delay_all;   % [n_users, n_save]             seconds (excess delay)
    n_paths_saved = n_save;
    save(out_file, 'coeff', 'delay', 'n_ants', 'n_users', 'n_paths_saved', ...
         'carrier_freq', 'sc_type', 'scenario', 'seed', '-v7');

    fprintf('   Saved: coeff [%dx%dx%d], delay [%dx%d]\n', ...
        size(coeff, 1), size(coeff, 2), size(coeff, 3), ...
        size(delay, 1), size(delay, 2));

end  % seed loop
end  % scenario loop

fprintf('\nDone. %d channel file(s) written to:\n  %s\n', total, mat_path);


%% =========================================================================
%% Helper functions
%% =========================================================================

function cfg = parse_yaml(filename)
% Very simple YAML parser for flat key: value config files.
% Returns a struct with string values for every key.
    cfg = struct();
    fid = fopen(filename, 'r');
    if fid < 0
        error('Cannot open config file: %s', filename);
    end
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if isempty(line) || line(1) == '#'
            continue;
        end
        % Strip inline comment
        comment_pos = strfind(line, ' #');
        if ~isempty(comment_pos)
            line = strtrim(line(1 : comment_pos(1) - 1));
        end
        sep = strfind(line, ':');
        if isempty(sep)
            continue;
        end
        key = strtrim(line(1 : sep(1) - 1));
        val = strtrim(line(sep(1) + 1 : end));
        % Remove surrounding quotes
        val = strrep(val, '''', '');
        val = strrep(val, '"', '');
        % Make valid struct fieldname
        key = strrep(key, '-', '_');
        cfg.(key) = val;
    end
    fclose(fid);
end

function v = get_int(cfg, key, default)
    if isfield(cfg, key) && ~isempty(cfg.(key))
        v = round(str2double(cfg.(key)));
    else
        v = default;
    end
end

function v = get_float(cfg, key, default)
    if isfield(cfg, key) && ~isempty(cfg.(key))
        v = str2double(cfg.(key));
    else
        v = default;
    end
end

function v = get_str(cfg, key, default)
    if isfield(cfg, key) && ~isempty(cfg.(key))
        v = cfg.(key);
    else
        v = default;
    end
end

function v = get_float_vec(cfg, key, default)
% Parse a YAML inline list like "[50, 500]" into a numeric row vector.
    if ~isfield(cfg, key) || isempty(cfg.(key))
        v = default;
        return;
    end
    s = strtrim(cfg.(key));
    s = strrep(s, '[', '');
    s = strrep(s, ']', '');
    parts = strsplit(s, ',');
    v = cellfun(@(x) str2double(strtrim(x)), parts);
end
