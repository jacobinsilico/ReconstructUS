% --------------------------------
% Batch UFF to .mat Preprocessing Script
% --------------------------------
% Converts all .uff files in input_dir to two separate .mat files:
% - rf_XXXXX.mat containing pre-beamformed RF data
% - img_XXXXX.mat containing beamformed image
% This assumes each .uff contains a single beamformed image + 75 PW RF frames.

clear; close all; clc;

% -----------------------------
% Settings
% -----------------------------
input_dir  = ...;
output_dir = ...;

% get list of all .uff files in input_dir
uff_files = dir(fullfile(input_dir, '*.uff'));
num_files = length(uff_files);

if num_files == 0
    error('No .uff files found in input_dir.');
end

disp(['Found ', num2str(num_files), ' UFF files.']);

% -----------------------------
% Processing Loop
% -----------------------------
pair_idx = 1;

for i = 1:num_files
    % full path to current .uff file
    uff_path = fullfile(input_dir, uff_files(i).name);
    disp(['Processing: ', uff_files(i).name]);

    % read metadata
    metadata = uff.index(uff_path);

    % extract expected objects (this assumes fixed ordering)
    bmode_data    = uff.read_object(uff_path, metadata{1}.location, true); % uff.beamformed_data
    channel_data  = uff.read_object(uff_path, metadata{2}.location, true); % uff.channel_data
    scan          = uff.read_object(uff_path, metadata{3}.location, true); % uff.linear_scan

    % extract dimensions for reshaping
    n_x = scan.N_x_axis;
    n_z = scan.N_z_axis;

    % ---- RF data ----
    % data format: [depth, channels, plane waves] = [samples, 128, 75]
    rf_raw = channel_data(1).data;

    % ---- Beamformed image ----
    % reshape from vector to [lateral, depth]
    img_raw = bmode_data(1).data;
    img = reshape(img_raw, [n_z, n_x])';  % transpose to get [lateral, depth]

    % ---- Save both ----
    rf_path  = fullfile(output_dir, sprintf('rf_%05d.mat', pair_idx));
    img_path = fullfile(output_dir, sprintf('img_%05d.mat', pair_idx));

    save(rf_path, 'rf_raw');
    save(img_path, 'img');

    disp(['Saved pair ', num2str(pair_idx)]);
    pair_idx = pair_idx + 1;
end

disp('✅ Done. All .uff files processed and saved.');
