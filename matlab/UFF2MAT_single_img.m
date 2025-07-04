% --------------------------------
% UFF to .mat Preprocessing Script
% --------------------------------
% This script is used to process .uff files into .mat files for further
% processing in Python. It assumes .uff files are already available in the
% input_dir and that they contain pre-beamformed RF data as well as already
% reconstructed ultrasound image within the same .uff file. This is a
% prototype that processes only single image. 

% prepare the command window
clear; close all; clc;

% set the input and output directory
input_dir  = ...;
output_dir = ...;

% read metadata of the .uff files and list the contents
% we expect uff.beamformed_data, uff.channel_data and uff.linear_scan
metadata = uff.index(input_dir);
for i = 1:length(metadata)
    disp(['Index ', num2str(i), ': ', metadata{i}.class]);
end

% read .uff objects
bmode_data    = uff.read_object(input_dir, metadata{1}.location, true); % uff.beamformed_data
channel_data  = uff.read_object(input_dir, metadata{2}.location, true); % uff.channel_data
scan          = uff.read_object(input_dir, metadata{3}.location, true); % uff.linear_scan

% extract image dimensions
n_x = scan.N_x_axis;
n_z = scan.N_z_axis;

% save first RF frame and reconstructed image
rf_raw = channel_data(1).data;              % [depth, channels, plane waves]
img_raw = bmode_data(1).data;
img = reshape(img_raw, [n_z, n_x])';        % reshape and transpose to [x, z]

% save to .mat
rf_path  = fullfile(output_dir, 'rf_00001.mat');
img_path = fullfile(output_dir, 'img_00001.mat');
save(rf_path, 'rf_raw');
save(img_path, 'img');


