%------------------------------
% PICMUS Data Exploration Script
%-------------------------------
% This script is used to explore the data in .uff files available on the
% PICMUS webpage. Script reads the data, checks the dimensionality and
% displays DAS-reconstructed ultrasound image.

% prepare the command window
clear; close all; clc;

% prepare the filename and absolute filepath
% list of available .uff files
file_list = {
    'Alpinion_L3-8_CPWC_hyperechoic_scatterers.uff';
    'Alpinion_L3-8_CPWC_hypoechoic.uff';
    'PICMUS_carotid_cross.uff';
    'PICMUS_carotid_long.uff';
    'PICMUS_experiment_contrast_speckle.uff';
    'PICMUS_experiment_resolution_distortion.uff';
    'PICMUS_simulation_contrast_speckle.uff';
    'PICMUS_simulation_resolution_distortion.uff'
};

% Index of the file you want to load (CHANGE ONLY THIS)
file_idx = 8;  % choose a number between 1 and 8

% Build full path
data_dir = '\Dataset_UFF';
filename = file_list{file_idx};
filepath = fullfile(data_dir, filename);

% Display selected file
disp(['Selected file: ', filename]);
disp(['Full path: ', filepath]);

metadata = uff.index(filepath);  % gives a cell array of structs

% display class types stored in the .uff file
for i = 1:length(metadata)
    disp(['Index ', num2str(i), ': ', metadata{i}.class]);
end

% load all 3 main objects
bmode_data    = uff.read_object(filepath, metadata{1}.location, true); % uff.beamformed_data
channel_data  = uff.read_object(filepath, metadata{2}.location, true); % uff.channel_data
scan          = uff.read_object(filepath, metadata{3}.location, true); % uff.linear_scan

% print info
num_images = length(bmode_data);
disp(['Number of reconstructed images: ', num2str(num_images)]);

num_rf_frames = length(channel_data);
disp(['Number of RF frames: ', num2str(num_rf_frames)]);

size(channel_data(1).data)  % inspect one RF frame
size(bmode_data(1).data)    % inspect one B-mode image

% display the image already saved in .uff file
n_x = scan.N_x_axis;
n_z = scan.N_z_axis;
raw = bmode_data(1).data;

img = reshape(raw, [n_z, n_x])';  % [z, x] → transpose to [x, z]

x = scan.x_axis;
z = scan.z_axis;

imagesc(x*1e3, z*1e3, 20*log10(abs(img)/max(abs(img(:)))));
axis image;
colormap gray;
clim([-60 0]);
xlabel('Lateral [mm]');
ylabel('Depth [mm]');
title('DAS Beamformed Image Read (UFF File)');
