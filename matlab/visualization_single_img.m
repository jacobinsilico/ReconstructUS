% ---------------------------------
% Load and display saved .mat image
% ---------------------------------
% This script is used to check correctness of UFF2MAT_single_img.m script.
% If everything is correct, this script should load and display an US image
% in .mat format saved after running the mentioned script.

% prepare the command window
clear; close all; clc;

% define the path to visualization example
output_dir = ...;
img_path = fullfile(output_dir, 'img_00001.mat'); %adjust this for new examples

% load image from .mat file
loaded = load(img_path);   % this loads a struct with field 'img'
img_loaded = loaded.img;   % access the actual matrix

% convert complex to magnitude
img_mag = abs(img_loaded);  

% normalize and display in dB scale
img_db = 20 * log10(img_mag / max(img_mag(:)));

% display
figure;
imagesc(img_db);
axis image;
colormap gray;
clim([-60 0]);  % adjust dynamic range as needed
xlabel('Lateral [pixels]');
ylabel('Depth [pixels]');
title('DAS Reconstructed Image (.mat)');