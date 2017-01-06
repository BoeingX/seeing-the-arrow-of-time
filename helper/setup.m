% SETUP  Add the required search paths to MATLAB
if exist('vl_version') ~= 3, run('vlfeat-0.9.20/toolbox/vl_setup') ; end
addpath(genpath('export_fig'));
addpath(genpath('helper'));
addpath(genpath('eccv2004Matlab'));
