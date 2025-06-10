"""
This script plots the learning behavioral data, for all tasks shown in Fig. 1E
Uses outputs from analysisRL.m and bin_compute_confidence.py.
If long-term memory, plots binned accuracies and conf. intervals, and RL fits
Otherwise just plots binned accuracies
Run for one subject and task at a time, specified below. 
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from bin import plotter

# Choose subject
subject = 'w' # one of 'w', 'je', 'jo'

# Choose task
tasks = ['Probe_4AFC', 'Train_4AFC', 'Train_2AFC', 'Train_2AFC_idtrials']
task = tasks[0] # which task to plot

# Load Data 
data_dir = 'results/learning'
out_dir = 'figures/fig1'
binned_data_path = subject + '_'+task+'_learning_curve_data_1000_all_binned.npz'

# Plotting specifications based on task
if 'Probe' in binned_data_path:
    has_RL_fit = True # plot RL fits
    has_CIs = True # plot confidence intervals
    has_years = True # include year transitions on x axis
    open_c = False # plot closed circles
    if subject == 'je':
        yceil = .8
        plot_size = (1.8,1.6) 
    else:
        yceil = None
        plot_size = (2,1.4) 
else:
    has_RL_fit = False
    has_CIs = False
    yceil = None
    open_c = True
    if '4AFC' in binned_data_path or 'idtrials' in binned_data_path:
        plot_size = (.78, .58)
    else:
        plot_size = (.18, .58)

# Load binned data 
plot_data = np.load(os.path.join(data_dir, binned_data_path))

# Extract binned accuracies
x = [plot_data['color_x'], plot_data['shape_x']]
y = [plot_data['color_accs'], plot_data['shape_accs']]
bin_data = np.array([x,y])

# If applicable, extract binned accuracy confidence intervals
if has_CIs: 
    x_ci = [plot_data['color_i'], plot_data['shape_i']]
    y_ci = [plot_data['color_ci'], plot_data['shape_ci']]
    ci_data = [x_ci, y_ci]
else: 
    ci_data = None

# If applicable, extract RL fits
if has_RL_fit:
    color_mat = loadmat(os.path.join(data_dir, subject+'_Probe_4AFC_choose_color_RLfit.mat'))
    shape_mat = loadmat(os.path.join(data_dir, subject+'_Probe_4AFC_choose_shape_RLfit.mat'))
    x_fit = [[i for i in range(len(color_mat['mvAvgModel']))], [i for i in range(len(shape_mat['mvAvgModel']))]]
    y_fit = [np.squeeze(color_mat['mvAvgModel']), np.squeeze(shape_mat['mvAvgModel'])] 
    fit_data = [x_fit, y_fit]
else: 
    fit_data = None
    
# If applicable, extract years
if has_years:
    year_labels = plot_data['bin_year']
    if year_labels.shape > x[0].shape: # if more years than bins, drop last, years are aligned to the first bin
        year_labels = year_labels[:x[0].shape[0]] 
    # Only want to plot a year mark at the start of each year
    year_changes = [i for i in range(year_labels.shape[0]) if year_labels[i] != year_labels[i-1]] # which bins are year transitions
    include_years = year_labels[year_changes] # keep those years
    year_bins = x[0][year_changes] # get corresponding trial number (bin) values
    year_data = [year_bins, include_years]
else:
    year_data = None

# Where to save plot out
data_name = str(binned_data_path).split('.')[0]
out_name =  data_name + '_plot'

# Plot
fig, axs = plt.subplots(figsize = plot_size)
test_plot = plotter.create_save_learning_curve(axs, fig, out_name, bin_data, ci_data=ci_data, fit_data=fit_data, year_data = year_data, 
                                       group_labels=['color', 'shape'], x=None, yceil=yceil, xposition=0., out_dir=out_dir,open_c=open_c,set_size=plot_size)
