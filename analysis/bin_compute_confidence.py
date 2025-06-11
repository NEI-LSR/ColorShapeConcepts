"""
This script takes trial learning data, bins it, computes accuracies in each
bin and computes confidence intervals using bootstrapping. 
Outputs numpy arrays to use to plot learning curves. 
Run for one subject and task at a time.
Tasks:
Probe_4AFC - long term mem trials, main plots in Fig. 1E
Train_2AFC - short term mem trials, insets in Fig. 1E
Train_4AFC - short term mem trials, insets in Fig. 1E
Train_2AFC_idtrials - control trials, Fig. S1
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Choose subject
subject = 'w' # one of 'w', 'je', 'jo'

# Choose task
tasks = ['Probe_4AFC', 'Train_2AFC_idtrials', 'Train_4AFC', 'Train_2AFC'] # options
task = tasks[1] # which task to bin data for

# Set directories
data_dir = 'data/learning_data'
out_dir = 'results/learning'
# Load Data 
subject_data_path = os.path.join(data_dir, subject + '_' + task + '.csv')
subject_data = pd.read_csv(subject_data_path)

# Get year of each trial for plotting later 
try:
    subject_data['year'] = [datetime.fromtimestamp(x/1000).strftime("%Y") for x in subject_data['timestamp']]
except:
    subject_data['year'] = '2016' # timestamp not currently in Train 2AFC id trial csvs, but all trials were in 2016

# Split into color and shape trials on the basis of the choice
# E.g, Probe_4AFC choose_shape means cued color, chose shape, but for Train_4AFC it means cued colored shape, chose shape
choose_shape_trials = subject_data[subject_data['is_choice_color']==0].reset_index(drop=True) 
choose_color_trials = subject_data[subject_data['is_choice_color']==1].reset_index(drop=True) 

# Bin trials 
# Define how many bins to use
if task == 'Train_2AFC_idtrials':
    n_in_bin = 50 # smaller than probe because many fewer trials 
elif task == 'Train_2AFC':
    n_in_bin = 75 # smaller than probe because many fewer trials 
else:
    n_in_bin = 500

# Bin data and get nested list containing the outcome values (0 or 1) of all trials in that bin
binned_choose_shape = [choose_shape_trials['chose_correct'][i:i+n_in_bin] for i in range(0, len(choose_shape_trials), n_in_bin)]
binned_choose_color = [choose_color_trials['chose_correct'][i:i+n_in_bin] for i in range(0, len(choose_color_trials), n_in_bin)]

# For each bin, approximate which year most trials in that bin were completed in, for plotting later
bin_year = [choose_shape_trials['year'][i:i+n_in_bin] for i in range(0, len(choose_shape_trials), n_in_bin)] 
bin_year_mode = [x.mode()[0] for x in bin_year] 

# Deal with last bins - may have few trials and one trial type may have one more bin than another
n_bins = np.min([len(binned_choose_shape), len(binned_choose_color)]) # min n bins shared by both trial types
if binned_choose_shape[n_bins-1].shape[0] < 10 or binned_choose_color[n_bins-1].shape[0] < 10:
    n_bins = n_bins-1 # if either final bin has very few trials, don't include in plot

# For each bin, bootstrap the accuracy 1000 times
n_boots = 1000
shape_color = np.zeros((2, 3, n_bins)) # <trial type, metric, bin number>
# For each trial type
for t, trial_type in enumerate([binned_choose_shape, binned_choose_color]):
    trial_type_binned = trial_type[:n_bins] 
    # For each bin
    for l, b in enumerate(trial_type_binned):
        boot_accs = []
        for i in range(n_boots):
            sample = np.random.choice(b, size=n_in_bin) # array of 0s and 1s, resample to bin size
            boot_accs.append(sample.mean()) # calculate accuracy for that sample of trials
        boot_mean_acc = np.array(boot_accs).mean() # accuracy at trial l
        boot_lcb = np.quantile(boot_accs, q=.025) # lower confidence bound of acccuracy at trial l
        boot_ucb = np.quantile(boot_accs, q=.975) # upper confidence bound of accuracy at trial l
        shape_color[t,0,l] = boot_mean_acc
        shape_color[t,1,l] = boot_lcb
        shape_color[t,2,l] = boot_ucb

# To preserve trial number as x axis, get trial number each bin would be centered on
x_vals = np.min([choose_shape_trials.shape[0],choose_color_trials.shape[0]])
use_x = list(range(int(n_in_bin/2),x_vals+int(n_in_bin/2), n_in_bin))
use_x = use_x[:n_bins]

# Save out all accuracies, confidence intervals, x ticks, and years
shape_accs = shape_color[0, 0, :]
color_accs = shape_color[1, 0, :]
shape_ci = shape_color[0, 1:,:].T
color_ci = shape_color[1, 1:,:].T
array_out_name = subject + '_'+task+'_learning_curve_data_'+str(n_boots)+'_all_binned.npz'
out_path = os.path.join(out_dir, array_out_name)
np.savez(out_path, color_x=use_x,color_accs=color_accs,
         shape_x=use_x,shape_accs=shape_accs,color_i=use_x,
         color_ci=color_ci,shape_i=use_x,shape_ci=shape_ci, bin_year = bin_year_mode)
