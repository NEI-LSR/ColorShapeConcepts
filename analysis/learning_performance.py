"""
This script estimates accuracies and confidence intervals for initial and 
plateau performance on the behavioral tablet tasks as reported in the text.
"""
import pandas as pd
import numpy as np
import os

# Set directories
data_dir = 'data/learning_data' # trial learning data csvs
out_dir = 'results/learning'

#######################
# PLATEAU PERFORMANCE #
#######################

# Check if plateau performance on different trial types were similar 
# Check for: long-term memory 4AFC ('Probe_4AFC'), 
# short-term memory color-to-color, shape-to-shape trials, ('Train_2AFC_idtrials'),
# and short-term memory 4AFC trials ('Train_4AFC')

tasks = ['Probe_4AFC', 'Train_2AFC_idtrials', 'Train_4AFC']
last_n_trials = 1000 # calculate accuracy and confidence interval in the last 1000 trials
n_boots = 1000 # number of bootstrap iterations
# For each task
for task in tasks:
    if task == 'Train_2AFC_idtrials':
        use_subjects = ['w', 'je'] # Jo did not receive these trials
    else:
        use_subjects = ['w', 'je', 'jo']
    plateau_perform = []
    # For each subject
    for subject in use_subjects:
        subject_data_path = os.path.join(data_dir, subject + '_' + task + '.csv')
        subject_data = pd.read_csv(subject_data_path)
        
        # Split into color and shape trials
        choose_shape_trials = subject_data[subject_data['is_choice_color']==0].reset_index(drop=True) # chose shape
        choose_color_trials = subject_data[subject_data['is_choice_color']==1].reset_index(drop=True) # chose color
        
        # Grab last n trials
        shape = choose_shape_trials['chose_correct'].to_numpy()[len(choose_shape_trials)-last_n_trials:]
        color = choose_color_trials['chose_correct'].to_numpy()[len(choose_color_trials)-last_n_trials:]
        
        # Bootstrap accuracy in last 1000 trials
        shape_samples = []
        color_samples = []
        for t in range(1000):
            shape_samp = np.random.choice(shape, last_n_trials, replace=True)
            color_samp = np.random.choice(color, last_n_trials, replace=True)
            shape_samples.append(shape_samp.mean()) # get accuracy within sample
            color_samples.append(color_samp.mean())
        color_mean = np.mean(color_samples) # chose color accuracy
        color_lcb, color_ucb = np.quantile(color_samples, .025), np.quantile(color_samples, .975) # confidence bounds
        shape_mean = np.mean(shape_samples) #  chose shape accuracy
        shape_lcb, shape_ucb = np.quantile(shape_samples, .025), np.quantile(shape_samples, .975) # confidence bounds
        
        plateau_perform.append([subject, color_mean, color_lcb, color_ucb, shape_mean, shape_lcb, shape_ucb])
    
    if task == 'Probe_4AFC':
        title_cols = ['subject','shape_to_color_mean', 'shape_to_color_lower_cb', 'shape_to_color_upper_cb', 'color_to_shape_mean', 'color_to_shape_lower_cb', 'color_to_shape_upper_cb']
    elif task == 'Train_4AFC':
        title_cols = ['subject','coloredshape_to_color_mean', 'coloredshape_to_color_lower_cb', 'coloredshape_to_color_upper_cb', 'coloredshape_to_shape_mean', 'coloredshape_to_shape_lower_cb', 'coloredshape_to_shape_upper_cb']
    else:
        title_cols = ['subject','color_to_color_mean', 'color_to_color_lower_cb', 'color_to_color_upper_cb', 'shape_to_shape_mean', 'shape_to_shape_lower_cb', 'shape_to_shape_upper_cb']
    
    plateau_df = pd.DataFrame(plateau_perform, columns=title_cols)
    df_out = os.path.join(out_dir, task + 'plateau_performance_last_'+str(last_n_trials)+'_trials.csv')
    plateau_df.to_csv(df_out, index=False)



#######################
# INITIAL PERFORMANCE #
#######################
# Check if performance on first long-term memory 2AFC trials was above chance
# Load Data 
task = 'Probe_2AFC'
n_boots = 1000 # number of bootstrap iterations
initial_perform = []
for subject in ['w', 'je']:
    subject_data_path = os.path.join(data_dir, subject + '_' + task + '.csv')
    subject_data = pd.read_csv(subject_data_path)
    
    # Keep only sessions before short-term memory, matching trials began
    subject_data = subject_data[subject_data['days_from_20160101']<27].reset_index(drop=True)
    
    # Split into color and shape trials
    choose_shape_trials = subject_data[subject_data['is_choice_color']==0].reset_index(drop=True) # chose shape
    choose_color_trials = subject_data[subject_data['is_choice_color']==1].reset_index(drop=True) # chose color
    
    # Get array of trial outcomes 
    shape = choose_shape_trials['chose_correct'].to_numpy()
    color = choose_color_trials['chose_correct'].to_numpy()
    # Get number of trials 
    shape_n = shape.shape[0]
    color_n = color.shape[0]
    
    # Bootstrap accuracy and estimate 95% confidence interval
    shape_samples = []
    color_samples = []
    for t in range(n_boots):
        shape_samp = np.random.choice(shape, shape_n, replace=True)
        color_samp = np.random.choice(color, color_n, replace=True)
        shape_samples.append(shape_samp.mean()) # get accuracy within sample
        color_samples.append(color_samp.mean())
    color_mean = np.mean(color_samples) # shape-to-color accuracy
    color_lcb, color_ucb = np.quantile(color_samples, .025), np.quantile(color_samples, .975) # confidence bounds
    shape_mean = np.mean(shape_samples) # color-to-shape trial accuracy
    shape_lcb, shape_ucb = np.quantile(shape_samples, .025), np.quantile(shape_samples, .975) # confidence bounds
    initial_perform.append([subject, color_mean, color_lcb, color_ucb, color_n, shape_mean, shape_lcb, shape_ucb, shape_n])

# Format to dataframe and save to results folder
title_cols = ['subject','shape_to_color_mean', 'shape_to_color_lower_cb', 'shape_to_color_upper_cb', 'shape_to_color_n_trials', 'color_to_shape_mean', 'color_to_shape_lower_cb', 'color_to_shape_upper_cb', 'color_to_shape_n_trials']
initial_df = pd.DataFrame(initial_perform, columns=title_cols)
df_out = os.path.join(out_dir, task + 'initial_performance.csv')
initial_df.to_csv(df_out, index=False)
