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
import matplotlib.pyplot as plt

# make sure text is saved in svgs as text, not path
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.serif'] = ['Times']
plt.rcParams['lines.linewidth'] = .75 * .4


# Choose task
probe_task = False
nafc = [4]
# Set directories
data_dir = 'data/learning_data'
out_dir = 'results/learning'

if probe_task:
    bin_size = [1000]
    task_names = ['_Probe_4AFC']
    plot_size = (3.5, 2)
    name = 'probe'
else:
    if len(nafc)==2:
        bin_size = [200,1000]
        task_names = ['_Train_2AFC', '_Train_4AFC']
        plot_size = (2.5,2) #(3.5, 2)
        name = 'train'
    elif len(nafc) == 1 and nafc[0] == 4:
        bin_size = [1000]
        task_names = ['_Train_4AFC']
        plot_size = (2.4,2)#(2.7 , 2)
        name = 'train4afc'
    else:
        bin_size = [200]
        task_names = ['_Train_2AFC']
        plot_size = (.7,2)#(.8, 2)
        name = 'train2afc'
    
x = []
y = []
y_ci = []
year_ticks = []
for subject in ['w', 'je']:
    subj_x = []
    subj_y = []
    subj_y_ci = []
    subj_year_ticks = []
    subj_years = []
    for j, t in enumerate(task_names):
        n_in_bin = bin_size[j]
        subject_data = pd.read_csv(os.path.join(data_dir, subject + t + '.csv'))
        # Get year of each trial for plotting later 
        
            
        # Bin data and get nested list containing the outcome values (0 or 1) of all trials in that bin
        binned_trials = [subject_data['chose_correct'][i:i+n_in_bin] for i in range(0, len(subject_data), n_in_bin)]
        ###
        # counting sewssions in bins
        sessions_ints = [int(x) for x in subject_data['days_from_20160101']]
        sessions_binned = [sessions_ints[i:i+n_in_bin] for i in range(0, len(subject_data), n_in_bin)]
        id_bin = []
        bin_how_many = []
        for bb, b in enumerate(sessions_binned):
            which_sessions = np.unique(b)
            bin_how_many.append(which_sessions.shape[0])
            id_bin.append(bb)
        
        to_bev = pd.DataFrame({'bin num':id_bin, 'num sessions':bin_how_many})
       # to_bev.to_csv('/home/ssbeast/Projects/HFTemp/ColorShapeContingency1/analysis/'+subject+'_sessions_bins.csv', index=False)
        ###
        # For each bin, bootstrap the accuracy 1000 times
        n_boots = 1000
        n_bins = len(binned_trials)
        shape_color = np.zeros((3, n_bins)) # <metric, bin number>
        
        for l, b in enumerate(binned_trials):
            boot_accs = []
            for i in range(n_boots):
                sample = np.random.choice(b, size=n_in_bin) # array of 0s and 1s, resample to bin size
                boot_accs.append(sample.mean()) # calculate accuracy for that sample of trials
            boot_mean_acc = np.array(boot_accs).mean() # accuracy at trial l
            boot_lcb = np.quantile(boot_accs, q=.025) # lower confidence bound of acccuracy at trial l
            boot_ucb = np.quantile(boot_accs, q=.975) # upper confidence bound of accuracy at trial l
            shape_color[0,l] = boot_mean_acc
            shape_color[1,l] = boot_lcb
            shape_color[2,l] = boot_ucb
            
         # To preserve trial number as x axis, get trial number each bin would be centered on
        x_vals = len(subject_data)
        use_x = list(range(int(n_in_bin/2),x_vals+int(n_in_bin/2), n_in_bin))
        use_x = use_x[:n_bins]
        
        try:
            subject_data['year'] = [datetime.fromtimestamp(x/1000).strftime("%Y") for x in subject_data['timestamp']]
        except:
            subject_data['year'] = '2016' # timestamp not currently in Train 2AFC id trial csvs, but all trials were in 2016
        # For each bin, approximate which year most trials in that bin were completed in, for plotting later
        
        if t == '_Train_2AFC':
            year_bins, include_years = [0], ['2016']
        else:
            bin_year = [subject_data['year'][i:i+n_in_bin] for i in range(0, len(subject_data), n_in_bin)] 
            year_labels = [x.mode()[0] for x in bin_year] 
            if len(year_labels) > n_bins: # if more years than bins, drop last, years are aligned to the first bin
                year_labels = year_labels[:n_bins] 
            # Only want to plot a year mark at the start of each year
            year_changes = [i for i in range(len(year_labels)) if year_labels[i] != year_labels[i-1]] # which bins are year transitions
            include_years = [year_labels[x] for x in year_changes] # keep those years
            year_bins = [use_x[i] for i in year_changes] # get corresponding trial number (bin) values
        if j == 1:
            use_x = [x + subj_x[0][-1] for x in use_x]
            year_bins = [x + subj_x[0][-1] for x in year_bins]
            if include_years[0] == '2016':
                include_years = include_years[1:]
                year_bins = year_bins[1:]
        subj_x.append(use_x)
        subj_y.append(shape_color[0])
        subj_y_ci.append(np.array([shape_color[1],shape_color[2]]).T)
        subj_year_ticks.extend(year_bins)
        subj_years.extend(include_years)
    x.append(subj_x)
    y.append(subj_y)
    y_ci.append(subj_y_ci)
    year_ticks.append([subj_year_ticks, subj_years])

bin_data = [x,y]
ci_data = [x, y_ci]

colors = ["black", "tab:gray"]#["#D95319", "tab:gray"]
open_c = False

fig, axs = plt.subplots(figsize = plot_size)
for i in range(len(bin_data)):
    for j in range(len(bin_data[0][i])):
        if open_c:
            axs.scatter(bin_data[0][i][j], bin_data[1][i][j], facecolor='none', edgecolor=colors[i], s=6, linewidth=.2, rasterized=False)
        else:
            axs.scatter(bin_data[0][i][j], bin_data[1][i][j], facecolor=colors[i], edgecolor=colors[i], s=6, linewidth=.5, rasterized=False)
        if ci_data is not None:
            axs.fill_between(ci_data[0][i][j], np.array(ci_data[1][i][j]).T[0], np.array(ci_data[1][i][j]).T[1], alpha=.2, color=colors[i], rasterized=False) # rasterize CIs else get svg rendering issues
        yr_ax = axs.secondary_xaxis(location=0)
        yr_ax.set_xticks(year_ticks[i][0],year_ticks[i][1],fontsize=7, color=colors[i], rotation=90)
if np.max(bin_data[0][0][-1]) > 20000:
    xtick = list(range(0, int(np.max(bin_data[0][0][-1])), 20000))
elif np.max(bin_data[0][0][-1]) < 15000:
    xtick = [0, 10000]
else:
    xtick = list(range(0, int(np.max(bin_data[0][0][-1])), 1000))
axs.margins(.05)

if len(nafc)==2:
    stop2 = np.max([np.max(bin_data[0][0][0]), np.max(bin_data[0][1][0])])
    start4 = np.min([np.min(bin_data[0][0][1]), np.min(bin_data[0][1][1])])
    stop4 = np.max([np.max(bin_data[0][0][1]), np.max(bin_data[0][1][1])])

axs.tick_params(axis="both", length=2., pad=1)
axs.tick_params(axis='x', pad=6)

ytick = [.25,.5,1.]
axs.set_yticks(ytick, labels=[str(yt) for yt in ytick], fontsize=7)
axs.set_xticks(xtick)
axs.set_xticklabels([int(xt/10000) for xt in xtick],fontsize=7)

fig.tight_layout()
#fig.savefig(os.path.join(out_dir, name + '_learning.svg'))
plt.show()
plt.close()
    
