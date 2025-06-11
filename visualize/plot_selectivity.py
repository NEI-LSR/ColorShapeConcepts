"""
Plots bar plot of the selectivity index calculated in ROIs for each of 
passive task 1 and passive task 2. Uses output of 'compute_selectivity.py'
Fig. 4c,d
"""

import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
from bin import plotter
    
# Set paths
data_dir = 'results/passive' # where the csvs containing data to plot are
out_dir = 'plots/fig4'

combined = True
individual = False

# Define some useful functions
def bootstrap_sel(w_df, je_df, plot_rois, boots=1000):
    """
    Given selectivity dataframe output from 'compute_selectivity.py'
    bootstraps the selectivity values to get a bootstrapped estimate
    of the mean and 95 % confidence intervals
    """
    # Get sample sizes (number of runs)
    n_w = len(w_df['run'].unique())
    n_je = len(je_df['run'].unique())
    n = n_w+n_je
    # Get weights for boostrapping concatenated subj data
    w_w = 1/(2*n_w) 
    w_je = 1/(2*n_je)
    # Initialize arrays
    w_arr = np.zeros((len(plot_rois),1,boots)) # rois, groups (1), boot samples
    je_arr = np.zeros((len(plot_rois),1,boots))
    combined_arr = np.zeros((len(plot_rois),1,boots))
    for i, roi in enumerate(plot_rois):
        # Get effect for all runs for each roi
        w_vals = w_df[w_df['roi']==roi]['effect'].values
        je_vals = je_df[je_df['roi']==roi]['effect'].values
        # Concat
        combined_vals = np.concat([w_vals, je_vals])
        # Create list of weights to match number of runs
        w_weights = [w_w for x in w_vals]
        je_weights = [w_je for x in je_vals]
        # Concat weights to mirror concatenated effects
        combined_weights = np.concat([w_weights, je_weights])
        # Bootstrap samples
        for b in range(boots):
            w_sample = np.random.choice(w_vals, n_w)
            je_sample = np.random.choice(je_vals, n_je)
            # Weight probability of drawing subj sample when getting combined sample
            combined_sample = np.random.choice(combined_vals, n, p=combined_weights)
            # Assign back to arrays
            w_arr[i][0][b] = np.nanmean(w_sample)
            je_arr[i][0][b] = np.nanmean(je_sample)
            combined_arr[i][0][b] = np.nanmean(combined_sample)
    return combined_arr, w_arr, je_arr

def plot_bars_save_vals(arrs, names, para_name, group_labels, plot_size, out_dir):
    """
    Just a wrapper for the bar plotter and will also compile
    selectivity and confidence interval values in order to save out as csv
    """
    vals = []
    for i, a in enumerate(arrs):
        title = names[i]+'_'+para_name
        fig, axs = plt.subplots(figsize=plot_size)
        fig = plotter.create_save_barplot(axs, fig, title, a, 
                                          group_labels, xposition=0., 
                                    out_dir=out_dir, 
                                    data_spread="ci",
                                    set_size=plot_size,
                                    rotate_x_labels=True)
        fig.show()
        for j, g in enumerate(a):
            effect = g.mean(axis=-1)[0]
            lower = np.quantile(g, .025)
            upper = np.quantile(g, .975)
            vals.append([names[i], group_labels[j], effect, lower, upper]) 
    return vals

# Plot selectivity bar plots 
plot_rois = ['V1', 'V2', 'V3', 'V4', 'pIT', 'cIT', 'aIT', 'TP', 'FC']
# Plot both passive tasks in turn
paradigms = ['color_assoc_selectivity', 'incongruency_selectivity']
for para in paradigms:
    # Load selectivity results into dataframes
    w_sel_df = pd.read_csv(os.path.join(data_dir, 'wooster_' + para + '.csv'))
    je_sel_df = pd.read_csv(os.path.join(data_dir, 'jeeves_' + para + '.csv'))
    
    # Bootstrap selectivity
    combined_sel, w_sel, je_sel = bootstrap_sel(w_df=w_sel_df, 
                                                            je_df=je_sel_df, 
                                                            plot_rois=plot_rois, 
                                                            boots=1000)
    # Plot and save out means & 95 % CIs
    plot_size = (0.9, 0.7)
    plot_arrs = []
    plot_names = []
    if combined:
        plot_arrs.append(combined_sel)
        plot_names.append('combined')
    if individual:
        plot_arrs.append(w_sel)
        plot_names.append('w')
        plot_arrs.append(je_sel)
        plot_names.append('je')
        
    v = plot_bars_save_vals(arrs=plot_arrs, names=plot_names, para_name = para, group_labels=plot_rois, 
                            plot_size = plot_size, out_dir=out_dir)
    v_df = pd.DataFrame(v, columns=['subject', 'roi', 'effect', 'lower_cb', 'upper_cb'])
    v_df.to_csv(os.path.join(out_dir, para + '_values.csv'), index=False)


