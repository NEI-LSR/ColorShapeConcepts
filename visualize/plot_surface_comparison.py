"""
Plots comparison of color-associated shapes minus non-color-associated shapes
and incongruent minus congruent objects biases calculated on surface.
"""
import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from bin import plotter

# Set paths
data_dir = 'results/passive' # where the csvs containing data to plot are
out_dir = 'plots/fig4'

combined = True # plot/save combined data,  
individual = False # plot/save subject data

# Load results data
binned_surf_data = pd.read_csv(os.path.join(data_dir, 'surface_binned_data.csv'))
roi_divisions = pd.read_csv(os.path.join(data_dir, 'surface_binned_data_rois.csv'))

contrasts = ['ca_nca', 'ic_c'] # color-assoc. minus non-assoc. and incongruent minus congruent
boots = 1000 # how many bootstrap iterations
n_bins = np.max(binned_surf_data['bin'])+1 # how many bins (+1 bc index starts at 0)

# Initialize arrays to store bootstrapped z scores for each contrast and bin
je_samples_arr = np.zeros((len(contrasts),n_bins,boots))
w_samples_arr = np.zeros((len(contrasts),n_bins,boots))
combined_samples_arr = np.zeros((len(contrasts),n_bins,boots))
for b in range(n_bins-1):
    w_bin_df = binned_surf_data[(binned_surf_data['subj']=='w')&(binned_surf_data['bin']==b)]
    je_bin_df = binned_surf_data[(binned_surf_data['subj']=='je')&(binned_surf_data['bin']==b)]
    for i, c in enumerate(contrasts):
        # Get effect size and variance
        w_eff = w_bin_df[w_bin_df['contrast']==c]['effect_size'].values
        je_eff = je_bin_df[je_bin_df['contrast']==c]['effect_size'].values
        w_var = w_bin_df[w_bin_df['contrast']==c]['variance'].values
        je_var = je_bin_df[je_bin_df['contrast']==c]['variance'].values
        
        # Combine
        combined_eff = np.concat([w_eff, je_eff])
        combined_var = np.concat([w_var, je_var])
        
        # Get sample sizes (n runs) 
        n_w = w_eff.shape[0]
        n_je = je_eff.shape[0]
        n = n_w+n_je
        # Get weights for the combined bootstrap
        w_w = 1/(2*n_w) 
        w_je = 1/(2*n_je)
        w_weights = [w_w for x in w_eff]
        je_weights = [w_je for x in je_eff] 
        combined_weights = np.concat([w_weights, je_weights])
        
        # Bootstrap z-score diff in each bin
        for boot in range(boots):
            # Sample indices (need to grab effect sizes and variances corresponding to same run)
            w_sample = np.random.choice(list(range(n_w)), n_w)
            je_sample = np.random.choice(list(range(n_je)), n_je)
            combined_sample = np.random.choice(list(range(n)), n, p=combined_weights)  
            # Get sample z score, mean effect size divided by standard error
            w_sample_z = np.mean(w_eff[w_sample])/sqrt(np.mean(w_var[w_sample])/n_w)
            je_sample_z = np.mean(je_eff[je_sample])/sqrt(np.mean(je_var[je_sample])/n_je)
            combined_sample_z = np.mean(combined_eff[combined_sample])/sqrt(np.mean(combined_var[combined_sample])/n)
            # Assign back to array
            je_samples_arr[i][b][boot] = je_sample_z
            w_samples_arr[i][b][boot] = w_sample_z
            combined_samples_arr[i][b][boot] = combined_sample_z
            
# Get average location of each roi boundary 
w_roi_divs = np.mean(roi_divisions[['w_lh', 'w_rh']].values, axis=1)
je_roi_divs = np.mean(roi_divisions[['je_lh', 'je_rh']].values, axis=1)
combined_roi_divisions = np.mean(roi_divisions[['w_lh', 'w_rh', 'je_lh', 'je_rh']].values, axis=1)

# Plot mean z score for each bin with bootstrapped 95% CI
plot_size = (1.2, .85) 
plot_arrs = []
plot_names = []
plot_rois = []
if combined:
    plot_arrs.append(combined_samples_arr)
    plot_names.append('combined')
    plot_rois.append(combined_roi_divisions)
if individual:
    plot_arrs.append(w_samples_arr)
    plot_names.append('w')
    plot_rois.append(w_roi_divs)
    plot_arrs.append(je_samples_arr)
    plot_names.append('je')
    plot_rois.append(je_roi_divs)
    
# Plot and save out
for i, a in enumerate(plot_arrs):
    title = plot_names[i] + '_surface_comparison'
    divs = plot_rois[i]
    fig, axs = plt.subplots(figsize=plot_size) 
    fig = plotter.create_save_surf_comparison(axs, fig, title, a, ['ca', 'ic'],
                              divs, x=None, ylim=None, xposition=0., yposition=0., 
                              out_dir=out_dir, set_size=plot_size)
    
    fig.show()