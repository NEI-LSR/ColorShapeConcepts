"""
Plots the color associated shape bias in color-biased compared to non-color-biased
regions, at different proportions of top color-biased voxels 
Figure 4f
"""
import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
from bin import plotter

# Set paths
data_dir = 'results/passive' # where the csvs containing data to plot are
out_dir = 'figures/fig4'

combined = True
individual = False

def bootstrap_diff(w_df, je_df, plot_rois, groups, boots=1000):
    """
    Given dataframe output from 'color_biased_regions.py'
    bootstraps the color-assoc. shape bias diff in color-biased and non-
    color-biased regions to get a bootstrapped estimate
    of the difference and 95 % confidence intervals
    """
    # Get sample sizes (number of runs)
    n_w = len(w_df['run'].unique())
    n_je = len(je_df['run'].unique())
    n = n_w+n_je
    # Get weights for boostrapping concatenated subj data
    w_w = 1/(2*n_w) 
    w_je = 1/(2*n_je)
    # Choose resamples ahead of time so that for each proportion of color biased voxels, the same resamples are used
    w_rs = [np.random.choice(np.arange(n_w), n_w) for i in range(boots)]
    je_rs = [np.random.choice(np.arange(n_je), n_je) for i in range(boots)]
    combined_weights = np.concat([np.full((n_w), w_w), np.full((n_je),w_je)])
    combined_rs = [np.random.choice(np.arange(n), n, p=combined_weights) for i in range(boots)]
    
    # Initialize arrays
    w_arr = np.zeros((len(plot_rois),len(groups),boots)) # rois, top voxel props, boot samples
    je_arr = np.zeros((len(plot_rois),len(groups),boots))
    combined_arr = np.zeros((len(plot_rois),len(groups),boots))
    for i, roi in enumerate(plot_rois):
        for j, gr in enumerate(groups):
            # Get effect for all runs for each roi
            w_vals = w_df[(w_df['roi']==roi)&(w_df['prop_top_vox']==gr)]['ca_bias_color_minus_noncolor'].values
            je_vals = je_df[(je_df['roi']==roi)&(je_df['prop_top_vox']==gr)]['ca_bias_color_minus_noncolor'].values
            # Concat
            combined_vals = np.concat([w_vals, je_vals])
            # Bootstrap samples
            for b in range(boots):
                w_sample = w_vals[w_rs[b]]
                je_sample = je_vals[je_rs[b]]
                combined_sample = combined_vals[combined_rs[b]]
                # Assign back to arrays
                w_arr[i][j][b] = np.nanmean(w_sample)
                je_arr[i][j][b] = np.nanmean(je_sample)
                combined_arr[i][j][b] = np.nanmean(combined_sample)
    return combined_arr, w_arr, je_arr

# Plot olor-associated shape bias in color-biased vs. non-color-biased voxels         
plot_rois = ['V4', 'pIT', 'cIT', 'aIT']
top_voxel_props = [0.1, 0.25, 0.5, 0.75, 1.]
# Load results into dataframes
w_color_biased = pd.read_csv(os.path.join(data_dir, 'wooster_color_assoc_bias_colorbiased_minus_noncolorbiased.csv'))
je_color_biased = pd.read_csv(os.path.join(data_dir, 'jeeves_color_assoc_bias_colorbiased_minus_noncolorbiased.csv'))

c, w, j = bootstrap_diff(w_color_biased, je_color_biased, plot_rois, top_voxel_props, boots=1000)

# Plot and save out means & 95 % CIs
plot_size = (0.9, 0.7)
plot_arrs = []
plot_names = []
if combined:
    plot_arrs.append(c)
    plot_names.append('combined')
if individual:
    plot_arrs.append(w)
    plot_names.append('w')
    plot_arrs.append(j)
    plot_names.append('je')

vals = []
for i, a in enumerate(plot_arrs):
    for j, roi_data in enumerate(a):
        roi_data_ = np.expand_dims(roi_data, axis=0) # function is expecting 3rd dim, e.g., if we were plotting multiple rois on the same plot 
        title = plot_names[i]+'_'+plot_rois[j]+'_color_assoc_bias_cb_noncb'
        fig, axs = plt.subplots(figsize=plot_size)
        fig = plotter.create_save_prop_line_plot(axs, fig, title, boot_data=roi_data_, 
                                                 x=top_voxel_props,out_dir=out_dir,
                                                 set_size = plot_size, 
                                                 rotate_x_labels=True)
        fig.show()
        for k, g in enumerate(roi_data):
            effect = g.mean(axis=-1)
            lower = np.quantile(g, .025)
            upper = np.quantile(g, .975)
            vals.append([plot_names[i], plot_rois[j], top_voxel_props[k], effect, lower, upper]) 
v_df = pd.DataFrame(vals, columns=['subject', 'roi', 'prop_top_voxel', 'effect', 'lower_cb', 'upper_cb'])
v_df.to_csv(os.path.join(out_dir, 'color_assoc_bias_cb_noncb.csv'), index=False)

