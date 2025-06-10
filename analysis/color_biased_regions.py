"""
Script to compare % sig. change difference between color-associated shapes and
non-color-associated shapes (passive task 1) in color-biased vs. non-biased 
regions in V4 and IT parcels
"""
import os
import pandas as pd
import nibabel as nib
import numpy as np
from bin import passive_task_functions as pf

# Choose subject
subject = 'jeeves' # one of 'wooster', 'jeeves'
content_root = 'data' # where are the data stored
subj_root = os.path.join(content_root, 'subjects', subject) # where is that subject's data

# Set out directory
outdir = 'results/passive' 

# LOAD DATA KEYS pointing to nifti beta weight images for each condition on each run
# for passive task 1 and for eccentricity 
    
# Passive task 1: Color-association
scp_mod_dir = os.path.join(subj_root, 'analysis', 'scp')
scp_beta_coeffs_key = pd.read_csv(os.path.join(subj_root, 'analysis', 'shape_color_passive_block_beta_coeffs_key.csv'))

# Dynamic Localizer
dyloc_mod_dir = os.path.join(subj_root, 'analysis', 'dyloc')

# Eccentricity
ecc_mod_dir = os.path.join(subj_root, 'analysis', 'ecc')
ecc_beta_coeffs_key = pd.read_csv(os.path.join(subj_root, 'analysis', 'eccentricity_mapper_beta_coeffs_key.csv'))


# GENERATE ROI DEFINITIONS
# Load subject's atlas parcels
atlas_path = os.path.join(subj_root, 'rois', 'major_divisions', 'final_atlas.nii.gz')

# Color-biased and non-color-biased regions
# Get mask of voxels that respond above baseline to at least one of the conditions
visu_responsive_mask_dyloc = pf.get_visually_responsive(dyloc_mod_dir, conditions=['bw_faceszscore_map','bw_bodieszscore_map', 'bw_sceneszscore_map','bw_objectszscore_map', 'bw_scrambledzscore_map',
                                                                  'c_faceszscore_map','c_bodieszscore_map', 'c_sceneszscore_map','c_objectszscore_map', 'c_scrambledzscore_map'],p=.01)

# Get ROI parcels from atlas, apply visually responsive mask
# returns list of roi names and masks as tuples corresponding to x,y,z voxel position
subdiv_rois_dyloc = pf.get_roi_defs(rois_dir=None, 
                                           atlas_path=atlas_path, 
                                           visu_responsive_mask=visu_responsive_mask_dyloc, 
                                           rois_from_nifti=False)

# Drop ROIs we aren't quantifying
subdiv_rois_dyloc = [r for r in subdiv_rois_dyloc if r[0] in ['V4', 'pIT', 'cIT', 'aIT']] 

# Apply eccentricity mask V4
conds = ['dva1', 'dva3', 'dva7', 'dva14']
ecc_maps = [nib.load(os.path.join(ecc_mod_dir, c_name + '.nii.gz')).get_fdata() for c_name in conds]
for ri, rs in enumerate(subdiv_rois_dyloc):
    if rs[0]=='V4':
        print(rs[0])
        dva1=ecc_maps[0][rs[1]]
        dva3=ecc_maps[1][rs[1]]
        dva7=ecc_maps[2][rs[1]]
        dva14=ecc_maps[3][rs[1]]
        # Voxel must respond significantly to at least one of the stimuli
        first_threshold = pf.get_visually_responsive(ecc_mod_dir, conds, p=.01)[rs[1]]
        # Voxel must respond most strongly to the 1 or 3 DVA condition
        z = ((dva1>dva7) & (dva1>dva14)& (dva1>dva3)) | ((dva3>dva7) & (dva3>dva14)& (dva3>dva1))
        full = (first_threshold & z) # Mask for voxels meeting both conditions
        masked = tuple([[l[i] for i in range(len(l)) if full[i]] for l in rs[1]])           
        subdiv_rois_dyloc[ri][1] = masked # reassign the masked roi into the list

# Split into color-biased and non-color-biased ('remaining')
subdiv_rois_color = []
subdiv_rois_remaining = []
for ri, rs in enumerate(subdiv_rois_dyloc):
    print(rs[0])
    print(len(rs[1][0]))
    color = pf.fdr_correct(rs[1], mod_dir=dyloc_mod_dir, conditions=['colored_minus_bw'], q=.01) # Voxels sig. color-biased with FDR correction
    remaining = np.logical_not(color) # all visually responsive voxels not meeting color-biased threshold
    subdiv_rois_color.append([rs[0], tuple([[l[i] for i in range(len(l)) if color[i]] for l in rs[1]])])
    subdiv_rois_remaining.append([rs[0], tuple([[l[i] for i in range(len(l)) if remaining[i]] for l in rs[1]])])
    print(rs[0])
    print(np.nonzero(color)[0].shape)

# Generate color-biased ROIs with higher thresholds
reduce_top_voxel_prop = [.1, .25, .5, .75, 1.0] # what proportions of top vox. to get
subdiv_rois_color_props = []
for p_top_voxel in reduce_top_voxel_prop:
    for roi in subdiv_rois_color:
        new_def = pf.get_top_vox(mod_dir=dyloc_mod_dir, 
                                      condition= ['colored_minus_bw'], 
                                      prop=p_top_voxel, 
                                      roi_def=roi[1])
        new_roi = [roi[0]+'_'+str(p_top_voxel), new_def]
        subdiv_rois_color_props.append(new_roi)

# COMPUTE % SIGNAL CHANGE
# Passive task 1
# 'uncolored_shape' is old naming for colorless color-assoc. shape, and 
# 'achromatic_shape' is old for colorless non-color-assoc. shape
# Get beta weights (3d brain volume) for each run and condition into a dataframe
# The 'constant' regressor gives us an estimate of the baseline activity
scp_betas = pf.load_betas(scp_beta_coeffs_key, 
                                 conditions_to_quant=['uncolored_shape', 'achromatic_shape', 'constant'], 
                                 content_root=content_root, subj_root=subj_root)

# Path to subject's masked funcitonal target; used for getting brain mask
ft_path = os.path.join(subj_root, 'mri', 'functional_target.nii.gz') 

# Compute in color-biased voxels
# This function computes the % signal change on each run for each voxel
color_assoc_psc_color_biased = pf.percent_signal_change(scp_betas,['uncolored_shape', 'achromatic_shape'], subdiv_rois_color_props, ft_path)
# To dataframe
color_assoc_psc_color_biased = pd.DataFrame(color_assoc_psc_color_biased, columns = ['run', 'condition', 'roi_full', 'voxel_effect', 'roi_effect', 'n_color_voxels'])
# Pivot to make calculating difference easy; only need voxel_effect
color_assoc_psc_color_biased_p = color_assoc_psc_color_biased.pivot(index=['run', 'roi_full', 'n_color_voxels'], columns='condition', values='voxel_effect').reset_index()
# Calculate psc difference between conditions at each voxel
color_assoc_psc_color_biased_p['diff_in_colorbiased'] = color_assoc_psc_color_biased_p['uncolored_shape'] - color_assoc_psc_color_biased_p['achromatic_shape']
# Average voxels in an ROI
color_assoc_psc_color_biased_p['mean_diff_in_colorbiased'] = [np.mean(arr) for arr in color_assoc_psc_color_biased_p['diff_in_colorbiased']]
# Split roi name into roi and proportion of top voxels
color_assoc_psc_color_biased_p['roi'] = [str(x).split('_')[0] for x in color_assoc_psc_color_biased_p['roi_full']]
color_assoc_psc_color_biased_p['prop_top_vox'] = [str(x).split('_')[1] for x in color_assoc_psc_color_biased_p['roi_full']]
# Keep only needed columns
color_biased = color_assoc_psc_color_biased_p[['run', 'roi', 'prop_top_vox', 'n_color_voxels', 'mean_diff_in_colorbiased']]

# Compute in non-color-biased voxels 
color_assoc_psc_noncolor_biased = pf.percent_signal_change(scp_betas,['uncolored_shape', 'achromatic_shape'], subdiv_rois_remaining, ft_path)
# To dataframe
color_assoc_psc_noncolor_biased = pd.DataFrame(color_assoc_psc_noncolor_biased, columns = ['run', 'condition', 'roi', 'voxel_effect', 'roi_effect', 'n_noncolor_voxels'])
# Pivot and compute as above
color_assoc_psc_noncolor_biased_p = color_assoc_psc_noncolor_biased.pivot(index=['run', 'roi', 'n_noncolor_voxels'], columns='condition', values='voxel_effect').reset_index()
color_assoc_psc_noncolor_biased_p['diff_in_noncolorbiased'] = color_assoc_psc_noncolor_biased_p['uncolored_shape'] - color_assoc_psc_noncolor_biased_p['achromatic_shape']
color_assoc_psc_noncolor_biased_p['mean_diff_in_noncolorbiased'] = [np.mean(arr) for arr in color_assoc_psc_noncolor_biased_p['diff_in_noncolorbiased']]
noncolor_biased = color_assoc_psc_noncolor_biased_p[['run', 'roi', 'n_noncolor_voxels', 'mean_diff_in_noncolorbiased']]

# Merge in order to columnwise subtractions
# how=left will repeat noncolorbiased rows for each top prop colorbiased voxels
color_minus_noncolor = pd.merge(color_biased, noncolor_biased, how='left', on=['run', 'roi']) 
# Compute difference in color-associated shape bias between colorbiased and 
# noncolorbiased voxels within each larger ROI parcel
color_minus_noncolor['ca_bias_color_minus_noncolor'] = color_minus_noncolor['mean_diff_in_colorbiased'] - color_minus_noncolor['mean_diff_in_noncolorbiased']

# Save file out
color_minus_noncolor_out = os.path.join(outdir, subject+'_color_assoc_bias_colorbiased_minus_noncolorbiased.csv')
color_minus_noncolor.to_csv(color_minus_noncolor_out, index=False)