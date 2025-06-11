"""
Script to compute selectivity values in ROI parcels for both passive task 1 and 2
This normalizes the difference in response to two conditions by their additive response

Uses .nii images storing the beta weights at each voxel for each run as computed by GLM
Run for one subject at a time, specified below

The prefix, 'scp' refers to passive task 1 (color-associated minus non-assoc. shapes)
The prefix 'congruency' refers to passive task 2 (incongruent minus congruent objects)
"""
import os
import pandas as pd
import nibabel as nib
from bin import passive_task_functions as pf

# Choose subject
subject = 'wooster' # one of 'wooster', 'jeeves'
content_root = 'data' # where are the data stored
subj_root = os.path.join(content_root, 'subjects', subject) # where is that subject's data

# Set out directory
outdir = 'results/passive' 

# LOAD DATA KEYS pointing to nifti beta weight images for each condition on each run
# for each passive task and eccentricity (for masking peripheral ret. cortx.)
    
# Passive task 1: Color-association
scp_mod_dir = os.path.join(subj_root, 'analysis', 'scp')
scp_beta_coeffs_key = pd.read_csv(os.path.join(subj_root, 'analysis', 'shape_color_passive_block_beta_coeffs_key.csv'))

# Passive task 2: Congruency
congruency_mod_dir = os.path.join(subj_root, 'analysis', 'congruency')
congruency_beta_coeffs_key = pd.read_csv(os.path.join(subj_root, 'analysis', 'shape_color_congruency_block_beta_coeffs_key.csv'))

# Eccentricity
ecc_mod_dir = os.path.join(subj_root, 'analysis', 'ecc')
ecc_beta_coeffs_key = pd.read_csv(os.path.join(subj_root, 'analysis', 'eccentricity_mapper_beta_coeffs_key.csv'))



# GENERATE ROI DEFINITIONS
# Load subject's atlas parcels
atlas_path = os.path.join(subj_root, 'rois', 'major_divisions', 'final_atlas.nii.gz')

# ROIs for passive task 1:
# Get mask of voxels that respond above baseline to at least one of the conditions
visu_responsive_mask_scp = pf.get_visually_responsive(scp_mod_dir, 
                                                                conditions=['scp_uncolored_shape','scp_achromatic_shape', 'scp_color', 'scp_colored_shape'], 
                                                                p=.01)
# Get ROI parcels from atlas, apply visually responsive mask
# returns list of roi names and masks as tuples corresponding to x,y,z voxel position
subdiv_rois_scp = pf.get_roi_defs(rois_dir=None, 
                                           atlas_path=atlas_path, 
                                           visu_responsive_mask=visu_responsive_mask_scp,
                                           rois_from_nifti=False)

# ROIs for passive task 2:
# Repeat process for passive task 1
visu_responsive_mask_congruency = pf.get_visually_responsive(congruency_mod_dir, 
                                                                conditions=['incongruent','congruent'], 
                                                                p=.01)

subdiv_rois_congruency = pf.get_roi_defs(rois_dir=None, 
                                           atlas_path=atlas_path, 
                                           visu_responsive_mask=visu_responsive_mask_congruency, 
                                           rois_from_nifti=False)

# Apply eccentricity masks to retinotopic areas V1, V2, V3, V4
conds = ['dva1', 'dva3', 'dva7', 'dva14']
ecc_maps = [nib.load(os.path.join(ecc_mod_dir, c_name + '.nii.gz')).get_fdata() for c_name in conds]
for h, roi_set in enumerate([subdiv_rois_scp, subdiv_rois_congruency]):
    for ri, rs in enumerate(roi_set):
        if rs[0] in ['V1', 'V2', 'V3', 'V4']:
            print('roi ', rs[0])
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
            roi_set[ri][1] = masked # reassign the masked roi into the list

# Combine vFC, dFC, and oFC
subdiv_rois_scp.append(['FC', pf.combine_rois(subdiv_rois_scp, ['vFC', 'dFC', 'oFC'])])
subdiv_rois_congruency.append(['FC', pf.combine_rois(subdiv_rois_congruency, ['vFC', 'dFC', 'oFC'])])

# Drop ROIs we aren't quantifying
subdiv_rois_scp = [r for r in subdiv_rois_scp if r[0] in ['V1', 'V2', 'V3', 'V4', 'pIT', 'cIT', 'aIT', 'TP', 'FC']]
subdiv_rois_congruency = [r for r in subdiv_rois_congruency if r[0] in ['V1', 'V2', 'V3', 'V4', 'pIT', 'cIT', 'aIT', 'TP', 'FC']]

# =============================================================================
# # If you want to save masks out as binary niftis to look at
# for (roi_set, roi_set_name) in zip([subdiv_rois_scp, subdiv_rois_congruency],['_subdiv_scp', '_subdiv_cg']):
#     for roi in roi_set:
#         names = pf.roi_def_to_nii(subj_root=subj_root, 
#                            roi_coords=roi[1], 
#                            roi_name=roi[0]+roi_set_name, 
#                            roi_dir='rois')
#         print(names)
# =============================================================================

# COMPUTE SELECTIVITY IN ROI 
# Passive task 1
# 'uncolored_shape' is old naming for colorless color-assoc. shape, and 
# 'achromatic_shape' is old for colorless non-color-assoc. shape
# Get beta weights (3d brain volume) for each run and condition into a dataframe
scp_betas = pf.load_betas(scp_beta_coeffs_key, 
                                 conditions_to_quant=['uncolored_shape', 'achromatic_shape'], 
                                 content_root=content_root, subj_root=subj_root)
# This function computes the selectivity on each run for each voxel, then averages voxels in an ROI
color_assoc_selectivity = pf.selectivity(scp_betas,'uncolored_shape', 'achromatic_shape', subdiv_rois_scp)
# To dataframe
color_assoc_selectivity = pd.DataFrame(color_assoc_selectivity, columns = ['run', 'roi', 'effect'])
# Add identifier
color_assoc_selectivity['comparison'] = 'uncolored_shape_vs_achromatic_shape'

# Passive task 2 (same as above)
congruency_betas = pf.load_betas(congruency_beta_coeffs_key, 
                                 conditions_to_quant=['incongruent', 'congruent'], 
                                 content_root=content_root, subj_root=subj_root)
incongruency_selectivity = pf.selectivity(congruency_betas,'incongruent', 'congruent', subdiv_rois_congruency)
incongruency_selectivity = pd.DataFrame(incongruency_selectivity, columns = ['run', 'roi', 'effect'])
incongruency_selectivity['comparison'] = 'incongruent_vs_congruent'

# Save files out
# Passive task 1
color_assoc_selectivity_out = os.path.join(outdir, subject+'_color_assoc_selectivity.csv')
color_assoc_selectivity.to_csv(color_assoc_selectivity_out, index=False) 
# Passive task 2
incongruency_selectivity_out = os.path.join(outdir, subject+'_incongruency_selectivity.csv')
incongruency_selectivity.to_csv(incongruency_selectivity_out, index=False) 