"""
Helper functions for the passive task analyses
"""
import numpy as np
import pandas as pd
import nibabel as nib
import os
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests as mt

def get_visually_responsive(mod_dir, conditions, p=.01):
    """
    Parameters:
        mod_dir : str; directory containing glm and contrasts used to define vis resp
        conditions : list of str; name of conditions to use in thresholding
        p : p-value for thresholding; subject to Bonferroni MC
    Returns:
        binary mask over the whole brain; 1 indicates a visually responsive voxel
    """
    visu_responsive = []
    for c in conditions:
        c_data = nib.load(os.path.join(mod_dir, str(c)+'.nii.gz')).get_fdata()
        visu_responsive.append(c_data)
    p_bonf = p/len(visu_responsive) # correct for multiple comparisons, comparing value to each of the conditions
    z = stats.norm.ppf(1 - p_bonf) # 1 tailed, voxel more responsive than baseline
    print('cutoff is ', z)
    visu_responsive_mask = np.logical_or.reduce([arr > z for arr in visu_responsive])
    return visu_responsive_mask

def get_roi_defs(rois_dir=None, 
                 atlas_path = None,
                 visu_responsive_mask=None,
                 rois_from_nifti=False):
    """
    Parameters:
        rois_dir: dir containing binary mask niftis if rois_from_nifti=True, otherwise None
        rois_from_nifti: True if rois are saved as separate binary mask niftis; False if rois are stored in atlas format
        visu_responsive_mask: any additional restrictions on voxels included in the roi 
        atlas_path: path to atlas nifti if rois_from_nifti=False, otherwise None, lookup.txt must be in same directory
    Returns:
        list of lists: roi name and tuple of x,y,z coords of voxels in roi
    """
    roi_defs = []
    if rois_from_nifti: 
        roi_mask_paths = [f for f in os.listdir(rois_dir) if ".nii.gz" in f and "~" not in f and "reg" not in f]
        
        for roi in roi_mask_paths:
            mask_img = nib.load(os.path.join(rois_dir, roi)).get_fdata()
            if len(mask_img.shape) == 4:
                mask_img = mask_img.mean(axis=-1)
                
            if visu_responsive_mask is not None:
                roi_mask = np.where((mask_img>0)&(visu_responsive_mask))
            else:
                roi_mask = np.where(mask_img>0)
            roi_defs.append([str(roi).split('.')[0], roi_mask])
    else:
        atlas = nib.load(atlas_path).get_fdata() # atlas itself
        atlas_lookup = os.path.join(os.path.dirname(atlas_path), 'lookup_match.txt') # atlas lookup table
        keys = []
        values = []
        with open(atlas_lookup) as f:
            next(f)
            next(f)
            for line in f:
                keys.append(str(line).split('\t')[1])
                values.append(str(line).split('\t')[0])
        atlas_dict = dict(zip(keys, values))
        atlas_dict.pop('tie', None)
        
        for roi in atlas_dict:
            if visu_responsive_mask is not None: # if there's a mask to use
                roi_mask = np.where((atlas==int(atlas_dict[roi]))&(visu_responsive_mask)) 
            else:
                roi_mask = np.where(atlas==int(atlas_dict[roi]))
            roi_defs.append([str(roi), roi_mask])
    return roi_defs

def combine_rois(roi_defs, roi_list):
    """
    given roi defs from get_roi_defs and a list of rois to combine
    voxels for, concatenates coords and returns in same format
    """
    a=[l for l in roi_defs if l[0] in roi_list]
    x = []
    y = []
    z = []
    for i in range(len(roi_list)):
        if isinstance(a[i][1][0], list):
            x += a[i][1][0]
            y += a[i][1][1]
            z += a[i][1][2]
        else:
            x += list(a[i][1][0])
            y += list(a[i][1][1])
            z += list(a[i][1][2])
    combined_def = tuple([x,y,z])
    return combined_def

def roi_def_to_nii(subj_root, roi_coords, roi_name, roi_dir):
    """
    Saves out each roi def as a nifti if you want to view it
    """
    subject_template = nib.load(os.path.join(subj_root, 'mri', 'functional_target.nii.gz'))
    binary_img = np.zeros_like(subject_template.get_fdata())
    roi_coordsT = np.array(roi_coords).T
    for voxel in roi_coordsT:
        binary_img[voxel[0]][voxel[1]][voxel[2]] = 1
    binary_img_complete = nib.Nifti1Image(binary_img, affine=subject_template.affine)
    out_path = os.path.join(subj_root, roi_dir, roi_name + '.nii.gz')
    nib.save(binary_img_complete, out_path)
    return 'niftis saved at ', out_path

def load_betas(beta_coeffs_key, conditions_to_quant, content_root, subj_root):
    """
    Get dataframe with runwise GLM beta weights for conditions of interest
    Parameters:
        beta_coeffs_key: pd.df, containing paths to niftis storing GLM beta weights
        conditions_to_quant: list, which regressors from the mod to grab
        content_root: str
        subj_root: str
    Returns:
        Df
    """
    # Add column combining session and ima names for easier id-ing of runs
    runs = [str(s_)+'_'+str(i_) for (s_,i_) in zip(beta_coeffs_key['session'], beta_coeffs_key['ima'])]
    beta_coeffs_key['session_ima'] = runs
    # For each run
    betas_3d = []
    for run in beta_coeffs_key['session_ima'].unique():
        # For each condition you are interested in
        for cond in conditions_to_quant:
            # Load the condition betas image
            cond_path = beta_coeffs_key[(beta_coeffs_key['session_ima']==run)&(beta_coeffs_key['condition']==cond)]['beta_path'].item()
            try:
                cond_data = nib.load(content_root + cond_path).get_fdata()
            except:
                cond_data = nib.load(cond_path).get_fdata() 
            betas_3d.append([run, cond, cond_data])
    betas_df = pd.DataFrame(betas_3d, columns = ['run', 'condition', 'betas'])
    return betas_df

def selectivity(betas_df, a, b, roi_defs):
    """
    Parameters:
        betas_df: pd.df, output of load_betas
        a, b: str, conditions to compute selectivity between
        roi_defs: list of lists, output of get_roi_defs
    Return:
        list of lists, for each run and roi, the selectivity for the entire 
        roi, where + values indicate bias for a, - for bias for b
    """
    selects = []
    for run in betas_df['run'].unique():
        # Grab betas for each condition in 3d brain space
        a_vals = betas_df[(betas_df['run']==run)&(betas_df['condition']==a)]['betas'].item()
        b_vals = betas_df[(betas_df['run']==run)&(betas_df['condition']==b)]['betas'].item()
        # Calculate selectivity within run, separately for each voxel
        a_minus_b = a_vals-b_vals # numerator
        a_plus_b = abs(a_vals)+abs(b_vals) # denominator
        voxel_selectivity = a_minus_b/a_plus_b
        
        # Mask out rois
        for roi in roi_defs:
            roi_voxel_selectivity = voxel_selectivity[roi[1]]
            # Average over voxels in the ROI (nanmean to deal with occassional inclusion of
            # voxel in ROI that is really outside the brain, which will have a nan value)
            roi_selectivity = np.nanmean(roi_voxel_selectivity) 
            selects.append([run, roi[0], roi_selectivity])
    return selects

def percent_signal_change(betas_df, conditions, roi_defs, functional_target_path):
    """
    Parameters:
        betas_df: pd.df, output of load_betas
        conditions: list, conditions to calculate % signal change for
        roi_defs: list of lists, output of get_roi_defs
        functional_target_path: path to subject's functional template, need for baseline over whole brain
    Return:
        list of lists, for each run, roi, and condition of interest, the percent signal change at each
        voxel in the roi and averaged over voxels in the roi, along with number of voxels in roi
    """
    # Get whole brain mask
    ft = nib.load(functional_target_path).get_fdata() # load masked functional target
    ft_mask = np.nonzero(ft) # get mask of nonzero values (since it is masked, only voxels on the brain are nonzero)
    psc = []
    for run in betas_df['run'].unique():
        # Get baseline activity across whole brain, using the nilearn constant regressor
        baseline_vals = betas_df[(betas_df['run']==run)&(betas_df['condition']=='constant')]['betas'].item()
        baseline_whole_brain = baseline_vals[ft_mask].mean() # avg over all voxels in brain, need one value
        # Now compare each condition of interest to baseline
        for cond in conditions:
            cond_vals = betas_df[(betas_df['run']==run)&(betas_df['condition']==cond)]['betas'].item()
            cond_psc = (cond_vals/baseline_whole_brain)*100 # % sig change = (beta/baseline)*100
            
            # Mask out rois
            for roi in roi_defs:
                voxel_psc = cond_psc[roi[1]]
                # Average over voxels in the ROI
                roi_psc = np.mean(voxel_psc)
                # Also count how many voxels are in the roi
                n_voxels = voxel_psc.shape[0]
                psc.append([run, cond, roi[0], voxel_psc, roi_psc, n_voxels])
    return psc
    
def fdr_correct(roi_mask, mod_dir, conditions, q):
    """
    Take an already existing roi mask and get all voxels meeting some condition 
    with FDR corrections with q=q
    """
    for c in conditions:
        c_data = nib.load(os.path.join(mod_dir, str(c)+'.nii.gz')).get_fdata() #load contrast
        masked_c_data = c_data[roi_mask] # mask contrast to restrict voxels 
        p_vals = []
        for z in masked_c_data:
            p_vals.append(1-stats.norm.cdf(z)) # Convert z score map to p values
        p_vals = np.array(p_vals)            
        r, p_c = mt(pvals=p_vals, alpha=q, method='fdr_bh')[:2] # get which voxels are significant at correct value
    return r
        
def get_top_vox(mod_dir, condition, prop, roi_def):
    """
    Parameters:
        mod_dir: str, path to directory containing model with contrast you will use as condition 
        condition: list of strs, contrast conditions that you want to retrieve the top voxels for
        prop: float, what proportion of the total number of voxels to reduce to
        roi_def: list of roi name and roi coords from get_roi_defs; the starting roi mask
    Returns:
        new tuple of coords for top voxels in roi
    """
    if len(roi_def) == 2:
        roi_coords = roi_def[1]
    else:
        roi_coords = roi_def
    n_top = int(len(roi_coords[0])*prop) # how many voxels in top prop
    if len(condition)==1:
        c_data = nib.load(os.path.join(mod_dir, str(condition[0])+'.nii.gz')).get_fdata()
        masked_c_data = c_data[roi_coords] # mask the thresholding map
    else:
        cond_masks = []
        for c in condition:
            c_data = nib.load(os.path.join(mod_dir, str(c)+'.nii.gz')).get_fdata()
            masked_c_data_ = c_data[roi_coords] # mask the thresholding map
            cond_masks.append(masked_c_data_)
        masked_c_data = np.max(cond_masks, axis=0)
    top_prop = np.argpartition(masked_c_data, -n_top)[-n_top:] # get indices of top voxels
    top_roi_coords = tuple([[l[i] for i in range(len(l)) if i in top_prop] for l in roi_coords]) 
    return top_roi_coords
