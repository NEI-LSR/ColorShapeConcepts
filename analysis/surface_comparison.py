"""
Script to compute color-associated shape > non-color-associated shape bias and
incongruent > congruent object bias on the inflated surface from V1 to the
temporal pole. 
Uses contrast effect sizes and variances for each run, projected to the surface,
to compute a z-scored contrast within bins along the surface
Uses package gdist to generate bins based on equal geodesic (along the surface) distancing.
General surface info:
    We work with freesurfer inflated surfaces, labels, and overlay files
    The surface consists of vertices. Each vertex has an id number. The inflated
    surface also has vertex positions (x,y,z), which each correspond to an id.
    The inflated surface lets us orient correctly in space, whereas overlay files
    give us the values we're interested in (and those are indexed by vertex id).
"""
import numpy as np
import nibabel as nib
import gdist
import nibabel.freesurfer.mghformat as mgh
import os
import pandas as pd
import matplotlib.pyplot as plt

# Set directories
content_root = 'data' # where are the data stored
outdir = 'results/passive' 

# Runs for both subjects in turn and compiles into one large csv
all_subj_data = []
roi_divisions = []
for subject in ['wooster', 'jeeves']:
    subj_root = os.path.join(content_root, 'subjects', subject) # where is that subject's data
    
    # Surface contrasts for each subj and paradgim are stored in a folder in the model directory
    # To grab them, need the session and run id info, so first get model dirs
    scp_mod_dir = os.path.join(subj_root, 'analysis', 'scp')
    congruency_mod_dir = os.path.join(subj_root, 'analysis', 'congruency')
    # Grab csvs containing the session names and run ('ima') numbers for each run in the model
    scp_mod_name = [f for f in os.listdir(scp_mod_dir) if 'ordered_runs.csv' in f]
    congruency_mod_name = [f for f in os.listdir(congruency_mod_dir) if 'ordered_runs.csv' in f]
    assert len(scp_mod_name)==1, 'more than one scp model found' # Double check there was not some duplicate
    assert len(congruency_mod_name)==1, 'more than one congruency model found'
    # Load session and run info
    scp_runs = pd.read_csv(os.path.join(scp_mod_dir, scp_mod_name[0]))
    congruency_runs = pd.read_csv(os.path.join(congruency_mod_dir, congruency_mod_name[0]))
    session_ids = scp_runs['session'].to_list() + congruency_runs['session'].to_list() # concat the session names
    run_ids = scp_runs['ima'].to_list() + congruency_runs['ima'].to_list() # concat the run numbers
    run_ids = [str(x) for x in run_ids]

    # Load labels used to mask the surface
    label_dir = os.path.join(subj_root, 'rois', 'major_divisions', 'merged_labels')
    main_label = 'VVS' # 'ventral visual stream', made from all labels for all V1 through TP
    boundary_labels = ['V1', 'V2', 'V3', 'V4', 'pIT', 'cIT', 'aIT', 'TP'] # the individual labels
    
    # Deal with subject quirks in their 3d surface positioning
    if subject =='wooster':
        ap = 2 # AnteriorPosterior axis
        si = 1 # SuperiorInferior axis
        ap_ascending = False # Vertex numbers decrease from post to ant
    else:
        ap = 1 
        si = 2 
        ap_ascending = True # Vertex numbers increase from post to ant
    
    for hemi in ['lh','rh']:
        # Load inflated cortical surface
        inf_surf_path = os.path.join(subj_root, 'surf', hemi + '.inflated')
        coords, faces = nib.freesurfer.read_geometry(inf_surf_path)
        
        # Check #1: Plot to ensure you have correct axes labeled as AP and SI
        plt.scatter(coords[:,ap], coords[:,si]) # should look like lateral surf
        plt.show()
        plt.close()
        
        # Correct dtypes because gdist is very specific
        coords = coords.astype(np.float64)
        faces = faces.astype(np.int32) 
        
        # Load main label for masking
        label_vertices = nib.freesurfer.read_label(os.path.join(label_dir, main_label + '_' + hemi + '.label'))
        label_vertices = np.array(label_vertices, dtype=np.int32)
        
        # Find most posterior vertex
        if ap_ascending: # the most posterior will be the smallest vertex id num
            start = int(label_vertices[np.argmin(coords[label_vertices, ap])])
        else: # the most posterior will be the largest vertex id num
            start = int(label_vertices[np.argmax(coords[label_vertices, ap])])
        # And compute the distance from that vertex to all other vertices
        dist_full_surf = gdist.compute_gdist(coords, faces, np.array([start], dtype=np.int32))
        # Keep only distances to vertices in mask
        masked_dist = dist_full_surf[label_vertices]

        # Bin using equal distances along the surface
        n_bins = 65
        bin_t = np.linspace(masked_dist.min(), masked_dist.max(), n_bins) # get cutoff dist for start of each bin
        bin_ids = np.digitize(masked_dist, bin_t) # assign each vertex to bin using the above
        bin_ids = bin_ids-1 # index bins 0-64 instead of 1-65
        
        # Check #2: Turn bins into a surface overlay to check that they increase
        # in the right direction. In freesurfer, color wheel, should see a color
        # gradient from posterior to anterior
        # Convert to full shape
        binned_overlay = np.full(coords.shape[0], -1, dtype=np.int32)
        # Assign value as bin number
        binned_overlay[label_vertices] = bin_ids
        # Reshape to expected .mgh which is num vertices, 1, 1
        overlay_reshaped = np.expand_dims(binned_overlay, axis=1) 
        overlay_reshaped = np.expand_dims(overlay_reshaped, axis=2)
        # Save .mgh overlay
        img = mgh.MGHImage(overlay_reshaped.astype(np.int32), affine=np.eye(4))
        img.to_filename(os.path.join(label_dir, main_label + '_' + hemi + '_binned.mgh'))
        
        # Set up df with coords, ids, and bins each vertex belongs to
        cx, cy, cz = coords[label_vertices,0],coords[label_vertices,1],coords[label_vertices,2]
        surf_data = pd.DataFrame(np.array([cx, cy, cz]).T, columns=['x_infl','y_infl','z_infl']) # coord on infl surf
        surf_data['vertex_id'] = np.arange(coords.shape[0])[label_vertices] # corresponding vertex id
        surf_data['bin'] = bin_ids # bin that vertex is assigned to
        
        # Now assign an ROI to each bin
        roi_vertices = []
        roi_n = []
        # Get vertex ids in each roi
        for roi in boundary_labels:
            roi_v = nib.freesurfer.read_label(os.path.join(label_dir, roi + '_' + hemi + '.label'))
            roi_vertices.extend(roi_v)
            roi_n.extend([roi for i in range(len(roi_v))])
            roi_info = pd.DataFrame({'roi':roi_n, 'vertices':roi_vertices})
        # Use vertex id to add ROI info to df
        roi_assignment = []
        for v_id in surf_data['vertex_id']:
            in_roi = roi_info[roi_info['vertices']==v_id]['roi'].values
            roi_assignment.append(in_roi)
        surf_data['roi'] = roi_assignment
        # For each bin, compute the most often seen ROI and treat that as the bin's ROI
        mode_per_bin = surf_data.groupby(['bin'])['roi'].agg(pd.Series.mode).to_list()
        
        # Get location of end of V1; V2/3/4 are hard to define on a PA axis, so just use the V1/extrastriate retinotopic as the border
        boundary_label_loc = [next(i for i, v in enumerate(mode_per_bin) if v[0] != 'V1')-.5]
        # Get starts of p,c,aIT, TP
        for boundary in ['pIT', 'cIT', 'aIT', 'TP']:
            boundary_label_loc.append(next(i for i, v in enumerate(mode_per_bin) if v[0] == boundary)-.5)
        roi_divisions.append(boundary_label_loc)

        # Now get the actual surface contrast data
        # For each run
        for (r_sid, r_iid) in zip(session_ids, run_ids):
            print(r_sid, r_iid)
            # Load run overlay effect size and variance
            if 'congruency' in r_sid:
                overlay_name = 'ic_c'
                effect_size_path = 'sigsurface_' + hemi + '_reg_' + r_sid + '_' + r_iid + '_incongruent_minus_congruent_effect_size_cfnii.mgh'
                variance_path = 'sigsurface_' + hemi + '_reg_' + r_sid + '_' + r_iid + '_incongruent_minus_congruent_effect_size_variance_cfnii.mgh'
                run_overlay_ef = os.path.join(congruency_mod_dir, 'per_run_contrasts', effect_size_path)
                run_overlay_var = os.path.join(congruency_mod_dir, 'per_run_contrasts', variance_path)
            else:
                overlay_name = 'ca_nca'
                effect_size_path = 'sigsurface_' + hemi + '_reg_' + r_sid + '_' + r_iid + '_uncolored_shape_minus_achromatic_shape_effect_size_cfnii.mgh'
                variance_path = 'sigsurface_' + hemi + '_reg_' + r_sid + '_' + r_iid + '_uncolored_shape_minus_achromatic_shape_effect_size_variance_cfnii.mgh'
                run_overlay_ef = os.path.join(scp_mod_dir, 'per_run_contrasts', effect_size_path)
                run_overlay_var = os.path.join(scp_mod_dir, 'per_run_contrasts', variance_path)
            
            surf_data['effect_size'] = nib.load(run_overlay_ef).get_fdata()[:,0,0][label_vertices]
            surf_data['variance'] = nib.load(run_overlay_var).get_fdata()[:,0,0][label_vertices]
            
            # Average values within each bin for each run
            binned_surf_data = surf_data.groupby(['bin'])[['effect_size', 'variance']].mean().reset_index()
            binned_surf_data['contrast'] = overlay_name # add paradigm info
            binned_surf_data['subj'] = subject
            binned_surf_data['hemi'] = hemi
            all_subj_data.append(binned_surf_data)

# Combine all binned data info into one df
binned_data = pd.concat(all_subj_data)
binned_data.loc[binned_data['subj']=='wooster', 'subj'] = 'w'
binned_data.loc[binned_data['subj']=='jeeves', 'subj'] = 'je'
binned_data['percent_dist'] = binned_data['bin']/64*100

# Compile where each roi transition is for each hemisphere
roi_divisions = np.array(roi_divisions)
lower = np.min(roi_divisions, axis=0)
upper = np.max(roi_divisions, axis=0)
roi_divisions_df = pd.DataFrame(roi_divisions.T, columns = ['w_lh', 'w_rh', 'je_lh', 'je_rh'])
roi_divisions_df['roi_boundaries'] = ['V1-V2/3/4', 'V4-pIT', 'pIT-cIT', 'cIT-aIT', 'aIT-TP']

# Save all out
binned_data.to_csv(os.path.join(outdir, 'surface_binned_data.csv'), index=False)
roi_divisions_df.to_csv(os.path.join(outdir, 'surface_binned_data_rois.csv'), index=False)