"""
author: helen feibes
MT1 move nifti from functional to anatomical space to inflated hemisphere surface

Moves contrasts in functional space to inflated hemisphere surface
via functional->downsampled t1 space (which shares physical space with high res t1)->surface
i.e., there is no voxel scaling to the high res, etc., once image is in anat space
Smoothing if wanted is accomplished in functional space. 
To change surface projection parameters, look at --projfrac-max or avg; min, max, delta

Supports continuous (e.g. contrast, decoding maps) as well as binary (i.e., ROI) images

Uses: ANTs and Freesurfer commands
Adapted from: support_functions.apply_warp() and support_functions.generate_subject_overlays()
Does NOT require environment variable SSMRI pointing to SSMRI toolkit
"""

import os
import subprocess
import numpy as np
import nibabel as nib


def functional2surface(subject, functional_nii, proj_root="data", output_dir="."):
    sd = os.path.join(proj_root, 'subjects')
    subject_root = os.path.join(sd, subject)
    indicator = '.nii.gz'

    # Anatomical target files, downsampled, and fulll resolution
    t1_nifti = os.path.join(subject_root, 'mri', 'ds_T1.nii.gz')
    us_t1_nifti = os.path.join(subject_root, 'mri', 'T1.nii.gz')

    # Transform from functional to ds anatomical space
    fine_forward_transform = subject_root + '/mri/Composite.h5'
    forward_gross_transform = subject_root + '/mri/itkManual.txt'

    nifti_dir = os.path.dirname(functional_nii)
    fname = os.path.basename(functional_nii)

    # Loop through files in dir and transform corresponding ones
    if indicator in fname and 'reg' not in fname:
        nifti = functional_nii
        outname = os.path.join(nifti_dir, 'reg_'+fname)
        if np.array_equal(np.unique(nib.load(nifti).get_fdata()), np.array([0,1])):
            # if nifti is a binary file, use nearest neighbor and max projection to maintain binary structure
            interpolation = 'NearestNeighbor'
            projection = '--projfrac-max'
            surf_name = 'roisurface'
            target_anat = us_t1_nifti
        else:
            interpolation = 'Linear'
            projection = '--projfrac-avg'
            surf_name = 'sigsurface'
            target_anat = t1_nifti
        subprocess.run(['antsApplyTransforms',
                                        '--default-value', '0',
                                        '--dimensionality', '3',
                                        '--float', '0',
                                        '--input', nifti,
                                        '--input-image-type', '3',
                                        '--interpolation', interpolation,
                                        '--output', outname,
                                        '--reference-image', target_anat,
                                        '--transform', fine_forward_transform,
                                        '--transform', forward_gross_transform
                                        ])
        print('reg nifti saved at ', outname)

        for hemi in ['lh', 'rh']:
            surf_outname = os.path.join(output_dir, surf_name + '_' + hemi + '_reg_' + str(fname).split('.')[0]+str(fname).split('.')[1] + '.mgh')
            subprocess.run(['mri_vol2surf',
                            '--mov', outname,
                            '--regheader', subject,
                            projection, '.1', '.9', '.025',
                            '--interp', 'nearest',
                            '--hemi', hemi,
                            '--out', surf_outname,
                            '--sd', sd
                            ])
            print('saved at:', surf_outname)
            if surf_name == 'roisurface':
                # if roi def, convert overlay to label and fill in spotty areas
                label_outname = os.path.join(output_dir, hemi + '_reg_' + str(fname).split('.')[0]+str(fname).split('.')[1] + '.label')
                subprocess.run(['mri_cor2label',
                                '--i', surf_outname,
                                '--id', '1',
                                '--surf', subject, hemi,
                                '--l', label_outname,
                                '--sd', sd
                                ])
    else:
        raise ValueError("Improper input file.")
