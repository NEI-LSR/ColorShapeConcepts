""""
Script to create nice plots form a decoding results file.
"""
import copy
import math
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import plotter

# make sure text is saved in svgs as text, not path
plt.rcParams['svg.fonttype'] = 'none'

out_path = "plots/fig2_decoding"

generate_main = False
generate_supp = True

supress_x_label = True # supress x labels so plotting is consistent

def combine_csv_rows(files):
    """
    Parameters:
        file (List[str]): Path to the  CSV files.
    """
    # Load the CSV files into DataFrames
    dfs = [pd.read_csv(f) for f in files]
    # Combine the DataFrames by rows
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    return combined_df


def _extract_numeric(x):
    # little func to deal with nested numeric types in dataframes
    try:
        y = eval(x)
        y = y[0]
        return y
    except Exception:
        return x


def bootstrap_by_group(data, rois, folds=1000, directive={}):
    """
    produce accuracy each roi, either combined over subject, sets, or not.
    provide directive t selct by (e.g. {col: values})
    return a  <subj, inset, xset: (id, cross, icc, ict), boot_folds, rois> matrix
    """
    # create output dictionary
    boots = []
    data = copy.copy(data)
    # directive specifies which columns to keep sepearte. By defualt will sample over all.
    for key in directive:
        data = data[data[key]==directive[key]]
    # We're only interested in the roi columns that hold data
    data = data[rois]
    data = data.map(_extract_numeric) # make sure right numeric format in roi cols
    data.select_dtypes(include='number')
    for _ in range(folds):
        ld = copy.copy(data.sample(frac=1., replace=True))
        mean = ld.mean(axis=0, numeric_only=True).to_numpy() # <rois,>
        boots.append(mean)
    return np.stack(boots, axis=1) # <rois, folds>


def generate_decoding_plots(csv_data, roi_names, output_dir='plots'):
    """
    Generates bar plots for decoding results.

    Parameters:
        csv_path (str): Path to the CSV file containing the decoding results.
        roi_names (list): List of column names representing ROIs (regions of interest).
        output_dir (str): Directory to save the generated plots (default: 'plots').
    """
    # Load the CSV file
    data = csv_data

    # func to shorten ROI names
    # reduced set
    roi_map = {
        'global': 'all',
        'V1': 'V1',
        'V2': 'V2',
        'V3': 'V3',
        'V4': 'V4',
        'pIT': 'pIT',
        'cIT': 'cIT',
        'aIT': 'aIT',
        'TP': 'TP',
        'oFC': 'oFC',
        'vFC': 'vFC',
        'dFC': 'dFC',
    }
    if generate_supp:
        roi_map.update({'V3a': 'V3a',
                    'MT': 'MT',
                    'IPS': 'IPS',
                    'S1S2': 'SC',
                    'A1': 'A1',
                    'dSTS': "dSTS",
                    'pHC': 'MTL',
                    'operculum': 'oprc',
                    'cingulate': 'CC',
                    'premotor': 'pMC',
                    'M1': 'M1',
                    'insular': 'IC',
                    'Striatum': 'str',
                    'Hippo.': 'hpc',
                    })
    nc = []
    abv_rn = []
    for c in data.columns:
        if c in roi_map:
            c = roi_map[c]
            abv_rn.append(c)
        nc.append(c)
    data.columns = nc

    roi_names = list(roi_map.values())

    boot_folds = 1000

    if generate_main:
        boots = bootstrap_by_group(data, roi_names, boot_folds, directive={"train_set": "shape", "test_set": "identity"})  # <rois, folds>
        chance = (1/3)
        boots = 100 * (boots - chance)

        # create shape identity plot
        figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
        ax = figure.add_axes(plt.axes())
        s_id_data = boots[:, None, ...]
        figure = plotter.create_save_barplot(ax, figure, "shape_identity", s_id_data, roi_names, out_dir=out_path, ylim=(-3, 45))
        figure.show()


        boots = bootstrap_by_group(data, roi_names, boot_folds, directive={"train_set": "color", "test_set": "identity"})  # <rois, folds>
        chance = (1/3)
        boots = 100 * (boots - chance)

        # create color identity plot
        figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
        ax = figure.add_axes(plt.axes())
        c_id_data = boots[:, None, ...]
        figure = plotter.create_save_barplot(ax, figure, "color_identity", c_id_data, roi_names, out_dir=out_path, ylim=(-3, 45))
        figure.show()


        boots = bootstrap_by_group(data, roi_names, boot_folds, directive={"test_set": "cross"})  # <rois, folds>
        chance = (1/3)
        boots = 100 * (boots - chance)

        # create combined cross decoding plot
        figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
        ax = figure.add_axes(plt.axes())
        x_data = boots[:, None, ...]
        figure = plotter.create_save_barplot(ax, figure, "both_cross", x_data, roi_names, out_dir=out_path, ylim=(-3, 14))
        figure.show()


        boots = bootstrap_by_group(data, roi_names, boot_folds, directive={"test_set": "ict"})  # <rois, folds>
        chance = (1/3)
        boots = 100 * (boots - chance)

        # create combined incorrect decoding plot
        figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
        ax = figure.add_axes(plt.axes())
        x_data = boots[:, None, ...]
        figure = plotter.create_save_barplot(ax, figure, "incorrect_true_combined", x_data, roi_names, out_dir=out_path, ylim=(-3, 45))
        figure.show()

        boots = bootstrap_by_group(data, roi_names, boot_folds, directive={"test_set": "icc"})  # <rois, folds>
        chance = (1 / 3)
        boots = 100 * (boots - chance)

        # create combined incorrect decoding plot
        figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
        ax = figure.add_axes(plt.axes())
        x_data = boots[:, None, ...]
        figure = plotter.create_save_barplot(ax, figure, "incorrect_choice_combined", x_data, roi_names, out_dir=out_path, ylim=(-3, 10))
        figure.show()


    ## CREATE INDIVIDUAL SUBJECT PLOTS:

    if generate_supp:
        subjects = ["jeeves", "wooster", ]
        for s in subjects:
            boot_folds = 1000
            boots = bootstrap_by_group(data, roi_names, boot_folds,
                                       directive={"subject": s, "train_set": "shape", "test_set": "identity"})  # <rois, folds>
            chance = (1 / 3)
            boots = 100 * (boots - chance)

            # create shape identity plot
            figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
            ax = figure.add_axes(plt.axes())
            s_id_data = boots[:, None, ...]
            figure = plotter.create_save_barplot(ax, figure, s + "_shape_identity", s_id_data, roi_names, out_dir=out_path, ylim=(-3, 48), rotate_x_labels=True, set_size=(2.62, 0.62), suppress_x_label=supress_x_label)
            figure.show()

            boots = bootstrap_by_group(data, roi_names, boot_folds,
                                       directive={"subject": s, "train_set": "color", "test_set": "identity"})  # <rois, folds>
            chance = (1 / 3)
            boots = 100 * (boots - chance)

            # create color identity plot
            figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
            ax = figure.add_axes(plt.axes())
            c_id_data = boots[:, None, ...]
            figure = plotter.create_save_barplot(ax, figure, s + "_color_identity", c_id_data, roi_names, out_dir=out_path, ylim=(-3, 48), rotate_x_labels=True, set_size=(2.62, 0.62), suppress_x_label=supress_x_label)
            figure.show()

            boots = bootstrap_by_group(data, roi_names, boot_folds, directive={"subject": s, "train_set": "shape", "test_set": "cross"})  # <rois, folds>
            chance = (1 / 3)
            boots = 100 * (boots - chance)

            # create combined cross decoding plot
            figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
            ax = figure.add_axes(plt.axes())
            x_data = boots[:, None, ...]
            figure = plotter.create_save_barplot(ax, figure, s + "_color_to_shape_cross", x_data, roi_names, out_dir=out_path, ylim=(-3, 15), rotate_x_labels=True, set_size=(2.62, 0.62), suppress_x_label=supress_x_label)
            figure.show()

            boots = bootstrap_by_group(data, roi_names, boot_folds, directive={"subject": s, "train_set": "color", "test_set": "cross"})  # <rois, folds>
            chance = (1 / 3)
            boots = 100 * (boots - chance)

            # create combined cross decoding plot
            figure = plt.figure(dpi=300, figsize=(2.62, 0.82))
            ax = figure.add_axes(plt.axes())
            x_data = boots[:, None, ...]
            figure = plotter.create_save_barplot(ax, figure, s + "_shape_to_color_cross", x_data, roi_names, out_dir=out_path, ylim=(-3, 15), rotate_x_labels=True, set_size=(2.62, 0.62), suppress_x_label=supress_x_label)
            figure.show()


    exit(0)


if __name__ == "__main__":
    # include paths to models for both subjects in list to create combined plots.
    #model_paths = ["/home/bizon/shared/isilon/PROJECTS/ColorShapeContingency1/MTurk1/analysis/decoding/models/jeeves_FLS___both_IC_2025-04-07_08-09", "/home/bizon/shared/isilon/PROJECTS/ColorShapeContingency1/MTurk1/analysis/decoding/models/wooster_FLS___both_IC_2025-04-06_16-05"]
    model_paths = ["results/models/jeeves_LSDM", "results/models/wooster_LSDM"]
    csv_paths = (os.path.join(m, "results.csv") for m in model_paths)
    csv_data = combine_csv_rows(csv_paths)
    generate_decoding_plots(csv_data, None)
    plt.show()