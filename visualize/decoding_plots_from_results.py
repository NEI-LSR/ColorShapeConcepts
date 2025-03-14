""""
Script to create nice plots form a decoding results file.
"""
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bin import plotter

# make sure text is saved in svgs as text, not path
plt.rcParams['svg.fonttype'] = 'none'

out_path = "plots/fig2_decoding"


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


def bootstrap_by_group(data, rois, folds=1000, x_sets=("identity", "cross", "icc", "ict")):
    """
    produce accuracy each roi, either combined over subject, sets, or not.
    return a  <subj, inset, xset: (id, cross, icc, ict), boot_folds, rois> matrix
    """
    subject = pd.unique(data['subject'])
    in_sets = pd.unique(data['train_set'])
    if x_sets is None:
        x_sets = pd.unique(data['test_set'])
    # create output dictionary
    boots = [[[[] for _ in x_sets] for _ in in_sets] for _ in subject]
    for i, s in enumerate(subject):
        sub_data = data[data['subject'] == s]
        for j, inset in enumerate(in_sets):
            s_data = sub_data[sub_data['train_set'] == inset]
            fxs = copy.copy(x_sets)
            for k, xset in enumerate(fxs):
                ts_data = s_data[s_data['test_set'] == xset]
                r_data = ts_data[rois]
                for b in range(folds):
                    boot_fold: pd.DataFrame = r_data.sample(frac=1., replace=True)
                    boot_fold = boot_fold.applymap(_extract_numeric)
                    mean = boot_fold.mean(axis=0, numeric_only=True).to_list()
                    boots[i][j][k].append(mean)
    boots = np.array(boots) # <s, is, xs, folds, roi>
    return boots


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
    roi_map = {
        'global': 'all',
        'V1': 'V1',
        'V2': 'V2',
        'V3': 'V3',
        'V4': 'V4',
        'pIT': 'pIT',
        'aIT': 'aIT',
        'TP': 'TP',
        'operculum': 'oprc',
        'oFC': 'oFC',
        'vFC': 'vFC',
        'dFC': 'dFC',
        'piSTS': 'piST',
        'aiSTS': 'aiST',
        'ppHC': 'ppHC',
        'apHC': 'apHC',
        'AC': 'AC',
        'V3a': 'V3a',
        'MT': 'MT',
        'IPS': 'IPS',
        'S1S2': 'SmC',
        'cingulate': 'cing',
        'agranular': 'MC',
        'insular': 'insl'
    }
    nc = []
    abv_rn = []
    for c in data.columns:
        if c in roi_map:
            c = roi_map[c]
            abv_rn.append(c)
        nc.append(c)
    data.columns = nc

    roi_names = list(roi_map.values())

    x_sets = ["identity", "cross", "icc", "ict"]

    boot_folds = 1000
    boots = bootstrap_by_group(data, roi_names, boot_folds, x_sets=x_sets)  # <s, is, xs, folds, roi>
    chance = (1/3)
    boots = 100 * (boots - chance)

    # plot grouped subject, combined over in set
    data = np.transpose(boots, (4, 2, 3, 0, 1))  # <roi, xs(id, x), folds, s, is, >

    # create shape identity plot
    figure = plt.figure(dpi=300, figsize=(4.0, 1))
    ax = figure.add_axes(plt.axes())
    s_id_data = data[:, 0, :, :, 0].mean(axis=-1)[:, None, ...]
    figure = plotter.create_save_barplot(ax, figure, "shape_identity", s_id_data, roi_names, out_dir=out_path)
    figure.show()

    # create color identity plot
    figure = plt.figure(dpi=300, figsize=(4.0, 1))
    ax = figure.add_axes(plt.axes())
    c_id_data = data[:, 0, :, :, 1].mean(axis=-1)[:, None, ...]
    figure = plotter.create_save_barplot(ax, figure, "color_identity", c_id_data, roi_names, out_dir=out_path)
    figure.show()

    # create combined cross decoding plot
    figure = plt.figure(dpi=300, figsize=(4, 1))
    ax = figure.add_axes(plt.axes())
    x_data = data[:, 1, :, :, :].mean(axis=(-1, -2))[:, None, ...]
    figure = plotter.create_save_barplot(ax, figure, "both_cross", x_data, roi_names, out_dir=out_path)
    figure.show()
    
    # create combined incorrect decoding plot
    figure = plt.figure(dpi=300, figsize=(4, 1))
    ax = figure.add_axes(plt.axes())
    x_data = data[:, 2:, :, :, :].mean(axis=(-1, -2)) # [:, None, ...]
    figure = plotter.create_save_barplot(ax, figure, "both_cross", x_data, roi_names, out_dir=out_path)
    figure.show()

    exit(0)


if __name__ == "__main__":
    # include paths to models for both subjects in list to create combined plots.
    model_paths = ["/home/bizon/shared/isilon/PROJECTS/ColorShapeContingency1/MTurk1/analysis/decoding/models/wooster_FLS___both_IC_2025-03-14_11-29"]
    csv_paths = (os.path.join(m, "results.csv") for m in model_paths)
    csv_data = combine_csv_rows(csv_paths)
    generate_decoding_plots(csv_data, None)
    plt.show()