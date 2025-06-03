""""
Script to create nice plots form a decoding results file.


We want ot generate two types of plot in each ROI. One, we want to compute an average rdm and generate an average MDS
embedding plot coded with our colors.

Second we want to compute the rank correlation of the weighted MDS in each ROI, and get barplots of it with CIs in each
ROI.

"""
import copy
import math
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from neurotools import geometry, util, embed
from utility import plotter

# make sure text is saved in svgs as text, not path
plt.rcParams['svg.fonttype'] = 'none'

out_path = "scripts/plots/fig_geometry"

# all expected rois and there abreviations
geom_roi_map = {
    'V1': 'V1',
    'V2': 'V2',
    'V3': 'V3',
    'V4': 'V4',
    'pIT': 'pIT',
    'cIT': "cIT",
    'aIT': 'aIT',
    'TP': 'TP',
    'vFC': 'vFC',
    'dFC': 'dFC',
}

mds_sets = [("V1",), ("V4", "pIT"), ("aIT", "TP",), ("dFC", "vFC")]

def load_rdm(path, root):
    arr = torch.from_numpy(np.load(os.path.join(root, path)))
    return arr


def get_csv_data(files):
    """
    this data has a bunch of rois, each of which has and rdm npy datafile we nned to unpacl

    Parameters:
        file (List[str]): Path to the  CSV files.
    """
    # Load the CSV files into DataFrames
    dfs = []
    for f in files:
        root = os.path.dirname(f)
        df = pd.read_csv(f)
        for roi in geom_roi_map.keys():
            # get rdm matrix from path into df.
            df[roi] = df[roi].apply(load_rdm, args=(root,))
        dfs.append(df)
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

def get_light_dark_rdm(rdm):
    square_rdm = util.triu_to_square(rdm, 12)
    light_rdm = square_rdm[:, 0::2, 0::2]
    dark_rdm = square_rdm[:, 1::2, 1::2]
    triu_ind = np.triu_indices(light_rdm.shape[1], 1)
    light_rdm = light_rdm[:, triu_ind[0], triu_ind[1]]
    dark_rdm = dark_rdm[:, triu_ind[0], triu_ind[1]]
    return light_rdm, dark_rdm

def _set_size(w,h, ax=None):
    """
        force axis to take set size.
        w, h: width, height in inches
    """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    return ax

def generate_plots(data, do_mds=True, combined_light_dark=True):
    """
    """
    if combined_light_dark:
        n = ["both"]
    else:
        n = ["light", "dark"]

    if do_mds:
        # these are true hex values of the colors.
        light_colors = ["#eb8ba7", "#caa65a", "#8dba7b", "#56bccd", "#9da3fe", "#e385f0"]
        dark_colors = ["#8d5461", "#7a6634", "#567248", "#327377", "#5f6695", "#88528b"]
        # generate canonical
        fig, ax = plt.subplots(1, 2)
        fig.tight_layout()
        for i, c in enumerate((light_colors, dark_colors)):
            angles = 2 * np.pi * np.arange(len(c)) / len(c)
            x_coord = np.cos(angles)
            y_coord = np.sin(angles)
            ax[i].scatter(x_coord, y_coord, s=100, color=c)
            ax[i].set_aspect('equal', adjustable='box')
        fig.suptitle("Canonical")
        fig.savefig(os.path.join(out_path, "canonical.svg"))

        # some subset of ROIs we want to compute an MDS on
        for roi in mds_sets:
            # want to do separate for shape and color
            for i, set in enumerate(["identity", "cross"]):
                # get mean rdm to compute mds embeddings
                rd = data[(data["train_set"] == "color") & (data["test_set"] == set)]
                rdata = []
                for r in roi:
                    rdata += rd[r].tolist()
                rdm = (torch.concatenate(rdata, dim=0))
                light_rdm, dark_rdm = get_light_dark_rdm(rdm)
                light_rdm = light_rdm.mean(dim=0, keepdim=True)
                light_rdm = light_rdm / light_rdm.std()
                dark_rdm = dark_rdm.mean(dim=0, keepdim=True)
                dark_rdm = dark_rdm / dark_rdm.std()
                if combined_light_dark:
                    do = [light_rdm + dark_rdm]
                else:
                    do = [light_rdm, dark_rdm]
                for j, rdm in enumerate(do):
                    # we need to separate the rdms into light and dark for comparison
                    # initializes Multi-Dimmensional Scaler
                    fig, ax = plt.subplots(1, )
                    fig.tight_layout()
                    fig.suptitle("_".join(roi) + "_" + set + "_" + n[j])
                    best_emb = None
                    best_stress = np.inf
                    for _ in range(5):
                        mds = embed.MDScale(6, 2, initialization="xavier")
                        embedings = mds.fit_transform(rdm.squeeze()).detach().numpy()
                        if mds.stress_history[-1] < best_stress:
                            best_emb = embedings
                            best_stress = mds.stress_history[-1]
                    ax.scatter(best_emb[:, 0], best_emb[:, 1], s=50, color=light_colors[:len(best_emb)])
                    ax.margins(x=0.2, y=0.2)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax = _set_size(.5, .5, ax)
                    #ax[i].set_aspect('equal')
                    fig.savefig(os.path.join(out_path, "_".join(roi) + "_" + set + "_" + n[j] + "_mds_embedding.svg"))

    # now we compute spearman rho on each fold separately and then get confidence intervals over the corr coefs.
    corr_data = copy.copy(data)

    def corr_and_to_numpy(x, mode="both"):
        x = x.squeeze()
        light_rdm, dark_rdm = get_light_dark_rdm(x)

        if mode == "light":
            rho = geometry.circle_corr(light_rdm, 6, metric="pearson").detach().numpy()
        elif mode == "dark":
            rho = geometry.circle_corr(dark_rdm, 6, metric="pearson").detach().numpy()
        elif mode == "both":
            light_rho = geometry.circle_corr(light_rdm, 6, metric="pearson").detach().numpy()
            dark_rho = geometry.circle_corr(dark_rdm, 6, metric="pearson").detach().numpy()
            rho = (light_rho + dark_rho) / 2
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'light', 'dark', or 'both'.")
        return rho.squeeze()
    for m in n:
        lc_data = corr_data.copy()
        for roi in geom_roi_map:
            # replace rdms with spearman rho (w/ ties) with ideal circle of items.
            lc_data[roi] = lc_data[roi].apply(lambda x: corr_and_to_numpy(x, mode=m))

        boot_data_c = bootstrap_by_group(lc_data, list(geom_roi_map.keys()), folds=1000, directive={"train_set": "color",
                                                                                                    "test_set": "identity"})
        boot_data_s = bootstrap_by_group(lc_data, list(geom_roi_map.keys()), folds=1000, directive={"train_set": "shape",
                                                                                                    "test_set": "identity"})
        boot_data = np.stack([boot_data_c, boot_data_s], axis=1)  # roi, is, folds

        figure = plt.figure(dpi=300, figsize=(3.0, 1))
        ax = figure.add_axes(plt.axes())
        name = "id_shape_color_spearman_rho_" + m
        if std_rsa:
            name = "STD_RSA_" + name
        figure = plotter.create_save_barplot(ax, figure, name, boot_data,
                                             tuple(geom_roi_map.values()), out_dir=out_path, ylim=(-.25, .65))
        figure.show()



if __name__ == "__main__":
    std_rsa = False  # False will use LSDM latent states. Setting this to true will use loaded model weight on a standard searchlight RSA
    model_paths = ["results/models/wooster_LSDM", "results/models/jeeves_LSDM"]
    if std_rsa:
        csv_paths = (os.path.join(m, "vw_rdm_key.csv") for m in model_paths)
    else:
        csv_paths = (os.path.join(m, "rdm_key.csv") for m in model_paths)
    csv_data = get_csv_data(csv_paths)
    generate_plots(csv_data, do_mds=False, combined_light_dark=False)
    plt.show()

