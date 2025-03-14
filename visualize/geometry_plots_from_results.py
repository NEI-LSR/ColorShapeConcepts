
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from neurotools import geometry, util, embed
from bin import plotter

""""
Script to create nice plots from the results of compute_embedding
We want ot generate two types of plot in each ROI. One, we want to compute an average rdm and generate an average MDS
embedding plot coded with our colors.
Second we want to compute the rank correlation of the weighted MDS in each ROI, and get barplots of it with CIs in each
ROI.

"""

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
    'aIT': 'aIT',
    'TP': 'TP',
    'operculum': 'oprc',
    'oFC': 'oFC',
    'vFC': 'vFC',
    'dFC': 'dFC',
}


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


def bootstrap_by_group(data, rois, folds=1000):
    """
    produce accuracy each roi, either combined over subject, sets, or not.
    return a  <subj, inset, xset, boot_folds, rois> matrix
    """
    subject = pd.unique(data['subject'])
    in_sets = ["shape", "color"] #pd.unique(data['train_set'])
    # create output dictionary
    boots = [[[] for _ in in_sets] for _ in subject]
    for i, s in enumerate(subject):
        sub_data = data[data['subject'] == s]
        for j, inset in enumerate(in_sets):
            s_data = sub_data[(sub_data['train_set'] == inset) & (sub_data['test_set'] == "identity")] # (sub_data['train_set'] == inset) &
            r_data = s_data[rois]
            for b in range(folds):
                boot_fold: pd.DataFrame = r_data.sample(frac=1., replace=True)
                mean = boot_fold.mean(axis=0, numeric_only=True).to_list()
                boots[i][j].append(mean)
    boots = np.array(boots)  # <s, is, folds, roi>
    return boots


def get_light_dark_rdm(rdm):
    square_rdm = util.triu_to_square(rdm, 12)
    light_rdm = square_rdm[:, 0::2, 0::2]
    dark_rdm = square_rdm[:, 1::2, 1::2]
    triu_ind = np.triu_indices(light_rdm.shape[1], 1)
    light_rdm = light_rdm[:, triu_ind[0], triu_ind[1]]
    dark_rdm = dark_rdm[:, triu_ind[0], triu_ind[1]]
    return light_rdm, dark_rdm


def generate_plots(data):
    """
    """
    # these are true hex values of the colors.
    light_colors = ["#eb8ba7", "#caa65a", "#8dba7b", "#56bccd", "#9da3fe", "#e385f0"]
    dark_colors = ["#8d5461", "#7a6634", "#567248", "#327377", "#5f6695", "#88528b"]

    # some subset of ROIs we want to compute an MDS on
    for roi in geom_roi_map:
        # want to do separate for shape and color
        fig, ax = plt.subplots(2, 2)
        fig.tight_layout()
        for i, set in enumerate(["shape", "color"]):
            # get mean rdm to compute mds embeddings
            rdata = data[(data["train_set"] == set) & (data["test_set"] == "identity")][roi].tolist()
            mean_rdm = torch.concatenate(rdata, dim=0).mean(dim=0, keepdim=True)
            # we need to separate the rdms into light and dark for comparison
            light_rdm, dark_rdm = get_light_dark_rdm(mean_rdm)
            # initializes Multi-Dimmensional Scaler
            mds = embed.MDScale(6, 2)
            light_embeddings = mds.fit_transform(light_rdm.squeeze()).detach().numpy()
            mds = embed.MDScale(6, 2)
            dark_embeddings = mds.fit_transform(dark_rdm.squeeze()).detach().numpy()
            ax[i, 0].scatter(light_embeddings[:, 0], light_embeddings[:, 1], s=100, color=light_colors[:len(light_embeddings)])
            ax[i, 1].scatter(dark_embeddings[:, 0], dark_embeddings[:, 1], s=100, color=dark_colors[:len(dark_embeddings)])
            fig.suptitle(roi)
            fig.savefig(os.path.join(out_path, roi + "_mds_embedding.svg"))

    # now we compute spearman rho on each fold separately and then get confidence intervals over the corr coefs.
    corr_data = copy.copy(data)

    def corr_and_to_numpy(x):
        x = x.squeeze()
        light_rdm, dark_rdm = get_light_dark_rdm(x)
        light_rho = geometry.circle_corr(light_rdm, 6, metric="rho").detach().numpy()
        dark_rho = geometry.circle_corr(dark_rdm, 6, metric="rho").detach().numpy()
        rho = (light_rho + dark_rho) / 2
        return rho.squeeze()

    for roi in geom_roi_map:
        # replace rdms with spearman rho (w/ ties) with ideal circle of items.
        corr_data[roi] = corr_data[roi].apply(corr_and_to_numpy)

    boot_data = bootstrap_by_group(corr_data, list(geom_roi_map.keys()), folds=1000)  #<s, is, folds, roi>
    boot_data = boot_data.mean(axis=0).transpose((2, 0, 1))  # roi, is, folds

    figure = plt.figure(dpi=300, figsize=(4.0, 1))
    ax = figure.add_axes(plt.axes())
    figure = plotter.create_save_barplot(ax, figure, "shape_color_spearman_rho", boot_data, tuple(geom_roi_map.values()), out_dir=out_path)
    figure.show()



if __name__ == "__main__":
    std_rsa = False  # setting this to true will use loaded model weight on a standard searchlight RSA over 5x5x5 cubes.
                     # Otherwise, we will use latent model representations.
    model_paths = ["/home/bizon/shared/isilon/PROJECTS/ColorShapeContingency1/MTurk1/analysis/decoding/models/jeeves_FLS___both_IC_2025-03-14_07-16"]
    if std_rsa:
        csv_paths = (os.path.join(m, "vw_rdm_key.csv") for m in model_paths)
    else:
        csv_paths = (os.path.join(m, "rdm_key.csv") for m in model_paths)
    csv_data = get_csv_data(csv_paths)
    generate_plots(csv_data)
    plt.show()

