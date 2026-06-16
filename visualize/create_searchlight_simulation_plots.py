import copy

import numpy as np
import pandas as pd

from bin import plotter
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.autolayout': True})


ITERS = [50, 100, 200, 400, 800,]
DIFFS = [3]
m1 = 0
m2 = 1
ext = ""

def bootstrap_by_group(data, folds=1000,):
    """
    produce accuracy each roi, either combined over subject, sets, or not.
    return a  <subj, inset, xset, boot_folds, rois> matrix
    """
    models = [m1, m2]
    cross = pd.unique(data['cross'])
    diffs = DIFFS
    iters = ITERS

    # create output dictionary
    boots = [[[[[] for _ in iters] for _ in diffs] for _ in cross] for _ in models]
    corr_boots = copy.deepcopy(boots)
    for i, m in enumerate(models):
        sub_data = data[data['model'] == m]
        for j, c in enumerate(cross):
            c_data = sub_data[sub_data['cross'] == c]
            for k, d in enumerate(diffs):
                d_data = c_data[c_data['difficulty'] == d]
                for a, itr in enumerate(iters):
                    itr_data = d_data[d_data['iters'] == itr]
                    for b in range(folds):
                        boot_fold: pd.DataFrame = itr_data.sample(frac=1., replace=True)
                        mean = boot_fold["acc"].mean(axis=0, numeric_only=True)
                        corr_mean = boot_fold["circ_corr"].mean(axis=0, numeric_only=True)
                        boots[i][j][k][a].append(mean)
                        corr_boots[i][j][k][a].append(corr_mean)

    boots = 100 * (np.array(boots) - (1/3)) # <m, c, d, it, folds>
    corr_boots = np.array(corr_boots)
    return boots, corr_boots

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    return ax


def create_fig(data):
    out_path = "figures/simulation/"
    data = pd.read_csv(data)
    data.head()
    boot_fold, corr_boot_fold = bootstrap_by_group(data) # <m, c, d, it, folds>

    # create line plot over number of examples for cross data
    data = boot_fold.mean(axis=(-3,))[:, 1].squeeze() # m, it, folds # select only cross data
    figure = plt.figure(dpi=300, figsize=(1.5, 1.0))
    ax = figure.add_axes(plt.axes())
    figure = plotter.create_save_line_plot(ax, figure, ext + "sim_cross_over_iter", data, ["layer", "std"],
                                           x=ITERS, out_dir=out_path, xposition=0.0, yposition=0,
                                           set_size=(1.5, 1.0))
    figure.show()

    # circle corr for cross
    data = corr_boot_fold.mean(axis=(-3,))[:, 1].squeeze() # m, it, folds # select only cross data
    figure = plt.figure(dpi=300, figsize=(1.5, 1.0))
    ax = figure.add_axes(plt.axes())
    figure = plotter.create_save_line_plot(ax, figure, ext + "circcorr_cross_over_iter", data, ["layer", "std"],
                                           x=ITERS, out_dir=out_path, xposition=0.0, yposition=0,
                                           set_size=(1.5, 1.0))
    figure.show()

    # create line plot over number of examples for identity data
    data = boot_fold.mean(axis=(-3,))[:, 0].squeeze() # m, it, folds # select only cross data
    figure2 = plt.figure(dpi=300, figsize=(1.5, 1.0))
    ax2 = figure2.add_axes(plt.axes())
    figure = plotter.create_save_line_plot(ax2, figure2, ext + "sim_ID_over_iter", data, ["layer", "std"],
                                           x=ITERS, out_dir=out_path, xposition=0.0, yposition=0,
                                           set_size=(1.5, 1.0))
    figure.show()

    # circle corr for cross
    data = corr_boot_fold.mean(axis=(-3,))[:, 0].squeeze() # m, it, folds # select only cross data
    figure = plt.figure(dpi=300, figsize=(1.5, 1.0))
    ax = figure.add_axes(plt.axes())
    figure = plotter.create_save_line_plot(ax, figure, ext + "circcorr_ID_over_iter", data, ["layer", "std"],
                                           x=ITERS, out_dir=out_path, xposition=0.0, yposition=0,
                                           set_size=(1.5, 1.0))
    figure.show()


if __name__ == "__main__":
    data = "" # insert sim results path here
    create_fig(data)