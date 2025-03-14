"""
Script to compare linear and layered searchlights.
"""
import torch
from neurotools import decoding
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
from bin.simulation_dataloader import SimulationDataloader
import pandas as pd

if __name__ == "__main__":

    # used to test the effectiveness of the layered searchlight model
    DATA_ITER = 3000
    DATA_EXAMPLE_PER_CLASS = [60, 120, 180, 300, 420, 660]
    DIFFS = [3]
    RUNS_PER_ITER = 10

    atlas = np.zeros((16, 16, 16))
    atlas[:8, :12, :] = 1  # Blue
    atlas[8:, 4:, :] = 2  # cyan
    lookup = {1: "roi_1", 2: "roi_2"}

    # construct pairwise weights (to control class comparisons as in real exp.)
    main_light = [3, 7, 11]
    main_dark = [2, 6, 10]
    other_light = [1, 5, 9]
    other_dark = [4, 8, 12]
    # we only want comparisons within a set and within luminance levels
    set_wieghts = []
    pair_weights = torch.empty((12, 12, 12))
    for item_set in [main_light, other_light, main_dark, other_dark]:
        rows = torch.zeros((12, 12))
        cols = torch.zeros((12, 12))
        ind = torch.tensor(item_set) - 1
        cols[:, ind] = 1
        rows[ind, :] = 1
        weights = torch.logical_and(cols, rows).float()
        for t in item_set:
            pair_weights[t - 1] = weights
    pm = pair_weights.to("cuda")

    model_1 = decoding.ROISearchlightDecoder(atlas, lookup, set_names=("a", "b"), in_channels=1, n_classes=12,
                                             spatial=(16, 16, 16), nonlinear=False, device="cuda",
                                             base_kernel_size=2, n_layers=4, dropout_prob=0.5, pairwise_comp=pm)

    model_2 = decoding.ROISearchlightDecoder(atlas, lookup, set_names=("a", "b"), in_channels=1, n_classes=12,
                                             spatial=(16, 16, 16), nonlinear=False, device="cuda",
                                             base_kernel_size=5, n_layers=1, dropout_prob=0.0, pairwise_comp=pm)

    model_3 = decoding.ROISearchlightDecoder(atlas, lookup, set_names=("a", "b"), in_channels=1, n_classes=12,
                                             spatial=(16, 16, 16), nonlinear=True, device="cuda",
                                             base_kernel_size=2, n_layers=4, dropout_prob=0.5, pairwise_comp=pm)

    models = [model_1, model_2, model_3]

    data = {"model": [], "difficulty": [], "iters": [], "cross": [], "acc": []}

    for i, m in enumerate(models):
        sal_maps = []
        x_sals_maps = []
        x_acc_maps = []
        in_acc_maps = []
        for j, d in enumerate(DIFFS):
            for k, n in enumerate(DATA_EXAMPLE_PER_CLASS):
                for z in range(RUNS_PER_ITER):
                    m.initialize_params()  # reset all parameters to defualts
                    # Train Searchlight
                    m.train_searchlight("a", True)
                    m.train_predictors("a", False)
                    vdl = SimulationDataloader(difficulty=d, num_examples=n, seed=8 + z, stable_seed=z + 1, batch_size=70)
                    m.fit(vdl.batch_iterator(use_set="A", epochs=DATA_ITER))

                    # # Train identity weights
                    m.train_searchlight("a", False)
                    m.train_predictors("a", True)
                    m.fit(vdl.batch_iterator(use_set="A", epochs=DATA_ITER // 2))

                    # get sals
                    sals = m.get_saliancy().squeeze()
                    sals = (sals[:, :, 6:10]).mean(axis=2)
                    if n == 120 and d == 3 and len(sal_maps) == 0:
                        sal_maps.append(sals)

                    # Train cross weights
                    m.train_searchlight("b", False)
                    m.train_predictors("b", True)
                    m.fit(vdl.batch_iterator(use_set="B", epochs=DATA_ITER // 2))

                    # get X sals
                    sals = m.get_saliancy().squeeze()
                    sals = (sals[:, :, 6:10]).mean(axis=2)
                    if n == 120 and d == 3 and len(x_sals_maps) == 0:
                        x_sals_maps.append(sals)

                    # Evaluate
                    vdl = SimulationDataloader(difficulty=d, num_examples=n, seed=16 + z, stable_seed=z + 1, batch_size=70)
                    m.eval("a")
                    resa, in_acc_map, _ = m.predict(vdl.batch_iterator(use_set="A", epochs=50))
                    a_acc = resa["roi_2"].squeeze()
                    if n == 120 and d == 3 and len(in_acc_maps) == 0:
                        in_acc_maps.append(in_acc_map[:, :, 6:10].mean(axis=2))

                    m.eval("b")
                    resb, x_acc_map, _ = m.predict(vdl.batch_iterator(use_set="B", epochs=50))
                    b_acc = resb["roi_2"].squeeze()
                    if n == 120 and d == 3 and len(x_acc_maps) == 0:
                        x_acc_maps.append(x_acc_map[:, :, 6:10].mean(axis=2))

                    data["model"] += [i, i]
                    data["difficulty"] += [d, d]
                    data["iters"] += [n, n]
                    data["cross"] += [0, 1]
                    data["acc"] += [a_acc, b_acc]
        try:
            sal_map = np.stack(sal_maps).mean(axis=0)
            x_sals_maps = np.stack(x_sals_maps).mean(axis=0)
            in_acc_map = np.stack(in_acc_maps).mean(axis=0)
            x_acc_map = np.stack(x_acc_maps).mean(axis=0)

            plt.imshow(sal_map, vmin=None, vmax=None)
            plt.savefig("figures/simulation/model_" + str(
                i + 1) + "ID_sal_search.svg")

            plt.imshow(x_sals_maps, vmin=None, vmax=None)
            plt.savefig("figures/simulation/model_" + str(
                i + 1) + "_X_sal_search.svg")

            plt.imshow(in_acc_map, vmin=.34, vmax=.55)
            plt.savefig("figures/simulation/model_" + str(
                i + 1) + "_in_search.svg")

            plt.imshow(x_acc_map, vmin=.34, vmax=.46)
            plt.savefig("figures/simulation/model_" + str(
                i + 1) + "_x_search.svg")
        except Exception as e:
            raise ValueError

    data = pd.DataFrame.from_dict(data)
    data.to_csv("results/simuation/searchsim.csv")
