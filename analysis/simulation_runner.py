"""
Script to compare linear and layered searchlights.
"""
import torch
from neurotools import decoding, util, geometry
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
from bin.simulation_dataloader import SimulationDataloader
import pandas as pd

if __name__ == "__main__":
    EXT = ""
    # used to test the effectiveness of the layered searchlight model
    DATA_ITER = 5000
    DATA_EXAMPLE_PER_CLASS = [50, 100, 200, 400, 800,]
    DIFFS = [3,] # [3, 4]
    RUNS_PER_ITER = 20

    atlas = np.zeros((16, 16, 16))
    atlas[:8, :12, :] = 1  # Blue
    atlas[8:, 4:, :] = 2  # cyan
    lookup = {1: "roi_1", 2: "roi_2"}

    # construct pairwise weights (to control class comparisons as in real exp.)
    # set, luminance level, class definitions
    main_light = [1, 3, 5]
    other_light = [2, 4, 6]
    main_dark = [7, 9, 11]
    other_dark = [8, 10, 12]

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
    pm = pair_weights = pair_weights.to("cuda")

    model_1 = decoding.ROISearchlightDecoder(atlas, lookup, set_names=("a", "b"), in_channels=1, n_classes=12,
                                             spatial=(16, 16, 16), nonlinear=False, device="cuda",
                                             base_kernel_size=2, n_layers=4, dropout_prob=0.0, pairwise_comp=pm)

    model_2 = decoding.ROISearchlightDecoder(atlas, lookup, set_names=("a", "b"), in_channels=1, n_classes=12,
                                             spatial=(16, 16, 16), nonlinear=False, device="cuda",
                                             base_kernel_size=5, n_layers=1, dropout_prob=0.0, pairwise_comp=pm)

    # models = [model_1, model_2,]
    models = [model_1, model_2, model_1, model_2]
    model_rsa_voxelwise = [False, True, False, True]
    use_group_diff = [False, False, True, True]

    data = {"model": [], "difficulty": [], "iters": [], "cross": [], "acc": [], "circ_corr": []}

    for i, m in enumerate(models):
        sal_maps = []
        x_sals_maps = []
        x_acc_maps = []
        in_acc_maps = []
        for j, d in enumerate(DIFFS):
            for k, n in enumerate(DATA_EXAMPLE_PER_CLASS):
                for z in range(RUNS_PER_ITER):
                    print("running model", i, "diff", d, "train examples", n, "iter", z)
                    m.initialize_params()  # reset all parameters to defualts
                    # Train Searchlight
                    m.train_searchlight("a", True)
                    m.train_predictors("a", False)
                    vdl = SimulationDataloader(difficulty=d, num_examples=n, seed=42 * z, stable_seed=42 * z + 1,
                                               batch_size=50, add_group_signal=use_group_diff[i])
                    m.fit(vdl.batch_iterator(use_set="A", epochs=DATA_ITER), lr=.05)

                    # # Train identity weights
                    m.train_searchlight("a", False)
                    m.train_predictors("a", True)
                    m.fit(vdl.batch_iterator(use_set="A", epochs=DATA_ITER // 2), lr=.05)

                    # get sals
                    sals = m.get_saliancy().squeeze()
                    sals = (sals[:, :, 6:10]).mean(axis=2)
                    if n == 200 and d == 3:
                        sal_maps.append(sals)

                    # get X sals
                    sals = m.get_saliancy().squeeze()
                    sals = (sals[:, :, 6:10]).mean(axis=2)
                    if n == 200 and d == 3:
                        x_sals_maps.append(sals)

                    # Evaluate
                    vdl = SimulationDataloader(difficulty=d, num_examples=n, seed=42 * z+2, stable_seed=42 * z + 1,
                                               batch_size=50, add_group_signal=use_group_diff[i])
                    m.eval("a")
                    # warmup
                    m.predict(vdl.batch_iterator(use_set="A", epochs=25))
                    # real
                    resa, in_acc_map, _ = m.predict(vdl.batch_iterator(use_set="A", epochs=100))
                    a_acc = resa["roi_2"].squeeze()
                    if n == 200 and d == 3:
                        in_acc_maps.append(in_acc_map[:, :, 6:10].mean(axis=2))
                    # Compute latent a
                    m.eval("a")
                    rdm, roi_rdm = m.get_latent(vdl.batch_iterator("A", 100), metric="pearson", voxelwise=model_rsa_voxelwise[i], level=0 if model_rsa_voxelwise[i] else 3)
                    r_rdm = roi_rdm["roi_2"]
                    square_rdm = util.triu_to_square(torch.from_numpy(r_rdm), 12).numpy()
                    square_rdm_l = square_rdm[:, 6:, 6:].squeeze()
                    square_rdm_d = square_rdm[:, :6, :6].squeeze()
                    square_rdm = square_rdm_d + square_rdm_l
                    triu_ind = np.triu_indices(6, 1)
                    sub_r_rdm = square_rdm[triu_ind]
                    if type(sub_r_rdm) is np.ndarray:
                        sub_r_rdm = torch.from_numpy(sub_r_rdm)
                    rho_a = geometry.circle_corr(sub_r_rdm.unsqueeze(0), 6, metric="rho").detach().cpu().item()


                    m.eval("b")
                    # warmup
                    m.predict(vdl.batch_iterator(use_set="B", epochs=25))
                    # real
                    resb, x_acc_map, _ = m.predict(vdl.batch_iterator(use_set="B", epochs=100))
                    b_acc = resb["roi_2"].squeeze()
                    if n == 200 and d == 3:
                        x_acc_maps.append(x_acc_map[:, :, 6:10].mean(axis=2))
                    # latent geometry b
                    rdm, roi_rdm = m.get_latent(vdl.batch_iterator("A", 100), metric="pearson", voxelwise=model_rsa_voxelwise[i], level=0 if model_rsa_voxelwise[i] else 3)
                    r_rdm = roi_rdm["roi_2"]
                    square_rdm = util.triu_to_square(torch.from_numpy(r_rdm), 12).numpy()
                    square_rdm_l = square_rdm[:, 6:, 6:].squeeze()
                    square_rdm_d = square_rdm[:, :6, :6].squeeze()
                    square_rdm = square_rdm_d + square_rdm_l
                    triu_ind = np.triu_indices(6, 1)
                    sub_r_rdm = square_rdm[triu_ind]
                    if type(sub_r_rdm) is np.ndarray:
                        sub_r_rdm = torch.from_numpy(sub_r_rdm)
                    rho_b = geometry.circle_corr(sub_r_rdm.unsqueeze(0), 6, metric="rho").detach().cpu().item()

                    data["model"] += [i, i]
                    data["difficulty"] += [d, d]
                    data["iters"] += [n, n]
                    data["cross"] += [0, 1]
                    data["acc"] += [a_acc, b_acc]
                    data["circ_corr"] += [rho_a, rho_b]

                    # print latest
                    print("RES: model", i, "diff", d, "train examples", n, "iter", z)
                    for k in data.keys():
                        print(k, ":", "A:", data[k][-2], "B:", data[k][-1])
                    sdata = pd.DataFrame.from_dict(data)
                    sdata.to_csv("results/simuation/searchsim" + EXT + ".csv")


        try:
            sal_map = np.stack(sal_maps).mean(axis=0)
            x_sals_maps = np.stack(x_sals_maps).mean(axis=0)
            in_acc_map = np.stack(in_acc_maps).mean(axis=0)
            x_acc_map = np.stack(x_acc_maps).mean(axis=0)

            plt.imshow(sal_map, vmin=None, vmax=None)
            plt.savefig("figures/simulation/model_" + str(
                i + 1) + "ID_sal_search" + EXT + ".svg")

            plt.imshow(x_sals_maps, vmin=None, vmax=None)
            plt.savefig("figures/simulation/model_" + str(
                i + 1) + "_X_sal_search" + EXT + ".svg")

            plt.imshow(in_acc_map, vmin=.34, vmax=.55)
            plt.savefig("figures/simulation/model_" + str(
                i + 1) + "_in_search" + EXT + ".svg")

            plt.imshow(x_acc_map, vmin=.34, vmax=.46)
            plt.savefig("figures/simulation/model_" + str(
                i + 1) + "_x_search" + EXT + ".svg")
        except Exception as e:
            print("FAILED")
            pass

    data = pd.DataFrame.from_dict(data)
    data.to_csv("results/simuation/searchsim" + EXT + ".csv")
