import os
import pickle
import pandas as pd
import torch
from neurotools import decoding
import numpy as np
from dataloader import TrialDataLoader
import nibabel as nib
from matplotlib import pyplot as plt

"""
This script runs fits the layered searchlight model on both color and shape fMRI data trials in a cross validated (CV)
fashion. It produces a directory in results/models/ the contains the following:
    - a pickle file that is a binary of the model trained on each CV fold called `model_i.pkl` where i is the fold.
    - Maps of searchlight weights and accuracy over the whole brain, for both identity and cross decoding, trained on 
      both shape and color data.  These are compressed Nifti files in the functional space of the subject. These can be 
      transformed into the subject's anatomical space and then projected onto a cortical surface. 
    - the accuracy on each CV fold for each roi for both identity and cross decoding, trained on both shape and color
      data. This file is used as input for visualization code to generate bar plots. 
"""


if __name__ == '__main__':
    # exp setup

    """
    !!! CONFIGURATION !!!
    """

    # Subject to fit
    SUBJECT = 'wooster'

    # Identity (IN) and Cross (X) set pairs to use. Will fit model for each index in series.
    TRAIN_SETS = ["shape", "color"]
    TRAIN_MODE = "identity"
    TEST_MODES = [TRAIN_MODE] + ["cross", "icc", "ict"]

    # class sets to consider
    ITEM_SET = "both"
    # root of the data directory
    CONTENT_ROOT = "data"

    # model hyperparams
    # number of epochs per set.
    EPOCHS = 2500
    # Initial gradient learning rate
    LR = .005
    # base convolutional kernel size
    KERNEL = 2
    # Number of searchlight layers (if N_LAYERS is 1, degenerates to logistic searchlight over KERNEL cube of input.)
    N_LAYERS = 4
    # Weight dropout fraction
    DROPOUT = 0.3
    # batch size for model. Set to the biggest value that will fit in GPU memory
    BATCH_SIZE = 70
    # number of channels in input data
    IN_CHAN = 2

    # Validation parameters
    # number of cross validation folds. Will always train on all but one and test on remaining.
    CV_FOLDS = 5
    # whether to start at different fold
    start_fold = 0
    # number of iterations. Will proceed to next CV fold on each iteration.
    n = 5

    # Compute Parameters
    # device to use for models. Will be mad slow on CPU.
    DEV = "cuda"
    # Numer of Dataloader workers to use. USE 1 IF DEBUGGING! (usually) otherwise >10 is good if you have sufficient cpu ram.
    NUM_WORKERS = 14

    # Reload Arguments
    # path to existing model directory if loading existing. Otherwise, None.
    LOAD = None
    # If loading existing model, whether to train the searchlight for additional epochs, otherwise go straight to eval.
    FIT_SEARCH = True
    FIT_WEIGHTS = True

    if __name__ == "__main__":
        # Initial dictionary ot hold results:
        # we get one line for each model fit
        # roi level results will be initialized at runtime
        results = {"subject": [], "train_set": [], "test_set": [], "items": [], "global": []}

        # set, luminance level, class definitions
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
        pair_weights = pair_weights.to(DEV)
        USE_CLASSES = sorted(main_light + main_dark + other_light + other_dark)

        CROP_WOOSTER = [(37, 93), (21, 85), (0, 42)]
        CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]

        # link abbreviation to handle and dataloader behavior mode
        abv_map = {"color": "colored_blobs",
                   "shape": "uncolored_shapes",
                   "identity": "correct",
                   "cross": "correct",
                   "ict": "incorrect_stim",
                   "icc": "incorrect_choice"}

        # Create a directory to save everything from this model run in.
        name = SUBJECT + "_LSDM"

        if LOAD is None:
            out_root = os.path.join(CONTENT_ROOT, "results", "models", name)
            os.mkdir(out_root)
        else:
            out_root = LOAD

        seed = 34
        DATA_KEY_PATH = CONTENT_ROOT + "/subjects/" + SUBJECT + "/analysis/shape_color_attention_data_key.csv"
        FT = CONTENT_ROOT + "/subjects/" + SUBJECT + "/mri/functional_target.nii.gz"
        ROI_ATLAS = CONTENT_ROOT + "/subjects/" + SUBJECT + "/rois/major_divisions/final_atlas.nii.gz"
        ROI_LOOKUP = CONTENT_ROOT + "/subjects/" + SUBJECT + "/rois/major_divisions/lookup_match.txt"
        subj_dir = os.path.join(out_root, SUBJECT)
        try:
            os.mkdir(subj_dir)
        except FileExistsError:
            pass

        if SUBJECT == 'wooster':
            crop = CROP_WOOSTER
            bad = ["scd1_20240308", "scd2_20240308", "scd_20230813",]  # excluded sessions
            test_good = None
        elif SUBJECT == 'jeeves':
            crop = CROP_JEEVES
            bad = []
            test_good = None
        else:
            exit(1)

        all_classes = set(range(1, 15))
        ignore_classes = all_classes - set(USE_CLASSES)

        if not os.path.exists(ROI_ATLAS):
            raise ValueError

        atlas = nib.load(ROI_ATLAS).get_fdata()
        lookup_df = pd.read_csv(ROI_LOOKUP, skiprows=[0], sep="\t")
        lookup = lookup_df.set_index("No.")["Label Name:"].to_dict()
        # add cols for each roi
        for k in lookup.values():
            if k not in results:
                results[k] = []

        stats = {}
        map_tracker = {}

        # construct dataloader unique set names and behavior modes from training / eval program
        behavior_modes = []
        joined_set_names = []
        for tv in TRAIN_SETS:
            for v in TEST_MODES:
                jn = abv_map[tv] + "_" + abv_map[v]
                if jn not in joined_set_names:
                    behavior_modes.append(abv_map[v])
                    joined_set_names.append(jn)

        # for each iteration we choose a new fold of the data.
        for boot in range(n):
            if (boot % CV_FOLDS) == 0:
                seed += 42
                MTurk1 = TrialDataLoader(
                    DATA_KEY_PATH,
                    BATCH_SIZE,
                    set_names=tuple(joined_set_names),
                    content_root=CONTENT_ROOT, ignore_class=ignore_classes, crop=crop,
                    cv_folds=int(CV_FOLDS), sep_class=False, std_by_feat=False, ignore_sessions=bad,
                    mask=FT, seed=seed, cube=False, start_tr=1, end_tr=3,
                    behavior_mode=behavior_modes,
                    override_use_sess=test_good,
                    permute=False)

                if boot == 0:
                    atlas = MTurk1.crop_volume(atlas, cube=True)
                    SPATIAL = atlas.shape

                    def map_2_nifti(data, fname):
                        # saves nifti for data and returns the max 5 avg.
                        if type(data) != np.ndarray:
                            data = data.numpy()
                        data = np.nan_to_num(data, nan=0., posinf=0., neginf=0.)
                        sdata = np.sort(data.flatten())[-5:]
                        metric = sdata.mean()
                        data = MTurk1.to_full(data)
                        nii = nib.Nifti1Image(data, header=MTurk1.header, affine=MTurk1.affine)
                        nib.save(nii, fname)
                        return metric

                    stat_dict = {}

                    def _get_stats(set):
                        # this is legacy for a version where we did explicit normalization
                        if set not in stat_dict:
                            stat_dict[set] = (0, 1, 0, 1)
                        return stat_dict[set]

            # for each in train set
            for s, inset in enumerate(TRAIN_SETS):
                # first we load the inset data and train the searchlight, then train a mask to combined searchlights
                # over ROIS
                stats = _get_stats(abv_map[inset] + "_" + abv_map[TRAIN_MODE])
                # create directory for storing models and maps using searchlight trained on this set
                boot_dir = os.path.join(subj_dir, inset)
                if not os.path.exists(boot_dir):
                    os.mkdir(boot_dir)
                # track map results
                if inset not in map_tracker:
                    map_tracker[inset] = {}

                model_file = os.path.join(boot_dir, str(boot + start_fold) + "_model_binary.pkl")
                SETS = TEST_MODES

                if LOAD is not None and os.path.exists(model_file):
                    with open(model_file, "rb") as f:
                        x_decoder = pickle.load(f)
                    # ensure consistency
                    x_decoder._update_setnames(SETS)
                    x_decoder.atlas = atlas
                    x_decoder.lookup = lookup
                    x_decoder.roi_names = list(lookup.values())
                    x_decoder.roi_indexes = [atlas.flatten()==int(k) for k in lookup.keys()]
                else:
                    # initialize searchlight

                    if FIT_SEARCH is False or FIT_WEIGHTS is False:
                        print("FIT was FALSE, but a new model was constructed. "
                              "Setting FIT_SEARCH and FIT_WEIGHTS to TRUE...")
                        FIT_WEIGHTS = True
                        FIT_SEARCH = True
                    x_decoder = decoding.ROISearchlightDecoder(atlas, lookup, set_names=SETS, in_channels=IN_CHAN,
                                                               n_classes=len(USE_CLASSES), spatial=SPATIAL, nonlinear=False,
                                                               device=DEV, base_kernel_size=KERNEL,
                                                               n_layers=N_LAYERS, dropout_prob=0.4, seed=8,
                                                               share_conv=False, pairwise_comp=pair_weights,
                                                               combination_mode="stack", mask=MTurk1.mask,
                                                               smooth_kernel_sigma=0.)
                print("Initialized decoder with", x_decoder.get_model_size(), "parameters...")

                # need to train searchlight on the identity (in) set
                print("***************************************")
                print("RUNNING", SUBJECT, inset, "multiclass: boot iter", boot + start_fold)
                print("***************************************")

                dl_train_set = abv_map[inset] + "_" + abv_map[TRAIN_MODE]
                if FIT_SEARCH:
                    # # now train searchlight  on inset
                    x_decoder.train_searchlight(TRAIN_MODE, True)  # want to train model on in set
                    x_decoder.train_predictors(TRAIN_MODE, False)
                    train_set = MTurk1.batch_iterator(dl_train_set, resample=False,
                                                      num_train_batches=max(EPOCHS, 1), n_workers=NUM_WORKERS,
                                                      standardize=True, mode="train", fold=(boot + start_fold) % int(CV_FOLDS),
                                                      stat_override=stats)
                    sfit_hist = x_decoder.fit(train_set, lr=LR)
                    plt.plot(sfit_hist)
                    plt.savefig(os.path.join(boot_dir, str((boot + start_fold)) + "_searchlight_train_loss.svg"))
                if FIT_SEARCH or FIT_WEIGHTS:
                    # # now train stacking wieghts on inset
                    x_decoder.train_searchlight(TRAIN_MODE, False)  # want to train model on in set
                    x_decoder.train_predictors(TRAIN_MODE, True)
                    train_set = MTurk1.batch_iterator(dl_train_set, resample=False,
                                                      num_train_batches=max(EPOCHS // 2, 1), n_workers=NUM_WORKERS,
                                                      standardize=True, mode="train", fold=(boot + start_fold) % int(CV_FOLDS),
                                                      stat_override=stats)
                    pred_hist = x_decoder.fit(train_set, lr=LR)
                    plt.plot(pred_hist)
                    plt.savefig(os.path.join(boot_dir, str((boot + start_fold)) + "_weights_train_loss.svg"))
                    plt.close()
                    # Train weightings based on identity


                # fit wieghts for all
                for x, xmode in enumerate(SETS):
                    print("Evaluating", inset, xmode)

                    # If this is cross decoding or incorrect choice decoding we need to use the other "modality" data.
                    if xmode == "cross" or xmode == "icc":
                        dl_x_set = abv_map[TRAIN_SETS[(s + 1) % len(TRAIN_SETS)]] + "_" + abv_map[xmode]
                    else:
                        dl_x_set = abv_map[inset] + "_" + abv_map[xmode]

                    stats = _get_stats(dl_x_set)

                    # evaluate fit for all
                    x_decoder.eval(xmode)
                    # warm up (for batch norm) if not already warm from training.
                    test_set = MTurk1.batch_iterator(dl_x_set, resample=False,
                                                     num_train_batches=max(EPOCHS // 30, 10),
                                                     n_workers=NUM_WORKERS,
                                                     standardize=True, mode="test", fold=(boot + start_fold) % int(CV_FOLDS),
                                                     stat_override=stats)
                    x_decoder.predict(test_set)

                    # Evaluate
                    test_set = MTurk1.batch_iterator(dl_x_set, resample=False,
                                                     num_train_batches=max(EPOCHS // 15, 10),
                                                     n_workers=NUM_WORKERS,
                                                     standardize=True, mode="test", fold=(boot + start_fold) % int(CV_FOLDS),
                                                     stat_override=stats)
                    roi_accs, acc_map, _ = x_decoder.predict(test_set)

                    sal_map = x_decoder.get_saliancy()
                    if np.isnan(sal_map).sum() > 100:
                        print("DETECTED NANs on boot iter. Skipping....")
                        break

                    # remember results
                    if xmode not in map_tracker[inset]:
                        map_tracker[inset][xmode] = {"acc_map": [], "sal_map": []}

                    map_tracker[inset][xmode]["acc_map"].append(acc_map)
                    map_tracker[inset][xmode]["sal_map"].append(sal_map)

                    # Track accuracy results from each iteration, and set
                    results["subject"].append(SUBJECT)
                    results["train_set"].append(inset)
                    results["test_set"].append(xmode)
                    results["items"].append(ITEM_SET)

                    for k in roi_accs.keys():
                        results[k].append(roi_accs[k])

                    print("boot", boot + start_fold, inset, xmode, "done.")
                    print("global acc:", roi_accs["global"])

                # Will save an updated copy every boot iter in case of crash.
                res_df = pd.DataFrame.from_dict(results)
                try:
                    res_df.to_csv(os.path.join(out_root, "results.csv"))
                except Exception:
                    # probably permission denied due to file in use.
                    res_df.to_csv(os.path.join(out_root, "results_recover.csv"))
                # save out model binary
                out = model_file
                if FIT_SEARCH or FIT_WEIGHTS or LOAD is None:
                    with open(out, "wb") as f:
                        pickle.dump(x_decoder, f)
                # save out maps
                for s in map_tracker.keys():
                    for x in map_tracker[s].keys():
                        for k in map_tracker[s][x].keys():
                            m = map_tracker[s][x][k]
                            map_name = "_".join([SUBJECT, s, x, k])
                            odir_m = os.path.join(subj_dir, s, map_name)
                            m = np.stack(m, axis=0)
                            effect = m.mean(axis=0)
                            quant = np.quantile(m, q=.05, axis=0)
                            map_2_nifti(effect, odir_m + "_effect.nii.gz")
