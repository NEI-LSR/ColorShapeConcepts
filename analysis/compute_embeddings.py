import os
import pickle
import pandas as pd
from neurotools import decoding, util, geometry
import numpy as np
from dataloader import TrialDataLoader
import nibabel as nib


"""
Script to take a trained ROI searchlight model and compute latent geometry for each ROI
In each ROI, will be weighted be weighted some of normalized latent space distances for each searchlight, weighted by confidence. 
"""

if __name__ == '__main__':
    # exp setup
    # Model path to load
    LOAD = "results/models/jeeves_LSDM"
    # Subjects to fit. Will do each in series.
    SUBJECT = 'jeeves'
    # time.sleep(7200)

    # Identity (IN) and Cross (X) set pairs to use. Will fit model for each index in series.
    TRAIN_SETS = ["shape", "color"]
    TRAIN_MODE = "identity"
    TEST_MODES = [TRAIN_MODE] + ["cross"]

    # model hyperparams
    # number of epochs per set.
    EPOCHS = 1
    # Leave pairwise as true for now.
    STRATEGY = "pairwise"
    # batch size for model. Set to the biggest value that will fit in GPU memory
    BATCH_SIZE = 100
    # number of cross validation folds.
    CV_FOLDS = 8
    # number of iterations. Will proceed to next CV fold on each iteration.
    n = 8
    # class sets to consider. Not important if not doing exemplar decoding. (i.e. don't ignore any classes)
    ITEM_SETS = ["both"]
    # number of channels in input data
    IN_CHAN = 2
    # device to use for models. Will be mad slow on CPU.
    DEV = "cuda"
    # Dataloader workers to use. USE 1 IF DEBUGGGING! otherwise >10 is good and fast if you have sufficient cpu ram.
    NUM_WORKERS = 1
    # If loading existing model, whether to train the searchlight for additional epochs, otherwise go straight to eval.
    RETRAIN = False
    # # whether to evaluate model performance by session, keep false if not using session info
    SEND_SESSIONS = False
    # If TRUE use trained LSDM latent space. if FALSE compute standard RSA rdms instead of model based (we'll still initialize an untrained model to access
    # its data management tools)
    USE_MODEL = True

    if STRATEGY not in ["multiclass", "pairwise"]:
        raise ValueError
    CONTENT_ROOT = "data"

    # Initial dictionary ot hold results:
    # we get one line for each model fit
    # roi level results will be initialized at runtime
    results = {"subject": [], "train_set": [], "test_set": []}
    USE_CLASSES = set(range(1, 13))

    CROP_WOOSTER = [(37, 93), (21, 85), (0, 42)]
    CROP_JEEVES = [(38, 89), (13, 77), (0, 42)]

    # link abbreviation to handle and dataloader behavior mode
    abv_map = {"color": "colored_blobs",
               "shape": "uncolored_shapes",
               "identity": "correct",
               "cross": "correct",
               "ict": "incorrect_stim",
               "icc": "incorrect_choice"}

    out_root = LOAD

    seed = 34
    DATA_KEY_PATH = CONTENT_ROOT + "/subjects/" + SUBJECT + "/analysis/shape_color_attention_decodemk2_nohighvar_stimulus_response_data_key.csv"
    FT = CONTENT_ROOT + "/subjects/" + SUBJECT + "/mri/functional_target.nii.gz"
    ROI_ATLAS = CONTENT_ROOT + "/subjects/" + SUBJECT + "/rois/major_divisions/final_atlas_old.nii.gz"
    ROI_LOOKUP = CONTENT_ROOT + "/subjects/" + SUBJECT + "/rois/major_divisions/lookup_match.txt"
    subj_dir = os.path.join(out_root, SUBJECT)
    try:
        os.mkdir(subj_dir)
    except FileExistsError:
        pass

    # create embedding results directory to house bunch of npy files.
    emb_dir = os.path.join(out_root, "geometry")
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)


    if SUBJECT == 'wooster':
        crop = CROP_WOOSTER
        # ["scd_20230623", "scd_20230804", "scd_20230806", "scd_20230813", "scd_20230821", "scd_20230824"]
        bad = ["scd1_20240308", "scd2_20240308",
               "scd_20230813", ]  # , "scd_20230806", "scd_20230824", "scd_20230804", "scd_20230813"]
    elif SUBJECT == 'jeeves':
        crop = CROP_JEEVES
        bad = []  # , "scd_20230918", "scd_20230813", "scd_20230919",
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
            results[k] = []  # this will have a store of pairwise distances for each fold

    # construct dataloader unique set names and behavior modes from training / eval program
    behavior_modes = []
    joined_set_names = []
    for tv in TRAIN_SETS:
        for v in TEST_MODES:
            jn = abv_map[tv] + "_" + abv_map[v]
            if jn not in joined_set_names:
                behavior_modes.append(abv_map[v])
                joined_set_names.append(jn)

    stats = {}
    latent_tracker = {}
    # for each iteration we choose a new fold of the data.
    for boot in range(n):
        if (boot % CV_FOLDS) == 0:
            seed += 42
            MTurk1 = TrialDataLoader(
                DATA_KEY_PATH,
                BATCH_SIZE,
                set_names=joined_set_names,
                content_root=CONTENT_ROOT, ignore_class=ignore_classes, crop=crop,
                cv_folds=int(CV_FOLDS), sep_class=False, std_by_feat=False, ignore_sessions=bad,
                mask=FT, seed=seed, cube=False, start_tr=1, end_tr=3,
                behavior_mode=behavior_modes, override_use_sess=None)
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
                    # helper to get model stats
                    if set not in stat_dict:
                        stat_dict[set] = (0, 1, 0, 1)  # MTurk1.get_pop_stats(set)
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
            if inset not in latent_tracker:
                latent_tracker[inset] = {}

            with open(os.path.join(boot_dir, str(boot) + "_model_binary.pkl"), "rb") as f:
                x_decoder = pickle.load(f)
            voxelwise = False

            if not USE_MODEL:
                print("WARN: Use model is set to False. Provided model will not be used"
                      "and we will compute standard rdms over K^3 voxels. Outputs will be saved in model directory "
                      "with voxelwise tag.")
                # initialize a clean model with kernel size of 5. (decoder will compute voxelwise rdms by defualt with this model)
                # since it is meaningless to compute RDMs on the channel dimension in this case
                nx_decoder = decoding.ROISearchlightDecoder(atlas, lookup, set_names=x_decoder.set_names, in_channels=IN_CHAN,
                                                           n_classes=len(USE_CLASSES), spatial=SPATIAL,
                                                           nonlinear=False,
                                                           device=DEV, base_kernel_size=5,
                                                           n_layers=1, dropout_prob=0.0, seed=8,
                                                           share_conv=False, pairwise_comp=x_decoder.pairwise_comp,
                                                           combination_mode="stack", mask=MTurk1.mask)
                # delete un-needed memory overhead (hacky)
                nx_decoder.conv_layers[0].weight = None
                nx_decoder.conv_layers[0].bias = None
                voxelwise = True

                # set stacking weights to loaded model (for combined ing spots)
                nx_decoder.weights = x_decoder.weights
                # remove loaded model.
                x_decoder = nx_decoder


            print("Initialized decoder")
            name = os.path.basename(LOAD)

            # need to train searchlight on the identity (in) set
            print("***************************************")
            print("RUNNING", SUBJECT, inset, "multiclass: boot iter", boot)
            print("***************************************")

            # Train weightings based on identity

            # fit wieghts for all
            for xmode in TEST_MODES:
                print("Analyzing", inset, xmode)

                if xmode == "cross":
                    dl_x_set = abv_map[TRAIN_SETS[(s + 1) % len(TRAIN_SETS)]] + "_" + abv_map[xmode]
                else:
                    dl_x_set = abv_map[inset] + "_" + abv_map[xmode]

                stats = _get_stats(dl_x_set)

                x_decoder.eval(xmode)
                # evaluate fit for all
                if USE_MODEL:
                    test_set = MTurk1.batch_iterator(dl_x_set, resample=False,
                                                     num_train_batches=1,
                                                     n_workers=NUM_WORKERS,
                                                     standardize=True, mode="test", fold=boot % int(CV_FOLDS),
                                                     stat_override=stats, meta_data=SEND_SESSIONS, random=False)
                    # WARM UP Batch Normalizers
                    x_decoder.predict(test_set)

                test_set = MTurk1.batch_iterator(dl_x_set, resample=False,
                                                 num_train_batches=1,
                                                 n_workers=NUM_WORKERS,
                                                 standardize=True, mode="test", fold=boot % int(CV_FOLDS),
                                                 stat_override=stats, meta_data=SEND_SESSIONS, random=False)
                # get latent embedding of each class in each roi
                rdm, rrdms = x_decoder.get_latent(test_set, metric="pearson", voxelwise=voxelwise)
                roi_indexes = x_decoder.roi_indexes
                roi_names = x_decoder.roi_names

                for roi in rrdms.keys():
                    rrdm = rrdms[roi]
                    assert(rrdm.shape[1]) == 66
                    ind = roi_names.index(roi)
                    ridx = roi_indexes[ind]
                    rname = roi + "_" + inset + "_" + xmode + "_rdm_boot_" + str(boot) + ".npy"
                    if voxelwise:
                        rname = "vw_" + rname
                    ### out path for rdm data file
                    rdm_out = os.path.join(emb_dir, rname)
                    np.save(rdm_out, rrdm)
                    results[roi].append(os.path.relpath(rdm_out, start=out_root))

                # Track accuracy results from each iteration, and set
                results["train_set"].append(inset)
                results["test_set"].append(xmode)
                results["subject"].append(SUBJECT)


        # make some quick plots of accurcies
        res_df = pd.DataFrame.from_dict(results)
        sets = pd.unique(res_df["test_set"])
        n_sets = len(sets)
        if voxelwise:
            df_out = os.path.join(out_root, "vw_rdm_key.csv")
        else:
            df_out = os.path.join(out_root, "rdm_key.csv")
        res_df.to_csv(df_out)
