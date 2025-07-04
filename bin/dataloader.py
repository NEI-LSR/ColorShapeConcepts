import copy
import math
import queue
import random
from neurotools.support import StratifiedSampler
import numpy as np
import nibabel as nib
import pandas as pd
import os
from neurotools import util
import multiprocessing as mp
from scipy.ndimage import affine_transform, gaussian_filter
import warnings
mp.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore")

# default value map for decoding active task. Will be used if none are provided. Maps MRI task indexes to shared index
# for associated color shape pairs
val_map = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14,
           8: 2, 9: 4, 10: 6, 11: 8, 12: 10, 13: 12, 14: 14,
           15: 1, 16: 3, 17: 5, 18: 7, 19: 9, 20: 11, 21: 13,
           22: 1, 23: 3, 24: 5, 25: 7, 26: 9, 27: 11, 28: 13}


class TrialDataLoader:
    """
    Designed to load batches of MTurk1 Decoding data from a data key csv file generated by "get_run_betas" subject level
    commands in parallel. That is, multiple batches are loaded and prepared while one is served.
    """

    def __init__(self, key_path, batch_size, set_names=("all",), class_map=val_map, crop=((9, 73), (0, 40), (13, 77)),
                 content_root="./", device="cuda", ignore_sessions=(), ignore_class=(), seed=456789, cv_folds=5,
                 behavior_mode="correct", verbose=False, mask=None, return_meta_data=False, replace=True, cube=True,
                 start_tr=1, end_tr=3, override_use_sess=None):
        """
        Parameters
        ----------
        key_path: <str> Path to the data key csv file that links class types, class labels, and paths to data for each
                        Required to have the following columns:
                        - condition_group: strs that indicate the type of set this condition belongs to, e.g. for mturk1 task
                                          would be "uncolored_shape" or "colored_blob"
                        - condition_integer: indicates the class that a stimulus belongs to. If a separate class_map is provided,
                                             it will remap these values internally. If you intend to do cross decoding, entries
                                             in each set in `condition_group` should have matching 'condition_integer' e.g. for
                                             mturk1 decoding task, `up_arrow` and `dark_red` would have the same condition_integer but
                                             different condition_group
                        - beta_path: paths to the input data files for this exemplar.
                        - session: column to provide information about data collection, i.e. a key used to denote parts of data
                                   across which signal may vary in ways not relevant to task. We ensure sessions are evenly represented
                                   across the resampled dataset during cross validation.
        batch_size: <int> number of independent examples to serve in each batch.
        set_names: List[str, ...] which condition groups will we serve data for.
        class_map: Dict[int:int] used to remap conditon_integers to new values. For SCD, remaps paradigm indexes to shared 1-14 indexes. Must have a mapping
                                for each unique condition_integer. If None, no remapping is performed. Defualt is mapping for SCD.
        crop: Tuple[Tuple[int, int], ...] bounding box to crop each example.
        content_root: <str> working directory from which paths in key will be taken realtive to
        device: Hardware device to return data on.
        ignore_sessions: Sessions to exclude from dataset
        ignore_class: Class integers / Condition integers to exclude from dataset
        seed: Random seed for dataloader
        cv_folds: default=5, How many cross validation folds are we using? defualt
        behavior_mode: Union[str, tuple, list], ['all', 'correct', 'incorrect_true_label', 'incorrect_choice_label'] whether to return all trials, just those where monkey chose
                        correctly, or just those were monkey made a choice but it was incorrect - with either the ground truth label or the choice that was made.
        verbose
        mask: mask used to zero regions of input without signal.
        return_meta_data: whether to return session information for each batch
        cube: whether to for served data to be cubic on spatial dimensions.
        """
        self.replace = replace
        self.return_meta = return_meta_data
        self.content_root = content_root
        data = pd.read_csv(key_path)
        try:
            data.drop("Unnamed: 0", axis=1, inplace=True)
        except KeyError:
            pass
        # fix missing expected cols
        if "session" not in data.columns:
            sess_df = pd.DataFrame({"session": [None] * len(data)})
            data = pd.concat([data, sess_df], axis=1)
        if "condition_group" not in data.columns:
            sess_df = pd.DataFrame({"condition_group": ["all"] * len(data)})
            data = pd.concat([data, sess_df], axis=1)
        if "correct" not in data.columns:
            sess_df = pd.DataFrame({"correct": [1] * len(data)})
            data = pd.concat([data, sess_df], axis=1)

        if override_use_sess is not None:
            sessions = set(pd.unique(data["session"]))
            ignore_sessions = sessions - set(override_use_sess)
        for ignore_sess in ignore_sessions:
            data = data[data["session"] != ignore_sess]
        data = data.sort_values(["session", "ima"])
        self.data = data.sample(frac=1.0, ignore_index=True, random_state=seed + 3, replace=False).reset_index(
            drop=True)  # only shuffling step

        self.seed = seed
        self.batch_size = batch_size
        self.full_size = None
        self.affine = None
        self.header = None
        self.ignore_class = ignore_class
        self.class_map = class_map
        self.verbose = verbose
        self.crop = crop
        self.cube = cube
        self.start_tr = start_tr
        self.end_tr = end_tr

        size = tuple([t[1] - t[0] for t in crop])

        if mask is not None:
            mask = self.crop_volume(nib.load(mask).get_fdata(), cube=False).astype(float)
            gaussian_filter(mask, sigma=1)
            self.mask = mask > .2 * mask.std()
        self.cv_folds = int(cv_folds)

        self.full_sets = {}
        self.folded_sets = {}

        # resample and stratify each set.
        self.set_names = set_names
        if type(behavior_mode) is str:
            self.behavior_mode = [behavior_mode] * len(set_names)
        assert len(set_names) == len(behavior_mode)
        for i, set_name in enumerate(set_names):
            # reindex according to map and behavior
            lset, options = self._get_set(set_name, behavior_mode=behavior_mode[i])
            if i == 0:
                self.options = options
            ldata = lset[~lset["condition_integer"].isin(self.ignore_class)]
            self.full_sets[set_name] = ldata
            # generate stratified folds for each set type
            self.folded_sets[set_name] = StratifiedSampler(ldata, self.cv_folds, target_col="condition_integer",
                                                           strat_cols=["session"], stratify=True)
        # get all unique sessions
        self.sessions = np.unique(
            np.concatenate([self.full_sets[s]["session"].to_numpy() for s in self.set_names])).tolist()

        print(self.sessions)
        print("Total included sessions:", len(self.sessions))

        self.shape = size
        self.device = device
        self._processed_ = 0
        self._max_q_size_ = 20

        self.set_mem = {}

    def _get_set(self, id, behavior_mode="correct", not_paired=True):
        ###############
        if id in self.data["condition_group"]:
            data = self.data[self.data["condition_group"] == id].copy()
        else:
            # find where part of the setname matches a valid group.
            cds = pd.unique(self.data["condition_group"])
            for cd in cds:
                if cd in id:
                    data = self.data[self.data["condition_group"] == cd].copy()
                    break
        print("")
        print("TOTAL", id, "blocks", len(data))
        #
        # These behavior modes are conceived for the Color-shape task, but are fairly general.
        if behavior_mode == "correct":
            data = data[data["correct"] == 1].reset_index(drop=True)  # should be == 1
        elif behavior_mode == "incorrect_choice" or behavior_mode == "incorrect_stim":
            data = data[(data["correct"] == 0) & (data["make_choice"] == 1)].reset_index(drop=True)
            if behavior_mode == "incorrect_choice":
                # need to create map of names to integers
                map = {}
                cond_names = self.data["condition_name"].tolist()
                cond_ints = self.data["condition_integer"].tolist()
                for i, n in enumerate(cond_names):
                    if n not in map:
                        map[n] = cond_ints[i]

                def choice_name_to_int(x):
                    if x in map:
                        return map[x]
                    else:
                        raise ValueError("Condition", x, "not in map")

                # map choice made values to target integers and replace the target integers.
                data["condition_integer"] = data["choice_name"].apply(choice_name_to_int)
        else:
            print("No behavior mode set. Using all data...")
            data = data.reset_index(drop=True)
        vals = sorted(pd.unique(data["condition_integer"]))
        if self.class_map is not None:
            reindexed_data = data.copy()
            # reidex to match standard key
            for i, item in enumerate(vals):
                reindexed_data["condition_integer"].loc[
                    data["condition_integer"] == item] = int(self.class_map[item])
            data = reindexed_data
        class_options = set(pd.unique(data["condition_integer"]).tolist())
        options = sorted(list(class_options - set(self.ignore_class)))

        # get standard affine for niis
        nii = nib.load(
            os.path.join(self.content_root, eval(data["beta_path"][0])[0]))
        self.full_size = nii.get_fdata().shape
        self.affine = nii.affine
        self.header = nii.header

        return data, options

    def pad_to_cube(self, input, time_axis=0):
        # changes an array to smallest possible n-cube
        if self.cube:
            return util._pad_to_cube(input, time_axis=time_axis)
        else:
            return input

    def to_full(self, data: np.array):
        """
        pads an array in dataloader cropped dimension to original file size. Useful for saving niftis that are in the same coordinate space.
        :param data: 3 dimmensional array
        :return: 3 dimmensional array padded to full size
        """
        full = np.zeros(self.full_size, dtype=float)
        crop_size = self.shape
        crop = []
        for i, c in enumerate(self.crop):
            dim = c[1] - c[0]
            diff = crop_size[i] - dim
            pad_l = int(math.floor(diff / 2))
            pad_r = int(math.ceil(diff / 2))
            crop.append((pad_l, crop_size[i] - pad_r))
        full[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1], self.crop[2][0]:self.crop[2][1]] = \
            data[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
        return full

    def clip_and_norm(self, x_in, example_normalization=False):
        """
        Parameters
        ----------
        input: <n, c, x, y, z>

        Returns
        -------
        """
        thresh_std = x_in[..., self.mask].std() * 6
        x_in = np.clip(x_in, -1 * thresh_std, thresh_std)
        ndims = x_in.ndim
        # mask = self.mask.reshape(self.mask.shape)
        mean = x_in[..., self.mask].reshape((len(x_in), -1)).mean(axis=1)  # mean across each trial
        std = x_in[..., self.mask].reshape((len(x_in), -1)).std(axis=1)  # std across ech trial
        x_in = (x_in - mean.reshape((-1,) + tuple([1] * (ndims - 1)))) / std.reshape((-1,) + tuple([1] * (ndims - 1)))
        x_in = np.nan_to_num(x_in, nan=0., posinf=0., neginf=0.)
        x_in[..., np.logical_not(self.mask)] = 0.
        return x_in

    def crop_volume(self, input, cube=True):
        out = input[self.crop[0][0]:self.crop[0][1], self.crop[1][0]:self.crop[1][1],
              self.crop[2][0]:self.crop[2][1]]
        if cube:
            out = self.pad_to_cube(out)
        return out

    def data_generator(self, dset, noise_frac_var=.1, spatial_translation_var=.33, max_base_examples=1,
                       cube=True, get_sess_data=False, example_class=None, index_override=None):
        """
        Combines some random number of examples from one random class with random weight with some amount of random translation and noise
        Parameters
        ----------
        fst: feature std
        fm: feature mean
        m: session means dict
        st: session stds dict
        dset
        noise_frac_var
        spatial_translation_var
        max_base_examples

        Returns
        -------
        """
        if index_override is None:
            options = self.options
            avail_options = list(set(pd.unique(dset["condition_integer"])).intersection(set(options)))
            if example_class is None:
                example_class = int(random.choice(avail_options))  # which class will this be?
            target = options.index(example_class)
            basis_dim = np.random.randint(1, max_base_examples + 1)  # how many real examples to use as basis?
            trajectory = np.random.random((basis_dim, 1, 1, 1, 1))  # weight vector for combining basis
            trajectory = trajectory / np.sum(trajectory)
            class_dset = dset[dset['condition_integer'] == example_class]
            if len(class_dset) == 0:
                return -1
            names = class_dset['condition_name']
            basis_idxs = np.random.randint(0, len(class_dset), size=(basis_dim,))
            if self.verbose:
                print("chose class", example_class, "desc", names.iloc[int(basis_idxs[0])], "basis examples",
                      len(class_dset))

            data = class_dset.iloc[basis_idxs]
        else:
            basis_idxs = np.array([index_override])
            trajectory = np.ones((1, 1, 1, 1, 1))
            data = dset.iloc[basis_idxs]
            condint = int(data["condition_integer"])
            if condint not in self.options:
                return -1
            target = self.options.index(condint)

        chosen_index = data.index[0]
        sessions = data["session"]
        beta_coef = []
        fail = False
        for z, path_str in enumerate(data["beta_path"]):
            chan_beta = []
            paths = eval(path_str)
            # ignore the final fir
            if type(paths) is str:
                paths = [paths]
            for path in paths[self.start_tr:self.end_tr]:
                # selest unnormed path to test something.
                # path = path.replace("mk2", "")
                nam = os.path.basename(path)
                path = os.path.join(os.path.dirname(path), nam)
                if path not in self.set_mem:
                    try:
                        beta_nii = nib.load(os.path.join(self.content_root, path))
                        b = beta_nii.get_fdata()
                        betas = self.crop_volume(b, cube=cube).astype(float)
                        betas = np.nan_to_num(betas, nan=0., posinf=0., neginf=0.)
                    except Exception as e:
                        print("data generator failed with", e)
                        return -1
                    nan_count = np.count_nonzero(np.isnan(betas))
                    if nan_count > 0:
                        print("WARNING:", nan_count, "NaN values encountered in", data["session"].iloc[z], "IMA",
                              data["ima"].iloc[z])
                    self.set_mem[path] = betas
                else:
                    betas = self.set_mem[path]
                chan_beta.append(betas)
            chan_beta = np.stack(chan_beta, axis=0)  # construct channel dimension of delays
            beta_coef.append(chan_beta)
        beta_coef = np.stack(beta_coef, axis=0)
        beta_coef = np.clip(beta_coef, a_min=-2.0, a_max=2.0)
        beta_coef = beta_coef * trajectory  # weight basis
        beta = beta_coef.sum(axis=0)
        sess = []
        # normalize by class and session (combined over trajectory)
        sess.append(sessions[0])
        mask = self.pad_to_cube(self.mask)
        # mask!!
        beta[:, np.logical_not(mask)] = 0.
        if np.sum(noise_frac_var) != 0:
            var = noise_frac_var  # each feature is scaled to have variance of 1
            noise = np.random.normal(0, var, beta.shape)
            beta = beta + noise

        if get_sess_data:
            if len(sess) > 1:
                print("WARNING, sessions returned when augmenting data via recombination is incomplete.")
            return beta, target, sess, chosen_index
        else:
            return beta, target, chosen_index

    def get_batch(self, bs, odset, resample=False, cube=True, start_from_class=0, randomize=True):
        beta_coef = []
        targets = []
        sess_data = []
        dset = copy.copy(odset)
        pre_target = np.arange(int(bs)) + start_from_class
        pre_target = pre_target % len(self.options)

        for j in range(int(bs)):
            if randomize:
                example = self.options[pre_target[j]]
                ind_override = None
            else:
                example = None
                ind_override = j
            if resample:
                if self.return_meta:
                    res = self.data_generator(dset, noise_frac_var=0.0, spatial_translation_var=0.0,
                                              max_base_examples=1, cube=cube, get_sess_data=True, example_class=example,
                                              index_override=ind_override)
                    if res == -1:
                        continue
                    beta, target, data, used_index = res
                    sess_data += data

                else:
                    res = self.data_generator(dset, noise_frac_var=0.0, spatial_translation_var=0.0,
                                              max_base_examples=1, cube=cube, example_class=example,
                                              index_override=ind_override)
                    if res == -1:
                        continue
                    beta, target, used_index = res
            else:
                # need to apply some degree of spline resampling
                if self.return_meta:
                    res = self.data_generator(dset, noise_frac_var=0.0, spatial_translation_var=0.00,
                                              max_base_examples=1, cube=cube, get_sess_data=True, example_class=example,
                                              index_override=ind_override)
                    if res == -1:
                        continue
                    beta, target, data, used_index = res
                    sess_data += data
                else:
                    res = self.data_generator(dset, noise_frac_var=0.0, spatial_translation_var=0.00,
                                              max_base_examples=1, cube=cube, example_class=example,
                                              index_override=ind_override)
                    if res == -1:
                        continue
                    beta, target, used_index = res
            beta_coef.append(beta)
            targets.append(target)
            if hasattr(self, "replace") and not self.replace:
                dset.drop(used_index, inplace=True)
                if len(dset) == 0:
                    # reset dataset
                    dset = copy.copy(odset)
        targets = np.array(targets, dtype=int)
        beta_coef = np.stack(beta_coef, axis=0)
        if self.return_meta:
            return beta_coef, targets, sess_data
        else:
            return beta_coef, targets

    def data_queuer(self, dset, bs, num_batches, resample, q, cube=True, start_from=0):
        cur_start = start_from
        while self._processed_ < num_batches:
            try:
                if self.return_meta:
                    beta_coef, targets, data = self.get_batch(bs, dset, resample=resample, cube=cube,
                                                              start_from_class=cur_start)
                    q.put((beta_coef, targets, data), timeout=500)
                else:
                    beta_coef, targets = self.get_batch(bs, dset, resample=resample, cube=cube,
                                                        start_from_class=cur_start)
                    q.put((beta_coef, targets), block=True, timeout=500)
                cur_start += 1
            except queue.Full:
                print("The data queue is full and does not appear to be emptying.")
                del beta_coef
                del targets
                exit(0)

    def batch_iterator(self, data_type="all", mode="train", fold=None, num_train_batches=1000,
                       resample=True, n_workers=16, cube=True, random=True):
        try:
            if fold is None or self.cv_folds <= 1 or mode == "all":
                dset = self.full_sets[data_type]
            elif mode == "train":
                dset = self.folded_sets[data_type].get_train(fold)
            elif mode == "test":
                dset = self.folded_sets[data_type].get_test(fold)
            else:
                raise ValueError
        except AttributeError:
            raise ValueError("This dataset was not defined.")
        batch_num = 0
        if not random:
            print("WARNING: Random is False. Returning fold in order, w/o sampling or balancing")
            n_workers = 1  # can't use mp if deterministic in current implimentation
            num_train_batches = (len(dset) // self.batch_size)

        bs = self.batch_size
        self._processed_ = 0
        context = mp.get_context("spawn")
        q = context.Queue(maxsize=self._max_q_size_)
        workers = []
        use_mp = n_workers > 1
        if use_mp:
            # Divide the dset into a chunk for each worker
            for i in range(n_workers):
                p = context.Process(target=self.data_queuer,
                                    args=(dset, bs, num_train_batches, resample, cube, i))
                p.start()
                workers.append(p)
        for i in range(num_train_batches):
            if use_mp:
                try:
                    res = q.get(block=True, timeout=200)
                except queue.Full:
                    print("The data queue is empty and does not appear to be populating.")
                    break
                if len(workers) == 0:
                    break
                self._processed_ += 1
                if self.return_meta:
                    beta_coef, targets, data = res
                else:
                    beta_coef, targets = res
            else:
                if not random:
                    if bs * batch_num > len(dset):
                        break
                    d = dset.iloc[bs * i:(i + 1) * bs]
                else:
                    d = dset
                if self.return_meta:
                    beta_coef, targets, data = self.get_batch(bs, d, resample=resample, cube=cube,
                                                              randomize=random)
                else:
                    beta_coef, targets = self.get_batch(bs, d, resample=resample, cube=cube,
                                                        randomize=random)
            if self.return_meta:
                yield beta_coef, targets, data
            else:
                yield beta_coef, targets
        if use_mp:
            print("killing...")
            try:
                q.get(block=False)
            except Exception as e:
                pass
            for i in range(n_workers):
                workers[i].terminate()
                workers[i].kill()
            q.close()
