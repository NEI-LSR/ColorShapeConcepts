import math

import matplotlib.pyplot as plt
import neurotools
import numpy as np
from neurotools import util, geometry
import torch

"""
Generate examples that simulate the type of fmri data we'd like to see.
Data is constructed via the following procedure.
- The data exists in <1, 16, 16, 16> space S. The last 3 of these dimensions are referred to as "spatial", and correspond to the axes of fmri data. 
- we select 2 regions r_1 and r_2, each of which size d and is a contiguous subset of S. 
    - r_1 only has signal when examples from type A are drawn. r_2 has signal when examples from either type A or B are drawn
- Cross-Signal: Each class is defined as a point on an ellipse in R2 and projected into r_1 and r_2 via a random orthonormal transform.
    - i.e looking in just one region with size d, each class center is a point on a unit circle on a 2d vector subspace of the d dimmensional ambient space.
    - This is the shared signal of A and B
    - random normal jitter is added, weighted by difficulty. 
- ID signal: The onehot encoding of each class is projected into r_1 and r_2, with a different basis for type A and type B examples. Magnitude weighterd by difficulty 
    - this represents high dimensional divergent signal for A and B
- Noise: random normal noise is added for each example, weighted by difficulty. 
- AdversarialNoise: noise that systematically biases us away from true class bounderies.
- MotionNoise: 
"""


def embed_unit_circles(n, k, d, generator=None, basis=None):
    """
    Helper function
    Embed n points along a unit circle on k parallel planes in a d-dimensional space.

    Parameters:
        n (int): Number of points to embed along a unit circle per plane.
        k (int): Number of parallel planes.
        d (int): Dimensionality of the ambient space.
        basis (ndarry, Optional): basis to use, otherwise use random

    Returns:
        np.ndarray: Array of shape (n * k, d) containing the embedded points.
    """
    if d < 2:
        raise ValueError("The dimensionality 'd' must be at least 2.")
    sd = 2
    if generator is None:
        generator = np.random.default_rng(np.random.randint(0, 100000))

    basis_dims = (d, sd)
    if basis is None:
        # Randomly choose a 2D subspace (basis vectors) # bias toward including all d in basis
        basis = generator.normal(size=basis_dims, loc=0, scale=1)
        basis, _ = np.linalg.qr(basis)  # Orthonormalize the basis

    # Generate unit circle points in the 2D subspace
    offset = generator.random() * 2 * np.pi
    angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
    angles = np.stack([angles + .2 * generator.random() + o * .05 for o in range(k)], axis=1)
    circle_points = np.stack((np.cos(angles), np.sin(angles)), axis=2)  # Shape (n, k, 2)

    # Embed circle points in the ambient d-dimensional space
    embedded_points = circle_points @ basis.T  # Shape (n, k, d)
    # Generate k parallel planes by translating the 2D subspace in a random direction
    translation_vector = generator.normal(size=d)
    translation_vector -= np.dot(translation_vector, basis[:, 0]) * basis[:, 0]
    translation_vector -= np.dot(translation_vector, basis[:, 1]) * basis[:, 1]

    # Normalize the translation vector
    translation_vector /= np.linalg.norm(translation_vector)
    translation_vector = translation_vector.reshape((1, 1, d))

    # Create offsets for the parallel planes
    offsets = .5 * np.linspace(-1, 1, k).reshape((1, -1, 1))  # Shape (1, k, 1)

    # Generate points on all planes
    plane_points = embedded_points + offsets * translation_vector

    # interleave parallel planes
    return plane_points.reshape((n, k, d))  # Shape (n, k, d)


def uniform_embedding(basis, n, d, generator: np.random.Generator):
    one_hot = np.eye(n)
    t_basis = generator.normal(size=(12, 12))
    t_basis, _ = np.linalg.qr(t_basis)
    basis = (basis @ t_basis).T
    one_hot = one_hot[generator.choice(np.arange(n), size=n, replace=False)]
    embed = one_hot @ basis
    return embed


class SimulationDataloader:

    def __init__(self, difficulty, seed, num_examples, batch_size, stable_seed=42,
                 add_group_signal=False):  # --- MODIFIED ---
        """

        Parameters
        ----------
        difficulty: how intense should noise and jitter b
        seed: seed to use for random noise
        num_examples: How many examples to generate with this seed (i.e. dataset size)
        batch_size: batch size for returned examples.
        stable_seed: stable seed for pattern generation (no generalizing across stable seeds)
        add_group_signal: bool, whether to add the group-wise signal
        """
        self.n_classes = 12
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.spatial = (16, 16, 16)
        self.difficulty = difficulty
        self.add_group_signal = add_group_signal  # --- NEW ---
        self.set_names = ["A", "B"]
        # origin of signal regions
        self.origins = [(2, 2, 6), (10, 10, 6)]
        # size of regions
        self.pat_size = (5, 5, 5)
        # A gets signal in both regions, B only in the second region
        self.signal_strength = {"A": [1., 1.],
                                "B": [0., 1.]}
        # need a generator that's stable across different resamples for generating patterns'
        self.stable_gen = np.random.default_rng(stable_seed)

        # need an unstable generator for drawing noise that should change across samples.
        self.gen = np.random.default_rng(seed)

        # create template signals
        self.joint_signal = []
        self.sep_signal = {"A": [],
                           "B": []}
        for _ in self.origins:
            basis = self.stable_gen.normal(size=(math.prod(self.pat_size), 12))
            basis, _ = np.linalg.qr(basis)
            signal = 1.0 * embed_unit_circles(n=self.n_classes // 2, k=2, d=math.prod(self.pat_size),
                                              generator=self.stable_gen, basis=basis[:, :2])
            signal = signal.reshape((self.n_classes,) + self.pat_size, order="F")
            self.joint_signal.append(signal)

            for k in self.sep_signal.keys():
                eb = 1. * uniform_embedding(basis, 12, math.prod(self.pat_size),
                                            generator=self.stable_gen).reshape((self.n_classes, ) + self.pat_size)
                self.sep_signal[k].append(.5 * self.difficulty * eb)


        # --- NEW: Define group signal ---
        # 0-based indices corresponding to the 1-based IDs provided
        self.group_indices = {
            "main_light": [0, 2, 4],  # 1-based [3, 7, 11]
            "main_dark": [1, 3, 5],  # 1-based [2, 6, 10]
            "other_light": [6, 8, 10],  # 1-based [1, 5, 9]
            "other_dark": [7, 9, 11]  # 1-based [4, 8, 12]
        }

        self.group_signal = []
        if self.add_group_signal:
            # Define different means for each group's signal
            group_means = {
                "main_light": 0.0,
                "main_dark": 0.0,
                "other_light": 0.0,
                "other_dark": 0.0
            }
            for _ in self.origins:
                # Create a template of shape (n_classes, *pat_size)
                g_signal = np.zeros((self.n_classes,) + self.pat_size)
                for group_name, indices in self.group_indices.items():
                    # Generate one random pattern for this group using the stable generator
                    # This pattern is drawn from a normal dist with a specific mean
                    group_pattern = self.stable_gen.normal(
                        loc=group_means[group_name],
                        scale=0.1 * self.difficulty,  # Scale signal std by difficulty
                        size=self.pat_size
                    )
                    # Assign this same pattern to all classes in the group
                    g_signal[indices] = group_pattern

                self.group_signal.append(g_signal)
        # --- END NEW ---

        # need to generate a pool of fixed examples for both set types to constrain dataset size.
        data = {}
        for cur_set in self.set_names:
            s_data = np.zeros((num_examples, self.n_classes) + self.spatial)
            # add combined joint and separable signal at correct spatial loctation in data
            for i, origin in enumerate(self.origins):
                indexer = np.s_[:, :, origin[0]:origin[0] + self.pat_size[0],
                          origin[1]:origin[1] + self.pat_size[1],
                          origin[2]:origin[2] + self.pat_size[2]]
                # add pattern at this origin weighted by strength
                s_data[indexer] += self.joint_signal[i][None, ...] * self.signal_strength[cur_set][i]
                s_data[indexer] += self.sep_signal[cur_set][i][None, ...] * self.signal_strength[cur_set][i]

                # --- NEW: Add group signal ---
                if self.add_group_signal:
                    # Add group signal, respecting the signal_strength mask
                    s_data[indexer] += self.group_signal[i][None, ...] * self.signal_strength[cur_set][i]
                # --- END NEW ---

            # add noise sampled for each exampls
            s_data += self.gen.normal(size=s_data.shape, scale=.5 * self.difficulty)
            data[cur_set] = s_data
        self.data = data

    def batch_iterator(self, use_set, epochs=1000):
        """
        Yield epochs batches of datas
        Parameters
        ----------
        use_set
        epochs

        Returns
        -------

        """
        s_data = self.data[use_set]
        for e in range(epochs):
            batch_inds = self.gen.integers(0, self.num_examples, size=self.batch_size)
            targets = self.gen.integers(0, self.n_classes, self.batch_size)
            sample = s_data[batch_inds, targets]
            sample = sample.reshape((self.batch_size, 1,) + self.spatial)  # add channel dim
            # normalize by example
            # s_m = np.mean(sample, axis=(2, 3, 4), keepdims=True)
            # s_std = np.std(sample, axis=(2, 3, 4), keepdims=True)
            # sample = (sample - s_m) / s_std
            yield sample, targets

    def plot_templates(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot()
        temp = np.zeros(self.spatial)
        for i, origin in enumerate(self.origins):
            indexer = np.s_[origin[0]:origin[0] + self.pat_size[0],
                      origin[1]:origin[1] + self.pat_size[1],
                      origin[2]:origin[2] + self.pat_size[2]]
            for cur_set in self.set_names:
                temp[indexer] += self.signal_strength[cur_set][i]
        ax.imshow(temp[:, :, 10])
        plt.show()

    def plot_circle_embedding(self):
        """
        Plots the modified 3D embedding as requested.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')

        # This line is from your original code.
        # It generates a (12, 3) array.
        # Due to order="f", the first 6 rows are the first points of the 6 pairs (Group 1)
        # and the second 6 rows are the second points (Group 2).
        test = embed_unit_circles(self.n_classes // 2, 2, 3).reshape((-1, 3), order="f")

        # --- MODIFICATIONS START ---

        # 1. Separate the two groups of points
        points_g1 = test[:6]
        points_g2 = test[6:]

        # 2. Define colors and size
        color_g1 = 'lightgrey'
        color_g2 = 'black'
        # Default scatter size (s) is roughly 36. We'll double it.
        point_size = 72

        # 3. Plot the scatter points with new colors and size
        ax.scatter(points_g1[:, 0], points_g1[:, 1], points_g1[:, 2],
                   color=color_g1, s=point_size, label="Group 1")

        # Note: You mentioned "second 5 black", but the data structure and
        # "second 6" imply 6 points. I've plotted all 6 points in black.
        ax.scatter(points_g2[:, 0], points_g2[:, 1], points_g2[:, 2],
                   color=color_g2, s=point_size, label="Group 2")

        # 4. Plot lines to connect points within each group
        # We add the first point to the end to close the loop
        loop_g1 = np.vstack([points_g1, points_g1[0]])
        loop_g2 = np.vstack([points_g2, points_g2[0]])

        ax.plot(loop_g1[:, 0], loop_g1[:, 1], loop_g1[:, 2], color=color_g1)
        ax.plot(loop_g2[:, 0], loop_g2[:, 1], loop_g2[:, 2], color=color_g2)

        # 5. Eliminate the default grey grid and background panes
        ax.grid(False)

        # Make the panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Make the pane edges white (invisible on a white figure background)
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # --- MODIFICATIONS END ---

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.legend()
        return fig, ax