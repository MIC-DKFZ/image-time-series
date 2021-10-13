from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from datetime import time
import itertools
import json
import numpy as np
import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib as mp
from typing import Optional, Union, List, Dict, Tuple, Iterable
from math import factorial, comb

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import (
    AbstractTransform,
    SpatialTransform_2,
    Compose,
    MirrorTransform,
    GammaTransform,
    GaussianNoiseTransform,
    GaussianBlurTransform,
    BrightnessMultiplicativeTransform,
)
from batchgenerators.transforms.spatial_transforms import Rot90Transform

from gliomagrowth.util.util import (
    modify_bbox,
    transformable_to_ct,
    ct_to_transformable,
    NpzLazyDict,
    Node,
)
from gliomagrowth.util.lightning import str2bool

# set a default data_dir for convenience
data_dir = "/media/jens/SSD/bovarec/multi"
file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_dir, data_dir)

# dirty, dirty hack
class MultiThreadedAugmenter(MultiThreadedAugmenter):
    def __len__(self):
        return len(self.generator)


def split(
    N: int = 5, seed: int = 1, ddir: Optional[str] = None, use_defaults: bool = True
) -> List[List[str]]:
    """Split dataset into N parts of equal size.

    When use_defaults is active, we expect a default_splits.json file in the data
    directory. Otherwise we take all .npy/.npz files as instances. Either way, there
    needs to be a .npy/.npz file for each identifier.

    Args:
        N: Split into this many parts. If division is not possible, last element
            collects the remaining instances
        seed: Seed the random split. Will create a separate RandomState.
        ddir: Optional data directory.
            Will default to gliomagrowth.data.glioma.data_dir.

    Returns:

        List of N elements, where each is a list of identifiers.

    """

    if ddir is None:
        ddir = data_dir

    # default splits
    if use_defaults:
        splits = json.load(open(os.path.join(ddir, "default_splits.json")))

    # custom splits
    else:
        all_ = all_subjects(ddir)
        r = np.random.RandomState(seed)
        r.shuffle(all_)
        num = len(all_) // N
        splits = []
        for i in range(N):
            if i < (N - 1):
                splits.append(sorted(all_[i * num : (i + 1) * num]))
            else:
                splits.append(sorted(all_[i * num :]))

    return splits


def load(
    mmap_mode: Optional[str] = None,
    subjects: Union[str, Iterable[str]] = "all",
    dtype: type = np.float32,
    npz: bool = False,
    ddir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load data.

    Args:
        mmap_mode: Should be either None (for in-memory array) or one of the
            numpy.memmap read-only modes.
        subjects: 'all' or iterable of subject identifiers.
        dtype: Data type for numpy arrays
        npz: Indicates that files are .npz instead of .npy
        ddir: Optional data directory.
            Will default to gliomagrowth.data.glioma.data_dir.

    Returns:
        A dict of subject identifiers and 5D np.arrays (time, channel, X, Y, Z).

    """

    if ddir is None:
        ddir = data_dir

    if subjects == "all":
        subjects = all_subjects(ddir)
    assert hasattr(subjects, "__iter__")
    assert type(dtype) == type

    if npz:
        data = NpzLazyDict()
    else:
        data = dict()

    for subject in subjects:
        if npz:
            fname = "{}.npz".format(subject)
            data[subject] = os.path.join(ddir, fname)
        else:
            fname = "{}.npy".format(subject)
            data[subject] = np.load(
                os.path.join(ddir, fname), mmap_mode=mmap_mode
            ).astype(dtype, copy=False)

    return data


def all_subjects(ddir: Optional[str] = None) -> List[str]:
    """Get list of all available identifiers.

    Args:
        ddir: Optional data directory.
            Will default to gliomagrowth.data.glioma.data_dir.

    Returns:
        A list of subject identifiers.

    """
    if ddir is None:
        ddir = data_dir
    all_ = os.listdir(ddir)
    subjects = sorted(filter(lambda x: x.endswith((".npy", ".npz")), all_))
    return list(map(lambda x: x.replace(".npy", "").replace(".npz", ""), subjects))


def load_days_and_associations(
    ddir: Optional[str] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    """Load scan date information and subject association to centers.

    Expects a multi_days.csv file in the data directory.

    Args:
        ddir: Optional data directory.
            Will default to gliomagrowth.data.glioma.data_dir.

    Returns:
        Dictionaries for scan dates (days starting from 0) and scan center associations.

    """

    if ddir is None:
        ddir = data_dir

    days = dict()
    associations = dict()

    with open(os.path.join(ddir, "multi_days.csv")) as info_file:
        reader = csv.reader(info_file)
        for line in reader:
            assoc, subject = line[0].split("_")
            associations[line[0]] = assoc
            days[line[0]] = []
            for day in line[1:]:
                if day != "":
                    days[line[0]].append(int(day))

    return days, associations


def all_subject_associations(ddir: Optional[str] = None) -> Dict[str, str]:
    """See load_days_and_associations."""
    return load_days_and_associations(ddir)[1]


def all_scan_days(ddir: Optional[str] = None) -> Dict[str, List[int]]:
    """See load_days_and_associations."""
    return load_days_and_associations(ddir)[0]


class PatientNode(Node):
    """A special leaf node of a tree that can iterate over different slices for a
    given patient. Indexing this node returns a tuple if slice objects that can directly
    be used to access data from its patient. Works with both 2D and 3D.

    Args:
        ct: A tuple of context and target set sizes.
        subject_id: Subject identifier string.
        time_size: Size of the time axis for this subject.
        forward_only: If active, target points will always come after context points.
        axis: If provided, the node will also iterate over spatial slices along this
            axis.
        first_slice: First slice along the provided axis.
        last_slice: Last slice along the provided axis. Inclusive!

    """

    def __init__(
        self,
        ct: Tuple[int, int],
        subject_id: str,
        time_size: int,
        forward_only: bool = False,
        axis: Optional[int] = None,
        first_slice: Optional[int] = None,
        last_slice: Optional[int] = None,
    ):

        super().__init__()

        self.ct = ct
        self.subject_id = subject_id
        self.time_size = time_size
        self.forward_only = forward_only
        self.axis = axis
        self.first_slice = first_slice
        self.last_slice = last_slice

    @property
    def is_2d(self) -> bool:
        """Checks if all attributes we need for 2D are there."""

        return (
            self.axis is not None
            and self.first_slice is not None
            and self.last_slice is not None
        )

    def replace(self, node: Node):

        self.ct = node.ct
        self.subject_id = node.subject_id
        self.time_size = node.time_size
        self.forward_only = node.forward_only
        self.axis = node.axis
        self.first_slice = node.first_slice
        self.last_slice = node.last_slice
        self.data = node.data
        self._children = node._children

    def __len__(self) -> int:

        # how many ways are there to draw context and target out of time_size elements?
        c, t = self.ct

        # in forward-only mode, the order is fixed (target comes after context),
        # so it's just about how many subsets we can draw where the elements are
        # indistinguishable.
        possibilities = comb(self.time_size, c + t)

        # if not in forward-only mode, we can again draw t elements out of the subset
        if not self.forward_only:
            possibilities *= comb(c + t, t)

        # if we work in 2D, we can do this for every slice!
        if self.is_2d:
            possibilities *= self.last_slice - self.first_slice + 1

        return possibilities

    def __getitem__(self, index: int) -> Tuple[str, tuple, tuple]:

        if not isinstance(index, int):
            raise ValueError("__getitem__ is only implemented for integers!")

        len_self = len(self)
        c, t = self.ct

        # check that index works
        if index < 0:
            index = len_self + index
        if index < 0 or index > len_self - 1:
            raise IndexError("list index out of range")

        # number of possible spatial slices
        if self.is_2d:
            num_slices = self.last_slice - self.first_slice + 1
        else:
            num_slices = 1

        # number of ways to draw context + target indices
        num_draws = comb(self.time_size, c + t)

        # number of ways to assign a draw to context and target
        if self.forward_only:
            num_permutations = 1
        else:
            num_permutations = comb(c + t, t)

        # now imagine an array of shape (num_slices, num_draws, num_permutations).
        # we convert the input index to an array index
        slice_index, draw_index, perm_index = np.unravel_index(
            index, (num_slices, num_draws, num_permutations)
        )
        time_indices = list(itertools.combinations(np.arange(self.time_size), c + t))[
            draw_index
        ]
        if self.forward_only:
            context_indices, target_indices = time_indices[:c], time_indices[-t:]
        else:
            context_indices = list(itertools.combinations(time_indices, c))[perm_index]
            target_indices = np.array(
                [idx for idx in time_indices if idx not in context_indices]
            )

        context_slc = [slice(None)] * 5
        target_slc = [slice(None)] * 5
        context_slc[0] = tuple(context_indices)
        target_slc[0] = tuple(target_indices)
        if self.is_2d:
            context_slc[self.axis + 2] = slice_index
            target_slc[self.axis + 2] = slice_index

        return self.subject_id, tuple(context_slc), tuple(target_slc)

    def __repr__(self) -> str:

        info_str = "({},{})-{}-t{}".format(
            self.ct[0], self.ct[1], self.subject_id, self.time_size
        )
        if self.forward_only:
            info_str += "forward"
        if self.is_2d:
            info_str += "-2D"
            info_str += "-ax{}".format(self.axis)
            info_str += "-{}:{}".format(self.first_slice, self.last_slice)
        else:
            info_str += "-3D"
        return info_str


class FutureContextGenerator2D(SlimDataLoaderBase):
    """DataLoader for 2D data.

    Will generate context/target pairs of data, essentially like input/GT,
    but each comes with the corresponding segmentation.

    Args:
        data: The data to generate patches from. Use load() to construct.
        batch_size: The batch size :)
        context_size: Number of context points. Either a fixed number or bounds for
            random draws. For the latter, the upper limit is EXCLUSIVE like in
            np.random.randint.
        target_size: Number of target points. Either a fixed number or bounds for
            random draws. For the latter, the upper limit is EXCLUSIVE like in
            np.random.randint.
        forward_only: If this is active, the target points will strictly be in the
            future.
        context_larger_than_target: If this is active, there will always be more context
            points than target points
        axis: Extract slices along this axis. Note that extraction along axes other than
            0 can be slow and it might be worth saving the data in a different
            orientation instead.
        patch_size: Target patch size (spatial dimension). Only sqquare patches
            supported at the moment.
        infinite: Restart generator instead of raising StopIteration. Note that
            depending on the batch size, not all data will be shown.
        subject_associations: Supplies meta info that will be included in the batch
            dictionaries.
        scan_days: Supplies meta info that will be included in the batch
            dictionaries.
        only_tumor: Only generate patches that contain tumor. Requires a
            multi_tumor_crop.json file in the data directory that contains the bounding
            boxes of the tumor for each case.
        whole_tumor: Merge all foreground labels (except the dropped ones).
        dtype_seg: Segmentation data type.
        ddir: Optional data directory.
            Will default to gliomagrowth.data.glioma.data_dir.
        merge_context_target: Stack context and target into a single array.
            Activate this if you want to use data augmentation!
        drop_labels: Move these labels to the background.
        normalize_date_factor: Multiply date values by this.

    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        batch_size: int,
        context_size: Union[int, Iterable[int]],
        target_size: Union[int, Iterable[int]],
        forward_only: bool = False,
        context_larger_than_target: bool = True,
        axis: int = 0,
        patch_size: int = 64,
        infinite: bool = False,
        subject_associations: Optional[Dict[str, str]] = None,
        scan_days: Optional[Dict[str, List[int]]] = None,
        only_tumor: bool = True,
        whole_tumor: bool = False,
        dtype_seg: type = np.int64,
        ddir: Optional[str] = None,
        merge_context_target: bool = False,
        drop_labels: Optional[Union[int, Iterable[int]]] = 3,
        normalize_date_factor: float = 0.01,
        **kwargs,
    ):

        number_of_threads_in_multithreaded = kwargs.get(
            "number_of_threads_in_multithreaded", None
        )
        super().__init__(data, batch_size, number_of_threads_in_multithreaded)

        if ddir is None:
            ddir = data_dir
        self.ddir = ddir

        self.context_size = context_size
        self.target_size = target_size
        self.forward_only = forward_only
        self.context_larger_than_target = context_larger_than_target
        self.axis = axis
        self.patch_size = patch_size
        self.infinite = infinite
        if subject_associations is None:
            self._all_subject_associations = all_subject_associations(ddir)
        else:
            self._all_subject_associations = subject_associations
        if scan_days is None:
            self._all_scan_days = all_scan_days(ddir)
        else:
            self._all_scan_days = scan_days
        self.only_tumor = only_tumor
        self.whole_tumor = whole_tumor
        self.dtype_seg = dtype_seg
        self.merge_context_target = merge_context_target
        if type(drop_labels) == int:
            self.drop_labels = (drop_labels,)
        else:
            self.drop_labels = drop_labels
        self.normalize_date_factor = normalize_date_factor

        self.current_position = 0
        self.was_initialized = False
        if self.number_of_threads_in_multithreaded is None:
            self.number_of_threads_in_multithreaded = 1

        self.data_order = np.arange(len(self))

    @property
    def possible_sets(self) -> List[Tuple[str, Tuple]]:
        """Construct a list of all possible (subject, slice) pairs that are allowed
        for the configuration of the generator.
        """

        try:
            return self._possible_sets
        except AttributeError:

            if self.only_tumor:
                with open(os.path.join(self.ddir, "multi_tumor_crop.json"), "r") as f:
                    tumor_crops = json.load(f)

            if hasattr(self.context_size, "__iter__"):
                context_size = list(range(*self.context_size))
            else:
                context_size = [self.context_size]
            if hasattr(self.target_size, "__iter__"):
                target_size = list(range(*self.target_size))
            else:
                target_size = [self.target_size]
            context_target_combinations = list(
                itertools.product(context_size, target_size)
            )
            if self.context_larger_than_target:
                context_target_combinations = [
                    (c, t) for (c, t) in context_target_combinations if c > t
                ]

            # we each context, target combination, we save all the possible slices
            possible_sets = {ct: [] for ct in context_target_combinations}

            # we collect the possible sets for all combinations of context and target
            for subject in sorted(self._data.keys()):

                # construct the bounds of the slice range
                if self.only_tumor:
                    possible_slices = tumor_crops[subject]
                    possible_slices = (
                        possible_slices[0][self.axis],
                        possible_slices[1][self.axis] + 1,
                    )
                else:
                    possible_slices = (
                        0,
                        self._data[subject].shape[self.axis + 2],
                    )

                current_timesteps = self._data[subject].shape[0]
                for c, t in possible_sets.keys():
                    pass

                slc = [slice(None)] * 5

                # limit patches to tumor region
                if self.patch_size is not None:
                    if self.only_tumor:
                        tumor_bbox = tumor_crops[subject]
                    else:
                        tumor_bbox = [[], []]
                        for i in range(3):
                            tumor_bbox[0].append(0)
                            tumor_bbox[1].append(self._data[subject].shape[i + 2])
                    tumor_bbox = modify_bbox(
                        *tumor_bbox,
                        self.patch_size,
                        self._data[subject].shape[2:],
                        skip_axes=self.axis,
                    )
                    for ax in range(3):
                        if ax == self.axis:
                            continue
                        slc[ax + 2] = slice(tumor_bbox[0][ax], tumor_bbox[1][ax] + 1)

                for c, t in possible_sets.keys():

                    if c + t > current_timesteps:
                        continue

                    for i in range(possible_slices[0], possible_slices[1] + 1):

                        slc[self.axis + 2] = i

                        # we now draw all possible indices of context and target
                        for index_subset in itertools.combinations(indices, c + t):
                            # now we have a subset of indices that we need to assign to
                            # context and target.
                            # In forward-only mode, there is only one possibility
                            if self.forward_only:
                                slc[0] = tuple(index_subset[:c])
                                context_slc = tuple(slc)
                                slc[0] = tuple(index_subset[-t:])
                                target_slc = tuple(slc)
                                possible_sets[(c, t)].append(
                                    (subject, context_slc, target_slc)
                                )
                            else:
                                for t_idx in itertools.combinations(index_subset, t):
                                    slc[0] = t_idx
                                    target_slc = tuple(slc)
                                    c_idx = tuple(
                                        [
                                            idx
                                            for idx in index_subset
                                            if idx not in t_idx
                                        ]
                                    )
                                    slc[0] = c_idx
                                    context_slc = tuple(slc)
                                    possible_sets[(c, t)].append(
                                        (subject, context_slc, target_slc)
                                    )

            self._possible_sets = possible_sets
            return possible_sets

    def __len__(self):

        return sum(map(len, self.possible_sets)) // self.batch_size

    def reset(self):
        """Resets the generator. Called automatically when infinite=True."""

        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True

    def generate_train_batch(self) -> Union[Tuple[Dict, Dict], Dict]:
        """This is called internally when you do next() on the generator."""

        if not self.was_initialized:
            self.reset()
        if self.current_position >= len(self.possible_sets):
            self.reset()
            if not self.infinite:
                raise StopIteration
        context, target = self.make_batch(self.current_position)
        self.current_position += (
            self.number_of_threads_in_multithreaded * self.batch_size
        )

        if not self.merge_context_target:
            return context, target
        else:
            return ct_to_transformable(context, target)

    def make_batch(self, idx: int) -> Tuple[Dict, Dict]:
        """Construct a batch from the specified position.

        Args:
            idx: Current position in self.data_order.

        Returns:
            A pair of context/target batch dictionaries.

        """

        context_data = []
        context_seg = []
        context_days = []
        target_data = []
        target_seg = []
        target_days = []

        if self.axis is not None:
            slices = []
        subjects = []
        timesteps = []
        subject_associations = []

        while len(context_data) < self.batch_size:

            idx = idx % len(self.data_order)

            subject, slc = self.possible_sets[self.data_order[idx]]

            if len(context_data) > 0:
                current_timesteps = slc[0].stop - slc[0].start
                if current_timesteps != context_data[-1].shape[0]:
                    idx += 1
                    continue

            if self.axis is not None:
                slices.append(slc[self.axis + 2])
            subjects.append(subject)
            timesteps.append(slc[0].start)
            subject_associations.append(self._all_subject_associations[subject])

            # if you want to change how context and target are constructed (e.g. have
            # more than 1 target timestep), this is the place.
            context_days.append(self._all_scan_days[subject][slc[0]])
            target_days.append(
                self._all_scan_days[subject][slc[0].stop : slc[0].stop + 1]
            )

            context_data.append(self._data[subject][slc][:, :-1])
            context_seg.append(self._data[subject][slc][:, -1:])
            slc = list(slc)
            slc[0] = slice(slc[0].stop, slc[0].stop + 1)
            slc = tuple(slc)
            target_data.append(self._data[subject][slc][:, :-1])
            target_seg.append(self._data[subject][slc][:, -1:])

            idx += 1

        context = dict(
            data=np.stack(context_data),
            seg=np.stack(context_seg).astype(self.dtype_seg),
            scan_days=np.stack(context_days)[:, :, None].astype(np.float32),
            timesteps=np.array(timesteps),
            subjects=np.array(subjects),
            subject_associations=np.array(subject_associations),
        )
        if self.axis is not None:
            context["slices"] = np.array(slices)
        target = dict(
            data=np.stack(target_data),
            seg=np.stack(target_seg).astype(self.dtype_seg),
            scan_days=np.stack(target_days)[:, :, None].astype(np.float32),
        )

        context["scan_days"] *= self.normalize_date_factor
        target["scan_days"] *= self.normalize_date_factor

        if self.drop_labels is not None:
            for l in self.drop_labels:
                context["seg"][context["seg"] == l] = 0
                target["seg"][target["seg"] == l] = 0

        if self.whole_tumor:
            context["seg"][context["seg"] > 0] = 1
            target["seg"][target["seg"] > 0] = 1

        return context, target


class FutureContextGenerator3D(FutureContextGenerator2D):
    """Same as FutureContextGenerator2D, but in 3D."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.axis = None

    @property
    def possible_sets(self) -> List[Tuple[str, Tuple]]:
        """Construct a list of all possible (subject, slice) pairs that are allowed
        for the configuration of the generator.
        """

        try:
            return self._possible_sets
        except AttributeError:

            if self.only_tumor:
                with open(os.path.join(self.ddir, "multi_tumor_crop.json"), "r") as f:
                    tumor_crops = json.load(f)

            sets = []
            for time_size in self.time_size:
                for subject in sorted(self._data.keys()):
                    current_timesteps = self._data[subject].shape[0]
                    if current_timesteps <= time_size:
                        continue
                    for t in range(current_timesteps - time_size):
                        slc = [
                            slice(None),
                        ] * 5
                        slc[0] = slice(t, t + time_size)
                        if self.patch_size is not None:
                            if self.only_tumor:
                                tumor_bbox = tumor_crops[subject]
                            else:
                                tumor_bbox = [[], []]
                                for i in range(3):
                                    tumor_bbox[0].append(0)
                                    tumor_bbox[1].append(
                                        self._data[subject].shape[i + 2]
                                    )
                            tumor_bbox = modify_bbox(
                                *tumor_bbox,
                                self.patch_size,
                                self._data[subject].shape[2:],
                            )
                            for ax in range(3):
                                slc[ax + 2] = slice(
                                    tumor_bbox[0][ax], tumor_bbox[1][ax] + 1
                                )
                        sets.append((subject, tuple(slc)))

            self._possible_sets = sets
            return sets


class RandomFutureContextGenerator2D(FutureContextGenerator2D):
    """Same as FutureContextGenerator2D, but shuffles data.

    Args:
        random_date_shift: Randomly shift dates (drawn uniformly). This is applied
            AFTER normalize_date_factor!
        random_rotation: Random 90 degree rotations. You can also do this with data
            augmentation, but if you only want these rotations, it's more convenient
            to do internally.

    """

    def __init__(
        self,
        *args,
        random_date_shift: Iterable[float] = (-1.0, 1.0),
        random_rotation: bool = True,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.random_date_shift = random_date_shift
        self.random_rotation = random_rotation
        self.num_restarted = 0

    def reset(self):
        """Resets the generator. Called automatically when infinite=True."""

        super().reset()
        self.rs = np.random.RandomState(
            self.thread_id
            + self.num_restarted * self.number_of_threads_in_multithreaded
        )
        self.rs.shuffle(self.data_order)
        self.num_restarted = self.num_restarted + 1

    def generate_train_batch(self) -> Union[Tuple[Dict, Dict], Dict]:
        """This is called internally when you do next() on the generator."""

        if not self.was_initialized:
            self.reset()
        if self.current_position >= len(self.possible_sets):
            self.reset()
            if not self.infinite:
                raise StopIteration
        context, target = self.make_batch(self.current_position)
        self.current_position += (
            self.number_of_threads_in_multithreaded * self.batch_size
        )

        if self.random_date_shift is not None:
            shift = self.rs.uniform(
                *self.random_date_shift, size=(context["scan_days"].shape[0], 1, 1)
            )
            context["scan_days"] = context["scan_days"] + shift
            target["scan_days"] = target["scan_days"] + shift

        if self.random_rotation:
            axes = (2, 3)
            rot = self.rs.randint(0, 4, size=(context["data"].shape[0],))
            for r, num_rot in enumerate(rot):
                context["data"][r] = np.rot90(context["data"][r], k=num_rot, axes=axes)
                context["seg"][r] = np.rot90(context["seg"][r], k=num_rot, axes=axes)
                target["data"][r] = np.rot90(target["data"][r], k=num_rot, axes=axes)
                target["seg"][r] = np.rot90(target["seg"][r], k=num_rot, axes=axes)

        if not self.merge_context_target:
            return context, target
        else:
            return ct_to_transformable(context, target)


class RandomFutureContextGenerator3D(FutureContextGenerator3D):
    """Same as FutureContextGenerator3D, but shuffles data.

    Args:
        random_date_shift: Randomly shift dates (drawn uniformly). This is applied
            AFTER normalize_date_factor!
        random_rotation: Random 90 degree rotations. You can also do this with data
            augmentation, but if you only want these rotations, it's more convenient
            to do internally.

    """

    def __init__(
        self,
        *args,
        random_date_shift: Iterable[float] = (-1.0, 1.0),
        random_rotation: bool = True,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.random_date_shift = random_date_shift
        self.random_rotation = random_rotation
        self.num_restarted = 0

    def reset(self):
        """Resets the generator. Called automatically when infinite=True."""

        super().reset()
        self.rs = np.random.RandomState(
            self.thread_id
            + self.num_restarted * self.number_of_threads_in_multithreaded
        )
        self.rs.shuffle(self.data_order)
        self.num_restarted = self.num_restarted + 1

    def generate_train_batch(self) -> Union[Tuple[Dict, Dict], Dict]:
        """This is called internally when you do next() on the generator."""

        if not self.was_initialized:
            self.reset()
        if self.current_position >= len(self.possible_sets):
            self.reset()
            if not self.infinite:
                raise StopIteration
        context, target = self.make_batch(self.current_position)
        self.current_position += (
            self.number_of_threads_in_multithreaded * self.batch_size
        )

        if self.random_date_shift is not None:
            shift = self.rs.uniform(
                *self.random_date_shift, size=(context["scan_days"].shape[0], 1, 1)
            )
            context["scan_days"] = context["scan_days"] + shift
            target["scan_days"] = target["scan_days"] + shift

        if self.random_rotation:
            for axes in ((2, 3), (2, 4), (3, 4)):
                rot = self.rs.randint(0, 4, size=(context["data"].shape[0],))
                for r, num_rot in enumerate(rot):
                    context["data"][r] = np.rot90(
                        context["data"][r], k=num_rot, axes=axes
                    )
                    context["seg"][r] = np.rot90(
                        context["seg"][r], k=num_rot, axes=axes
                    )
                    target["data"][r] = np.rot90(
                        target["data"][r], k=num_rot, axes=axes
                    )
                    target["seg"][r] = np.rot90(target["seg"][r], k=num_rot, axes=axes)

        if not self.merge_context_target:
            return context, target
        else:
            return ct_to_transformable(context, target)


def get_train_transforms(
    patch_size: int = 64,
    dim: int = 2,
    axis: int = 0,
    transform_spatial: bool = True,
    transform_elastic: bool = False,
    transform_mirror: bool = True,
    transform_brightness: bool = True,
    transform_gamma: bool = True,
    transform_noise: bool = True,
    transform_blur: bool = True,
    p_per_sample: float = 0.15,
) -> AbstractTransform:
    """Convenience function to construct data augmentation transforms that work well.

    Args:
        patch_size: Desired patch size. Only square patches supported at the moment.
        dim: Should be 2 or 3.
        axis: Extract 2D slices along this axis. Will be ignored in 3D.
        transform_spatial: Rotation and scaling.
        transform_elastic: Elastic deformation.
        transform_mirror: Random mirroring.
        transform_brightness: Brightness transformation.
        transform_gamma: Random gamma curve variations.
        transform_noise: Add Gaussian noise.
        transform_blur: Random blurring.
        p_per_sample: Probability for individual transforms.

    Returns:
        A composition of the chosen transforms.

    """

    if dim == 3:
        axis = -1

    transforms = []

    if transform_spatial:
        transforms.append(
            SpatialTransform_2(
                patch_size=[
                    patch_size,
                ]
                * dim,
                patch_center_dist_from_border=[
                    patch_size // 2,
                ]
                * dim,
                do_elastic_deform=transform_elastic,
                deformation_scale=(0, 0.25),
                do_rotation=True,
                angle_x=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
                angle_y=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
                angle_z=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
                do_scale=True,
                scale=(0.75, 1.25),
                border_mode_data="constant",
                border_cval_data=0,
                border_mode_seg="constant",
                border_cval_seg=0,
                order_seg=1,
                order_data=3,
                random_crop=True,
                p_el_per_sample=p_per_sample,
                p_rot_per_sample=p_per_sample,
                p_scale_per_sample=p_per_sample,
            )
        )

    if transform_mirror:
        transforms.append(MirrorTransform(axes=[i for i in range(3) if i != axis]))

    if transform_brightness:
        transforms.append(
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.7, 1.5), per_channel=True, p_per_sample=p_per_sample
            )
        )

    if transform_gamma:
        transforms.append(
            GammaTransform(
                gamma_range=(0.5, 2),
                invert_image=False,
                per_channel=True,
                p_per_sample=p_per_sample,
            )
        )

    if transform_blur:
        transforms.append(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.5),
                different_sigma_per_channel=True,
                p_per_channel=0.5,
                p_per_sample=p_per_sample,
            )
        )

    if transform_noise:
        transforms.append(
            GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=p_per_sample)
        )

    transforms = Compose(transforms)

    return transforms


class GliomaModule(pl.LightningDataModule):
    """LightningDataModule that wraps data generator and augmentation.

    Args:
        data_dir: The data directory. Data will only be loaded when generators are
            created.
        batch_size: The batch size :)
        time_size: Number of consecutive timesteps that make up context/target pairs.
            Target will always be the last timestep, context all but the last. Other
            configurations are not supported atm, but should be easy to implement.
            This can also be an iterable of multiple options. For example, if you want
            to predict a future timestep from between 2 and 4 inputs, this should be
            [3, 4, 5].
        dim: Should be 2 or 3.
        axis: Extract slices along this axis. Note that extraction along axes other than
            0 can be slow and it might be worth saving the data in a different
            orientation instead.
        split_N: Split data into this many parts. split_val and split_test select the
            parts that will be used as the respective sets.
        seed: Random seed.
        split_val: Index for validation data.
        split_test: Index for test data.
        patch_size: Target patch size (spatial dimension). Only sqquare patches
            supported at the moment.
        only_tumor: Only generate patches that contain tumor. Requires a
            multi_tumor_crop.json file in the data directory that contains the bounding
            boxes of the tumor for each case.
        whole_tumor: Merge all positive labels into one.
        drop_labels: Set these labels to background.
        transform_spatial: Rotation and scaling.
        transform_elastic: Elastic deformation.
        transform_rot90: Random 90 degree rotations.
        transform_mirror: Random mirroring.
        transform_brightness: Brightness transformation.
        transform_gamma: Random gamma curve variations.
        transform_noise: Add Gaussian noise.
        transform_blur: Random blurring.
        random_date_shift: Randomly shift dates in this range. This is applied AFTER
            normalize_date_factor!
        normalize_date_factor: Multiply date values by this to
            adjust value range. Original units are days, so we often get values >100.
        n_processes: Use this many processes in parallel for data augmentation.
        validate_random: Use RandomFutureContextGenerator2D instead of
            FutureContextGenerator2D for validation.

    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        time_size: Iterable[int] = (2, 3, 4, 5),
        dim: int = 2,
        axis: int = 0,
        split_N: int = 5,
        seed: int = 1,
        split_val: int = 3,
        split_test: int = 4,
        patch_size: int = 64,
        only_tumor: bool = True,
        whole_tumor: bool = False,
        drop_labels: Optional[Union[int, Iterable[int]]] = 3,
        transform_spatial: bool = True,
        transform_elastic: bool = False,
        transform_rot90: bool = True,
        transform_mirror: bool = True,
        transform_gamma: bool = True,
        transform_noise: bool = True,
        transform_blur: bool = True,
        transform_brightness: bool = True,
        random_date_shift: Iterable[float] = (-1.0, 1.0),
        normalize_date_factor: float = 0.01,
        n_processes: int = 8,
        validate_random: bool = True,
        **kwargs,
    ):
        super().__init__(dims=dim)

        # if dim != 2:
        #     raise NotImplementedError(
        #         "At the moment, only two-dimensional dataloading is implemented!"
        #     )
        if dim not in (2, 3):
            raise ValueError("'dim' mist be 2 or 3, but is {}.".format(dim))

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_size = time_size
        self.dim = dim
        self.axis = axis
        self.split_N = split_N
        self.seed = seed
        self.split_val = split_val
        self.split_test = split_test
        self.split_train = list(range(split_N))
        self.split_train.remove(self.split_test)
        if split_val != split_test:
            self.split_train.remove(self.split_val)
        self.patch_size = patch_size
        self.only_tumor = only_tumor
        self.whole_tumor = whole_tumor
        self.drop_labels = drop_labels
        self.transform_spatial = transform_spatial
        self.transform_elastic = transform_elastic
        self.transform_rot90 = transform_rot90
        self.transform_mirror = transform_mirror
        self.transform_gamma = transform_gamma
        self.transform_noise = transform_noise
        self.transform_blur = transform_blur
        self.transform_brightness = transform_brightness
        self.random_date_shift = random_date_shift
        self.normalize_date_factor = normalize_date_factor
        self.n_processes = n_processes
        self.validate_random = validate_random

    def setup(self, stage: Optional[str] = None):
        """Initialize data split (without actually loading data).

        Args:
            stage: ignored.

        """

        spl = split(self.split_N, self.seed, self.data_dir)
        self.subjects_val = sorted(spl[self.split_val])
        self.subjects_test = sorted(spl[self.split_test])
        self.subjects_train = []
        for s in self.split_train:
            self.subjects_train = self.subjects_train + spl[s]
        self.subjects_train = sorted(self.subjects_train)

    def train_dataloader(
        self,
    ) -> Union[RandomFutureContextGenerator2D, RandomFutureContextGenerator3D]:
        """Construct training dataloader."""

        if self.dim == 2:
            gen = RandomFutureContextGenerator2D
        else:
            gen = RandomFutureContextGenerator3D

        train_data = load("r", subjects=self.subjects_train, ddir=self.data_dir)
        train_gen = gen(
            data=train_data,
            batch_size=self.batch_size,
            time_size=self.time_size,
            axis=self.axis,
            patch_size=int(1.5 * self.patch_size),  # for SpatialTransform rotations
            only_tumor=self.only_tumor,
            whole_tumor=self.whole_tumor,
            ddir=self.data_dir,
            random_date_shift=self.random_date_shift,
            random_rotation=self.transform_rot90,
            number_of_threads_in_multithreaded=self.n_processes,
            merge_context_target=True,
            drop_labels=self.drop_labels,
        )

        transforms = get_train_transforms(
            patch_size=self.patch_size,
            dim=self.dim,
            axis=self.axis,
            transform_spatial=self.transform_spatial,
            transform_elastic=self.transform_elastic,
            transform_mirror=self.transform_mirror,
            transform_brightness=self.transform_brightness,
            transform_gamma=self.transform_gamma,
            transform_noise=self.transform_noise,
            transform_blur=self.transform_blur,
            p_per_sample=0.15,
        )

        train_gen = MultiThreadedAugmenter(
            train_gen,
            transforms,
            self.n_processes,
            num_cached_per_queue=4,
            seeds=list(range(self.seed, self.seed + self.n_processes)),
            pin_memory=True,
        )
        ### Use single threaded for debugging
        # train_gen = SingleThreadedAugmenter(
        #     train_gen,
        #     transforms,
        # )

        return train_gen

    def val_dataloader(
        self,
    ) -> Union[
        FutureContextGenerator2D,
        FutureContextGenerator3D,
        RandomFutureContextGenerator2D,
        RandomFutureContextGenerator3D,
    ]:
        """Construct validation dataloader."""

        if self.validate_random:
            if self.dim == 2:
                gen = RandomFutureContextGenerator2D
            else:
                gen = RandomFutureContextGenerator3D
        else:
            if self.dim == 2:
                gen = FutureContextGenerator2D
            else:
                gen = FutureContextGenerator3D

        val_data = load("r", subjects=self.subjects_val, ddir=self.data_dir)
        val_gen = gen(
            data=val_data,
            batch_size=self.batch_size,
            time_size=self.time_size,
            axis=self.axis,
            patch_size=self.patch_size,
            only_tumor=self.only_tumor,
            whole_tumor=self.whole_tumor,
            ddir=self.data_dir,
            number_of_threads_in_multithreaded=1,
            merge_context_target=True,
            drop_labels=self.drop_labels,
        )

        return val_gen

    def test_dataloader(
        self,
    ) -> Union[FutureContextGenerator2D, FutureContextGenerator3D]:
        """Construct test dataloader."""

        if self.dim == 2:
            gen = FutureContextGenerator2D
        else:
            gen = FutureContextGenerator3D

        test_data = load("r", subjects=self.subjects_test, ddir=self.data_dir)
        test_gen = gen(
            data=test_data,
            batch_size=1,
            time_size=self.time_size,
            axis=self.axis,
            patch_size=self.patch_size,
            only_tumor=self.only_tumor,
            whole_tumor=self.whole_tumor,
            ddir=self.data_dir,
            number_of_threads_in_multithreaded=1,
            merge_context_target=True,
            drop_labels=self.drop_labels,
        )

        return test_gen

    @staticmethod
    def add_data_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add module arguments to parser."""

        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--time_size", type=int, nargs="+", default=(2, 3, 4, 5))
        parser.add_argument("--dim", type=int, default=2)
        parser.add_argument("--axis", type=int, default=0)
        parser.add_argument("--split_N", type=int, default=5)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--split_val", type=int, default=3)
        parser.add_argument("--split_test", type=int, default=4)
        parser.add_argument("--patch_size", type=int, default=64)
        parser.add_argument("--whole_tumor", type=str2bool, default=False)
        parser.add_argument("--transform_spatial", type=str2bool, default=True)
        parser.add_argument("--transform_elastic", type=str2bool, default=False)
        parser.add_argument("--transform_rot90", type=str2bool, default=True)
        parser.add_argument("--transform_mirror", type=str2bool, default=True)
        parser.add_argument("--transform_gamma", type=str2bool, default=True)
        parser.add_argument("--transform_noise", type=str2bool, default=True)
        parser.add_argument("--transform_blur", type=str2bool, default=True)
        parser.add_argument("--transform_brightness", type=str2bool, default=True)
        parser.add_argument("--n_processes", type=int, default=8)
        parser.add_argument("--data_dir", type=str, required=True)
        parser.add_argument("--normalize_date_factor", type=float, default=0.01)
        parser.add_argument("--drop_labels", type=int, nargs="+", default=(3,))

        return parser

    @classmethod
    def load_from_checkpoint(cls: type, checkpoint_path: str, **kwargs) -> GliomaModule:
        """Load module from checkpoint.

        Args:
            checkpoint_path: Location of the checkpoint.

        Returns:
            The module initialized from the checkpoint.

        """
        checkpoint = pl.utilities.cloud_io.load(checkpoint_path)["hyper_parameters"]
        checkpoint.update(kwargs)
        return cls(**checkpoint)

    @staticmethod
    def show_examples(
        *dataloaders: SlimDataLoaderBase,
        channel: int = 0,
        seg_cmap: Optional[mp.colors.Colormap] = mp.cm.viridis,
        figsize: int = 2,
        axis: int = 0,
    ) -> mp.figure.Figure:
        """Show examples from all dataloaders.

        Mainly for visual debugging. Only 2D at the moment.

        Args:
            dataloaders: Show examples from these dataloaders. These will most likely
                be what you get from .train_dataloader() etc.
            channel: Which channel to show.
            seg_cmap: Colormap for the segmentation.
            figsize: Size of an individual panel in the figure.
            axis: Use this axis for 3D loaders.

        Returns:
            A figure object

        """

        batches = [next(dl) for dl in dataloaders]

        nrows = 2 * len(batches)
        ncols = max([b["data"].shape[1] // 4 for b in batches])
        fig, ax = plt.subplots(nrows, ncols, figsize=(figsize * ncols, figsize * nrows))
        if ax.ndim == 1:
            ax = ax[:, None]

        for t in range(ncols):
            for b, batch in enumerate(batches):
                if t * 4 < batch["data"].shape[1]:
                    data_current = batch["data"][0, t * 4 + channel]
                    seg_current = batch["seg"][0, t]
                    if data_current.ndim == 3:
                        slc = [slice(None)] * 3
                        slc[axis] = data_current.shape[axis] // 2
                        data_current = data_current[tuple(slc)]
                        seg_current = seg_current[tuple(slc)]
                    ax[2 * b, t].imshow(data_current, cmap="gray")
                    ax[2 * b + 1, t].imshow(seg_current, cmap=seg_cmap, vmin=0, vmax=3)

        for r in range(ax.shape[0]):
            for c in range(ax.shape[1]):
                ax[r, c].axis("off")

        fig.tight_layout()
        return fig
