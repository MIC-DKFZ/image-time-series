import argparse
import numpy as np
import torch
from torch import nn
from typing import Union, Iterable, Optional, Dict, Tuple
from types import ModuleType


def str2bool(v: Union[bool, str]) -> bool:
    """Use this as the type for an ArgumentParser argument.

    Allows you to do --my_flag false and similar, so the flag name can be positive,
    regardless of the default value.

    Args:
        v: The parsed value.

    Returns:
        v interpreted as a boolean.

    """

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def nn_module_lookup(
    name: str,
    dim: int = 2,
    additional_module_sources: Optional[Union[ModuleType, Iterable[ModuleType]]] = None,
) -> type:
    """Look up nn.Modules from strings.

    By default only looks in torch.nn, but you can manually provide more sources.

    Examples:
    >>> nn_module_lookup("conv", dim=3)
    torch.nn.modules.conv.Conv3d
    >>> nn_module_lookup("relu")
    torch.nn.modulesReLU

    Args:
        name: Name of the desired module. Can be either be all lowercase or match
            exactly (i.e. "ReLu" wouldn't match). Doesn't have to include Nd suffix,
            but if it does, it takes precedence:
            >>> nn_module_lookup("conv2d", dim=3)
            torch.nn.modules.conv.Conv2d
        dim: Will try to find "{name}{dim}d", but a match without the suffix takes
            precedence.
        additional_module_sources: Additional modules for the lookup. These will take
            precedence over torch.nn

    Returns:
        The found type, will raise if none is found.

    """

    if isinstance(name, type):
        return name

    if name in ("none", None):
        return None

    lookup_dict = {}

    for item in dir(torch.nn):
        if item.startswith("_"):
            continue
        lookup_dict[item] = getattr(torch.nn, item)
        lookup_dict[item.lower()] = getattr(torch.nn, item)

    if additional_module_sources is not None:
        if not hasattr(additional_module_sources, "__iter__"):
            additional_module_sources = [additional_module_sources]
        for mod_source in additional_module_sources:
            for item in dir(mod_source):
                if item.startswith("_"):
                    continue
                lookup_dict[item] = getattr(mod_source, item)
                lookup_dict[item.lower()] = getattr(mod_source, item)

    if name in lookup_dict:
        return lookup_dict[name]
    elif name + str(dim) + "d" in lookup_dict:
        return lookup_dict[name + str(dim) + "d"]
    else:
        raise AttributeError(
            "Couldn't find module with name {} and dimension {} in torch.nn.".format(
                name, dim
            )
        )


def num_parameters(model: torch.nn.Module, count_all: bool = True) -> int:
    """Count parameters in a model.

    Args:
        model: We iterate over model.parameters() and count .numel()
        count_all: Also count parameters with p.requires_grad False

    Returns:
        The number of parameters.

    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad or count_all)


def make_onehot(
    array: Union[np.ndarray, torch.tensor],
    labels: Optional[Iterable[int]] = None,
    axis: int = 1,
    newaxis: bool = False,
) -> Union[np.ndarray, torch.tensor]:
    """Convert input array to onehot.

    Args:
        array: The array to be converted.
        labels: The labels that should be included in the onehot array
            (others will become background). If None, we do np.unique on the array.
        axis: Stack onehot slices along this axis. If the initial size of this axis is
            larger than one, each array along this axis will be converted individually,
            so the new size will be old_size * len(labels).
        newaxis: Create a new axis to stack onehot slices.

    Returns:
        The onehot array (same type, dtype, device as input).

    """

    # get labels if necessary
    if labels is None:
        labels = np.unique(array)
        labels = list(map(lambda x: x.item(), labels))

    # get target shape
    new_shape = list(array.shape)
    if newaxis:
        new_shape.insert(axis, len(labels))
    else:
        new_shape[axis] = new_shape[axis] * len(labels)

    # make zero array
    if type(array) == np.ndarray:
        new_array = np.zeros(new_shape, dtype=array.dtype)
    elif torch.is_tensor(array):
        new_array = torch.zeros(new_shape, dtype=array.dtype, device=array.device)
    else:
        raise TypeError(
            "Onehot conversion undefined for object of type {}".format(type(array))
        )

    # fill new array
    n_seg_channels = 1 if newaxis else array.shape[axis]
    for seg_channel in range(n_seg_channels):
        for l, label in enumerate(labels):
            new_slc = [
                slice(None),
            ] * len(new_shape)
            slc = [
                slice(None),
            ] * len(array.shape)
            new_slc[axis] = seg_channel * len(labels) + l
            if not newaxis:
                slc[axis] = seg_channel
            new_array[tuple(new_slc)] = array[tuple(slc)] == label

    return new_array


def stack_batch(
    tensor: Union[np.ndarray, torch.tensor]
) -> Union[np.ndarray, torch.tensor]:
    """Stacks first axis along second axis.

    Args:
        tensor: Tensor with shape (B, C, ...).

    Returns:
        tensor reshaped to (B*C, ...).

    """

    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def unstack_batch(
    tensor: Union[np.ndarray, torch.tensor], B: int
) -> Union[np.ndarray, torch.tensor]:
    """Reverses stack_batch.

    Args:
        tensor: Tensor with shape (B*C, ...).
        B: Size of new zero axis, must evenly divide zero axis of input tensor.

    Returns:
        tensor: Tensor with shape (B, C, ...).

    """

    N = tensor.shape[0] // B
    return tensor.reshape(B, N, *tensor.shape[1:])


def merge_axes(
    tensor: Union[np.ndarray, torch.tensor], ax1: int, ax2: int
) -> Union[np.ndarray, torch.tensor]:
    """Merge two neighbouring axes.

    Args:
        tensor: Input tensor.
        ax1: First axis.
        ax2: Second axis.

    Returns:
        Tensor where ax1 now has shape ax1_size * ax2_size.

    """

    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)

    shape = list(tensor.shape)
    shape[ax1] = shape[ax1] * shape[ax2]
    del shape[ax2]

    return tensor.reshape(*shape)


def unmerge_axes(
    tensor: Union[np.ndarray, torch.tensor], ax1: int, ax2: int, ax1_newsize: int
) -> Union[np.ndarray, torch.tensor]:
    """Reverse merge_axes().

    Args:
        tensor: Input tensor.
        ax1: First axis.
        ax2: Second axis.
        ax1_newsize: New size of ax1.

    Returns:
        Tensor where ax1 was split.

    """

    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)

    shape = list(tensor.shape)
    shape.insert(ax2, shape[ax1] // ax1_newsize)
    shape[ax1] = ax1_newsize

    return tensor.reshape(*shape)


def ct_to_transformable(
    context: Dict[str, Union[np.ndarray, torch.tensor]],
    target: Dict[str, Union[np.ndarray, torch.tensor]],
    keys: Iterable[str] = ("scan_days",),
) -> Dict[str, Union[np.ndarray, torch.tensor]]:
    """Convert a pair of context/target batch dictionaries to a single one.

    This allows you to apply data augmentation. Data in context and target dictionaries
    will be concatenated and placed in the context dict.

    Args:
        context: Context dict, needs to contain provided keys and "data" and "seg".
        target: Target dict, needs to contain provided keys and "data" and "seg".
        keys: Keys for which data should be concatenated ("data" and "seg" always used).

    Returns:
        Context dict (same object) with concatenated data.

    """

    keys = ["data", "seg"] + list(keys)

    for key in keys:
        if key not in context or key not in target:
            continue
        context[key] = np.concatenate((context[key], target[key]), 1)
        context[key] = merge_axes(context[key], 1, 2)
    context["target_size"] = target["data"].shape[1]
    context["batch_size"] = target["data"].shape[0]
    return context


def transformable_to_ct(
    batch: Dict[str, Union[np.ndarray, torch.tensor]],
    keys: Iterable[str] = ("scan_days",),
    make_new: bool = False,
) -> Tuple[
    Dict[str, Union[np.ndarray, torch.tensor]],
    Dict[str, Union[np.ndarray, torch.tensor]],
]:
    """Undo ct_to_transformable().

    Args:
        batch: Batch dict with concatenated context/target data. Also needs to contain
            key "target_size".
        keys: Keys for which data should be concatenated.
        make_new: Create new dict instead of using input as context dict.

    Returns:
        Context and target dicts.

    """

    if make_new:
        context = {}
    else:
        context = batch
    target = {}

    target_size = batch["target_size"]
    time_size = batch["seg"].shape[1]

    for key in keys:
        x = unmerge_axes(batch[key], 1, 2, time_size)
        target[key] = x[:, -target_size:]
        context[key] = x[:, :-target_size]

    return context, target


def modify_bbox(
    old_bbox_lower: Iterable[int],
    old_bbox_upper: Iterable[int],
    target_size: Union[int, Iterable[int]],
    max_size: Union[int, Iterable[int]],
    skip_axes: Optional[Union[int, Iterable[int]]] = None,
    min_bbox_vals: Optional[Union[int, Iterable[int]]] = None,
    max_bbox_vals: Optional[Union[int, Iterable[int]]] = None,
) -> Tuple[Iterable[int], Iterable[int]]:
    """Modify a bounding box to a new size.

    Expand or shrink a bounding box symmetrically to a desired size. However, you can
    provide limits for the indices. If these are met, the bounding box would continue
    to be expanded only on one side, for example.

    Args:
        old_bbox_lower: Lower limits of the input bounding box.
        old_bbox_upper: Upper limits of the input bounding box.
        target_size: Desired size of the bounding box (not individual limits!).
        max_size: Size limit(s).
        skip_axes: Ignore these axes.
        min_bbox_vals: Lower limit(s) on indices.
        max_bbox_vals: Upper limit(s) on indices.

    Returns:
        New lower and upper indices of the bounding box.

    """

    if skip_axes is None:
        skip_axes = []
    if not hasattr(skip_axes, "__iter__"):
        skip_axes = [skip_axes]
    if min_bbox_vals is None:
        min_bbox_vals = [
            0,
        ] * len(old_bbox_lower)
    if max_bbox_vals is None:
        if hasattr(max_size, "__iter__"):
            max_bbox_vals = [x - 1 for x in max_size]
        else:
            max_bbox_vals = [
                max_size - 1,
            ] * len(old_bbox_lower)

    new_lower_lim = []
    new_upper_lim = []

    for i in range(len(old_bbox_lower)):

        ll = old_bbox_lower[i]
        ul = old_bbox_upper[i]

        if ul < ll:
            raise ValueError(
                "Input upper bbox limit is smaller than lower bbox limit along axis {}.".format(
                    i
                )
            )

        if i in skip_axes:
            new_lower_lim.append(ll)
            new_upper_lim.append(ul)
            continue

        if hasattr(max_size, "__iter__"):
            ms = max_size[i]
        else:
            ms = max_size
        if hasattr(target_size, "__iter__"):
            ts = target_size[i]
        else:
            ts = target_size
        if hasattr(min_bbox_vals, "__iter__"):
            min_bv = min_bbox_vals[i]
        else:
            min_bv = min_bbox_vals
        if hasattr(max_bbox_vals, "__iter__"):
            max_bv = max_bbox_vals[i]
        else:
            max_bv = max_bbox_vals

        width = ul - ll + 1
        switch = 0
        while width != ts and width <= ms and width > 0:

            if width > ts:

                if switch % 2 == 0:
                    ul -= 1
                else:
                    ll += 1

            else:

                if switch % 2 == 0 and ul < max_bv:
                    ul += 1
                if switch % 2 != 0 and ll > min_bv:
                    ll -= 1

            switch += 1
            width = ul - ll + 1

        new_lower_lim.append(ll)
        new_upper_lim.append(ul)

    return new_lower_lim, new_upper_lim


class NpzLazyDict(dict):
    """A dict data type that lazily loads npz files.

    You fill the dict with file names but receive a numpy array when accessing existing
    keys.
    """

    def __getitem__(self, key):
        return np.load(super().__getitem__(key))


def is_conv(op: type) -> bool:
    """Check if the input is one of the torch conv operators (or a subclass)."""

    conv_types = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )
    if type(op) == type and issubclass(op, conv_types):
        return True
    elif type(op) in conv_types:
        return True
    else:
        return False


def match_to(x: torch.tensor, ref: torch.tensor, keep_axes: Iterable[int] = (1,)):
    """Modify the shape of an input to match a reference by repeating elements.

    Uses .expand() on the input.

    Args:
        x: Input tensor.
        ref: Reference tensor.
        keep_axes: Don't change these axes.

    Returns:
        The modified input.

    """

    target_shape = list(ref.shape)
    for i in keep_axes:
        target_shape[i] = x.shape[i]
    target_shape = tuple(target_shape)
    if x.shape == target_shape:
        pass
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    else:
        while x.ndim < len(target_shape):
            x = x.unsqueeze(-1)

    x = x.expand(*target_shape)
    x = x.to(device=ref.device, dtype=ref.dtype)

    return x
