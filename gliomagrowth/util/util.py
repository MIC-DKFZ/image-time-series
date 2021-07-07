import argparse
import numpy as np
import torch
from typing import Union, Iterable, Optional
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


def merge_axes(tensor, ax1, ax2):

    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)

    shape = list(tensor.shape)
    shape[ax1] = shape[ax1] * shape[ax2]
    del shape[ax2]

    return tensor.reshape(*shape)


def unmerge_axes(tensor, ax1, ax2, ax1_newsize):

    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)

    shape = list(tensor.shape)
    shape.insert(ax2, shape[ax1] // ax1_newsize)
    shape[ax1] = ax1_newsize

    return tensor.reshape(*shape)


def ct_to_transformable(context, target, keys=("data", "seg", "scan_days")):

    for key in keys:
        context[key] = np.concatenate((context[key], target[key]), 1)
        context[key] = merge_axes(context[key], 1, 2)
    context["target_size"] = target["data"].shape[1]
    context["batch_size"] = target["data"].shape[0]
    return context


def transformable_to_ct(batch, keys=("data", "seg", "scan_days"), make_new=False):

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
    old_bbox_lower,
    old_bbox_upper,
    target_size,
    max_size,
    skip_axes=None,
    min_bbox_vals=None,
    max_bbox_vals=None,
):

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
    def __getitem__(self, key):
        return np.load(super().__getitem__(key))[key]
