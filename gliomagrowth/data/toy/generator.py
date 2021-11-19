from __future__ import annotations

import argparse
import numpy as np
from collections import defaultdict
import inspect
from typing import Union, Optional, Iterable, Callable, Dict, Any, Tuple

from batchgenerators.dataloading import SlimDataLoaderBase
import pytorch_lightning as pl
from gliomagrowth.util.lightning import str2bool, DictArgument
from gliomagrowth.util.util import matchcall, ct_to_transformable, save_gif_grid
from gliomagrowth.data.toy import shape, trajectory


class RandomTrajectoryGenerator(SlimDataLoaderBase):
    """Generate random objects moving along trajectories.

    Args:
        batch_size: Number of independent examples.
        context_size: Size of context set.
        target_size: Size of target set.
        target_includes_context: If True, the context is added to the target set.
        merge_context_target: If True, stack context and target.
        shape_function: Function to generate shapes.
        shape_kwargs: Keyword arguments for shape_function. Individual arguments can
            also be tuples. In that case we either draw randomly
            (if meta_trajectories=False) or vary the parameter linearly along the
            trajectory (if meta_trajectories=True).
        shape_trajectory_identifiers: These are the keywords for the shape function that
            will be provided by the trajectory function.
        trajectory_function: Function to generate trajectories.
        trajectory_kwargs: Keyword arguments for trajectory_function. Individual
            arguments can/should be tuples for random draws.
        num_objects: Number of objects per image.
        min_length: Minimum length of trajectory. Only works if the trajectory_function
            uses start_x, start_y, end_x, end_y keywords.
        meta_trajectories: See shape_kwargs explanation.
        circular_position: Activate this if start and end points of trajectories are the
            same. We then use two-dimensional positions with sin/cos embedding.
        num_max: Maximum number of elements along trajectory. This should be higher than
            context and target size combined and will otherwise be exactly that. It
            essentially controls the spacing between points, because we use linspace to
            construct the trajectory.
        image_size: Size of the image.

    """

    def __init__(
        self,
        batch_size: int,
        context_size: Union[int, Iterable[int]] = (3, 10),
        target_size: Union[int, Iterable[int]] = (10, 50),
        target_includes_context: bool = False,
        merge_context_target: bool = False,
        shape_function: Union[Callable, Iterable[Callable]] = shape.circle,
        shape_kwargs: Union[
            Dict[str, Union[float, Tuple[float, float], Any]],
            Iterable[Dict[str, Union[float, Tuple[float, float], Any]]],
        ] = dict(radius=(0.05, 0.25), image_size=64),
        shape_trajectory_identifiers: Union[
            Tuple[str, str], Iterable[Tuple[str, str]]
        ] = ("center_x", "center_y"),
        trajectory_function: Union[Callable, Iterable[Callable]] = trajectory.line_,
        trajectory_kwargs: Union[
            Dict[str, Tuple[float, float]], Iterable[Dict[str, Tuple[float, float]]]
        ] = dict(
            start_x=(0.0, 1.0), start_y=(0.0, 1.0), end_x=(0.0, 1.0), end_y=(0.0, 1.0)
        ),
        num_objects: int = 1,
        min_length: float = 0.0,
        meta_trajectories: bool = False,
        circular_position: Union[bool, Iterable[bool]] = False,
        num_max: int = 100,
        image_size: int = 64,
        **kwargs,
    ):
        number_of_threads_in_multithreaded = kwargs.get(
            "number_of_threads_in_multithreaded", None
        )
        super().__init__(None, batch_size, number_of_threads_in_multithreaded)

        self.context_size = context_size
        self.target_size = target_size
        self.target_includes_context = target_includes_context
        self.merge_context_target = merge_context_target
        if hasattr(shape_function, "__iter__"):
            self.shape_function = shape_function
        else:
            self.shape_function = [shape_function] * num_objects
        if isinstance(shape_kwargs, dict):
            self.shape_kwargs = [shape_kwargs.copy()] * num_objects
        else:
            self.shape_kwargs = [sk.copy() for sk in shape_kwargs]
        if isinstance(shape_trajectory_identifiers[0], str):
            self.shape_trajectory_identifiers = [
                shape_trajectory_identifiers
            ] * num_objects
        else:
            self.shape_trajectory_identifiers = shape_trajectory_identifiers
        if hasattr(trajectory_function, "__iter__"):
            self.trajectory_function = trajectory_function
        else:
            self.trajectory_function = [trajectory_function] * num_objects
        if isinstance(trajectory_kwargs, dict):
            self.trajectory_kwargs = [trajectory_kwargs.copy()] * num_objects
        else:
            self.trajectory_kwargs = [tk.copy() for tk in trajectory_kwargs]
        self.num_objects = num_objects
        self.min_length = min_length
        self.meta_trajectories = meta_trajectories
        if hasattr(circular_position, "__iter__"):
            self.circular_position = circular_position
        else:
            self.circular_position = [circular_position] * num_objects
        self.num_max = num_max
        self.image_size = image_size

    def generate_item(
        self,
        context_size: int,
        target_size: int,
        shape_params: Iterable[str],
        traj_params: Iterable[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate a single example.

        Args:
            context_size: Number of context points.
            target_size: Number of target points.
            shape_params: Iterable of parameter names for shape function.
            traj_params: Iterable of parameter names for trajectory function.

        Returns:
            Dicts for context and target.

        """

        num_max = max(self.num_max, context_size + target_size)

        params_context = defaultdict(list)
        params_target = defaultdict(list)

        coordinates = []
        positions = []

        for t in range(self.num_objects):

            # get coordinates along trajectory
            while True:
                call_dict = {}
                for key in traj_params:
                    if key not in self.trajectory_kwargs[t]:
                        continue
                    val = self.trajectory_kwargs[t][key]
                    if hasattr(val, "__iter__") and not isinstance(val, str):
                        try:
                            val = np.random.uniform(*val)
                        except ValueError:
                            val = np.random.choice(val)
                    call_dict[key] = val
                if self.min_length > 0 and not self.circular_position[t]:
                    len_ = np.sqrt(
                        (call_dict["start_x"] - call_dict["end_x"]) ** 2
                        + (call_dict["start_y"] - call_dict["end_y"]) ** 2
                    )
                    if len_ < self.min_length:
                        continue
                    else:
                        break
                else:
                    break

            coordinates.append(
                np.array(self.trajectory_function[t](num_max, **call_dict)).astype(
                    np.float32
                )
            )  # (N, len(shape_trajectory_identifiers))
            if not self.circular_position[t]:
                positions.append(
                    np.linspace(0, 1, num_max).reshape(-1, 1).astype(np.float32)
                )  # (N, 1)
            else:
                pos = (
                    np.linspace(0, 2 * np.pi, num_max + 1)[:-1]
                    .reshape(-1, 1)
                    .astype(np.float32)
                )  # (N, 1)
                positions.append(np.concatenate((np.sin(pos), np.cos(pos)), 1))

        coordinates = np.concatenate(
            coordinates, 1
        )  # (N, T*len(shape_trajectory_identifiers))
        positions = np.concatenate(positions, 1)  # (N, T*(1 or 2))

        indices_all = np.random.choice(
            np.arange(num_max), context_size + target_size, replace=False
        )

        indices_context = indices_all[:context_size]
        indices_context.sort()
        coordinates_context = coordinates[indices_context]
        positions_context = positions[indices_context]
        params_context["position"] = positions_context

        if self.target_includes_context:
            indices_target = indices_all
            target_size = target_size + context_size
        else:
            indices_target = indices_all[-target_size:]
        indices_target.sort()
        coordinates_target = coordinates[indices_target]
        positions_target = positions[indices_target]
        params_target["position"] = positions_target

        # now create images
        data_context = np.zeros(
            (context_size, 1, self.image_size, self.image_size), dtype=np.uint8
        )
        data_target = np.zeros(
            (target_size, 1, self.image_size, self.image_size), dtype=np.uint8
        )

        for t in range(self.num_objects):

            joint_call_dict = {}
            for key in shape_params:

                if (
                    key not in self.shape_kwargs[t]
                    or key in self.shape_trajectory_identifiers[t]
                ):
                    continue
                val = self.shape_kwargs[t][key]

                # value not fixed, so we draw randomly or construct trajectory
                if hasattr(val, "__iter__") and not isinstance(val, str):

                    # random
                    if not self.meta_trajectories:

                        try:
                            val = np.random.uniform(*val)
                        except ValueError:
                            val = np.random.choice(val)
                        params_context[key].append(
                            np.array(
                                [
                                    val,
                                ]
                                * context_size
                            )
                        )
                        params_target[key].append(
                            np.array(
                                [
                                    val,
                                ]
                                * target_size
                            )
                        )

                    # meta trajectory
                    else:

                        limits = np.random.uniform(*val, size=(2,))
                        if not self.circular_position[t]:
                            vals = np.linspace(*limits, num_max)
                        else:
                            vals = np.cos(np.linspace(-np.pi, np.pi, num_max))
                            vals = limits[0] + 0.5 * (vals + 1) * limits[1]
                        params_context[key].append(vals[indices_context])
                        params_target[key].append(vals[indices_target])

                else:

                    joint_call_dict[key] = val

            for i in range(coordinates_context.shape[0]):
                call_dict = {}
                for key, val in params_context.items():
                    if key != "position":
                        call_dict[key] = params_context[key][-1][i]
                c = coordinates_context[i, 2 * t : 2 * (t + 1)]
                for n, name in enumerate(self.shape_trajectory_identifiers[t]):
                    call_dict[name] = c[n]
                img = matchcall(
                    self.shape_function[t],
                    **call_dict,
                    **joint_call_dict,
                ).astype(bool)
                data_context[i, 0][img] = t + 1

            for i in range(coordinates_target.shape[0]):
                call_dict = {}
                for key, val in params_target.items():
                    if key != "position":
                        call_dict[key] = params_target[key][-1][i]
                c = coordinates_target[i, 2 * t : 2 * (t + 1)]
                for n, name in enumerate(self.shape_trajectory_identifiers[t]):
                    call_dict[name] = c[n]
                img = matchcall(
                    self.shape_function[t],
                    **call_dict,
                    **joint_call_dict,
                ).astype(bool)
                data_target[i, 0][img] = t + 1

        for key, val in params_context.items():
            if key != "position":
                params_context[key] = np.array(val, dtype=np.float32).T
        for key, val in params_target.items():
            if key != "position":
                params_target[key] = np.array(val, dtype=np.float32).T

        params_context["data"] = data_context
        params_target["data"] = data_target

        return params_context, params_target

    def generate_train_batch(
        self,
    ) -> Union[
        Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
    ]:

        # everything we need to look at for generator function
        shape_params = []
        for i in range(self.num_objects):
            shape_params.extend(
                list(inspect.signature(self.shape_function[i]).parameters)
            )

        # everything we need to look at for trajectory function
        traj_params = []
        for i in range(self.num_objects):
            traj_params.extend(
                list(inspect.signature(self.trajectory_function[i]).parameters)
            )

        # draw context_size and target_size for batch
        if hasattr(self.context_size, "__iter__"):
            context_size = np.random.randint(*self.context_size)
        else:
            context_size = self.context_size
        if hasattr(self.target_size, "__iter__"):
            target_size = np.random.randint(*self.target_size)
        else:
            target_size = self.target_size

        for b in range(self.batch_size):

            params_context, params_target = self.generate_item(
                context_size, target_size, shape_params, traj_params
            )

            if b == 0:
                batch_params_context = {
                    key: [
                        val,
                    ]
                    for key, val in params_context.items()
                }
                batch_params_target = {
                    key: [
                        val,
                    ]
                    for key, val in params_target.items()
                }
            else:
                for key, val in params_context.items():
                    batch_params_context[key].append(val)
                for key, val in params_target.items():
                    batch_params_target[key].append(val)

        for key, val in batch_params_context.items():
            val = np.stack(val)
            if val.ndim == 2:
                val = val[:, :, None]
            batch_params_context[key] = val

        for key, val in batch_params_target.items():
            val = np.stack(val)
            if val.ndim == 2:
                val = val[:, :, None]
            batch_params_target[key] = val

        if self.merge_context_target:
            return ct_to_transformable(
                batch_params_context,
                batch_params_target,
                keys=("position", "data"),
            )
        else:
            return batch_params_context, batch_params_target


class SegmentationTrajectoryModule(pl.LightningDataModule):
    """LightningDataModule that wraps data generator. For each argument there is also
    argument_test, which can optionally be used to change the test data.

    Args:
        batch_size: Number of independent examples.
        context_size: Size of context set.
        target_size: Size of target set.
        target_includes_context: If True, the context is added to the target set.
        merge_context_target: If True, stack context and target.
        shape_function: Function to generate shapes. We will perform a lookup in the
            shape module for this string.
        shape_kwargs: Keyword arguments for shape_function. Individual arguments can
            also be tuples. In that case we either draw randomly
            (if meta_trajectories=False) or vary the parameter linearly along the
            trajectory (if meta_trajectories=True).
        shape_trajectory_identifiers: These are the keywords for the shape function that
            will be provided by the trajectory function.
        trajectory_function: Function to generate trajectories. We will perform a lookup
            in the trajectory module for this string.
        trajectory_kwargs: Keyword arguments for trajectory_function. Individual
            arguments can/should be tuples for random draws.
        num_objects: Number of objects per image.
        min_length: Minimum length of trajectory. Only works if the trajectory_function
            uses start_x, start_y, end_x, end_y keywords.
        meta_trajectories: See shape_kwargs explanation.
        circular_position: Activate this if start and end points of trajectories are the
            same. We then use two-dimensional positions with sin/cos embedding.
        num_max: Maximum number of elements along trajectory. This should be higher than
            context and target size combined and will otherwise be exactly that. It
            essentially controls the spacing between points, because we use linspace to
            construct the trajectory.
        image_size: Size of the image.

    """

    def __init__(
        self,
        batch_size: int = 128,
        context_size: Union[int, Iterable[int]] = (3, 10),
        target_size: Union[int, Iterable[int]] = (10, 50),
        target_includes_context: bool = False,
        merge_context_target: bool = False,
        shape_function: Union[str, Iterable[str]] = "circle",
        shape_kwargs: Union[
            Dict[str, Union[float, Tuple[float, float], Any]],
            Iterable[Dict[str, Union[float, Tuple[float, float], Any]]],
        ] = dict(radius=(0.05, 0.25)),
        shape_trajectory_identifiers: Union[
            Tuple[str, str], Iterable[Tuple[str, str]]
        ] = ("center_x", "center_y"),
        trajectory_function: Union[str, Iterable[str]] = "line_",
        trajectory_kwargs: Union[
            Dict[str, Tuple[float, float]], Iterable[Dict[str, Tuple[float, float]]]
        ] = dict(
            start_x=(0.0, 1.0), start_y=(0.0, 1.0), end_x=(0.0, 1.0), end_y=(0.0, 1.0)
        ),
        num_objects: int = 1,
        min_length: float = 0.0,
        meta_trajectories: bool = False,
        circular_position: Union[bool, Iterable[bool]] = False,
        num_max: int = 100,
        image_size: int = 64,
        batch_size_test: Optional[int] = None,
        context_size_test: Optional[Union[int, Iterable[int]]] = None,
        target_size_test: Optional[Union[int, Iterable[int]]] = None,
        target_includes_context_test: Optional[bool] = None,
        merge_context_target_test: Optional[bool] = None,
        shape_function_test: Optional[Union[str, Iterable[str]]] = None,
        shape_kwargs_test: Optional[
            Union[
                Dict[str, Union[float, Tuple[float, float], Any]],
                Iterable[Dict[str, Union[float, Tuple[float, float], Any]]],
            ]
        ] = None,
        shape_trajectory_identifiers_test: Optional[
            Union[Tuple[str, str], Iterable[Tuple[str, str]]]
        ] = None,
        trajectory_function_test: Optional[Union[str, Iterable[str]]] = None,
        trajectory_kwargs_test: Optional[
            Union[
                Dict[str, Tuple[float, float]], Iterable[Dict[str, Tuple[float, float]]]
            ]
        ] = None,
        num_objects_test: Optional[int] = None,
        min_length_test: Optional[float] = None,
        meta_trajectories_test: Optional[bool] = None,
        circular_position_test: Optional[Union[bool, Iterable[bool]]] = None,
        num_max_test: Optional[int] = None,
        image_size_test: Optional[int] = None,
        **kwargs,
    ):

        self.batch_size = batch_size
        self.context_size = context_size
        self.target_size = target_size
        self.target_includes_context = target_includes_context
        self.merge_context_target = merge_context_target
        if hasattr(shape_function, "__iter__") and not isinstance(shape_function, str):
            self.shape_function = [getattr(shape, sf) for sf in shape_function]
        else:
            self.shape_function = [getattr(shape, shape_function)] * num_objects
        if isinstance(shape_kwargs, dict):
            self.shape_kwargs = [shape_kwargs.copy()] * num_objects
        else:
            self.shape_kwargs = [sk.copy() for sk in shape_kwargs]
        if isinstance(shape_trajectory_identifiers[0], str):
            self.shape_trajectory_identifiers = [
                shape_trajectory_identifiers
            ] * num_objects
        else:
            self.shape_trajectory_identifiers = shape_trajectory_identifiers
        if hasattr(trajectory_function, "__iter__") and not isinstance(
            trajectory_function, str
        ):
            self.trajectory_function = [
                getattr(trajectory, tf) for tf in trajectory_function
            ]
        else:
            self.trajectory_function = [
                getattr(trajectory, trajectory_function)
            ] * num_objects
        if isinstance(trajectory_kwargs, dict):
            self.trajectory_kwargs = [trajectory_kwargs.copy()] * num_objects
        else:
            self.trajectory_kwargs = [tk.copy() for tk in trajectory_kwargs]
        self.num_objects = num_objects
        self.min_length = min_length
        self.meta_trajectories = meta_trajectories
        if hasattr(circular_position, "__iter__"):
            self.circular_position = circular_position
        else:
            self.circular_position = [circular_position] * num_objects
        self.num_max = num_max
        self.image_size = image_size
        for kw in self.shape_kwargs:
            kw["image_size"] = image_size

        # test args
        self.num_objects_test = (
            num_objects_test if num_objects_test is not None else self.num_objects
        )
        self.batch_size_test = (
            batch_size_test if batch_size_test is not None else self.batch_size
        )
        self.context_size_test = (
            context_size_test if context_size_test is not None else context_size
        )
        self.target_size_test = (
            target_size_test if target_size_test is not None else target_size
        )
        self.target_includes_context_test = (
            target_includes_context_test
            if target_includes_context_test is not None
            else target_includes_context
        )
        self.merge_context_target_test = (
            merge_context_target_test
            if merge_context_target_test is not None
            else merge_context_target
        )
        if shape_function_test is not None:
            if hasattr(shape_function_test, "__iter__") and not isinstance(
                shape_function_test, str
            ):
                self.shape_function_test = [
                    getattr(shape, sf) for sf in shape_function_test
                ]
            else:
                self.shape_function_test = [
                    getattr(shape, shape_function_test)
                ] * num_objects_test
        else:
            self.shape_function_test = self.shape_function
        if shape_kwargs_test is not None:
            if isinstance(shape_kwargs_test, dict):
                self.shape_kwargs_test = [shape_kwargs_test.copy()] * num_objects_test
            else:
                self.shape_kwargs_test = [sk.copy() for sk in shape_kwargs_test]
        else:
            self.shape_kwargs_test = self.shape_kwargs.copy()
        if shape_trajectory_identifiers_test is not None:
            if isinstance(shape_trajectory_identifiers_test[0], str):
                self.shape_trajectory_identifiers_test = [
                    shape_trajectory_identifiers_test
                ] * num_objects_test
            else:
                self.shape_trajectory_identifiers_test = (
                    shape_trajectory_identifiers_test
                )
        else:
            self.shape_trajectory_identifiers_test = self.shape_trajectory_identifiers
        if trajectory_function_test is not None:
            if hasattr(trajectory_function_test, "__iter__") and not isinstance(
                trajectory_function_test, str
            ):
                self.trajectory_function_test = [
                    getattr(trajectory, tf) for tf in trajectory_function_test
                ]
            else:
                self.trajectory_function_test = [
                    getattr(trajectory, trajectory_function_test)
                ] * num_objects_test
        else:
            self.trajectory_function_test = self.trajectory_function
        if trajectory_kwargs_test is not None:
            if isinstance(trajectory_kwargs_test, dict):
                self.trajectory_kwargs_test = [
                    trajectory_kwargs_test.copy()
                ] * num_objects_test
            else:
                self.trajectory_kwargs_test = [
                    tk.copy() for tk in trajectory_kwargs_test
                ]
        else:
            self.trajectory_kwargs_test = self.trajectory_kwargs.copy()
        self.min_length_test = (
            min_length_test if min_length_test is not None else self.min_length
        )
        self.meta_trajectories_test = (
            meta_trajectories_test
            if meta_trajectories_test is not None
            else self.meta_trajectories
        )
        if circular_position_test is not None:
            if hasattr(circular_position_test, "__iter__"):
                self.circular_position_test = circular_position_test
            else:
                self.circular_position_test = [
                    circular_position_test
                ] * num_objects_test
        else:
            self.circular_position_test = self.circular_position
        self.num_max_test = num_max_test if num_max_test is not None else self.num_max
        self.image_size_test = (
            image_size_test if image_size_test is not None else self.image_size
        )
        for kw in self.shape_kwargs_test:
            kw["image_size"] = self.image_size_test

    def setup(self, stage: Optional[str] = None) -> None:
        """Doesn't do anything..."""
        return super().setup(stage=stage)

    def train_dataloader(
        self,
    ) -> RandomTrajectoryGenerator:
        """Construct training dataloader."""

        train_gen = RandomTrajectoryGenerator(
            batch_size=self.batch_size,
            context_size=self.context_size,
            target_size=self.target_size,
            target_includes_context=self.target_includes_context,
            merge_context_target=self.merge_context_target,
            shape_function=self.shape_function,
            shape_kwargs=self.shape_kwargs,
            shape_trajectory_identifiers=self.shape_trajectory_identifiers,
            trajectory_function=self.trajectory_function,
            trajectory_kwargs=self.trajectory_kwargs,
            num_objects=self.num_objects,
            min_length=self.min_length,
            meta_trajectories=self.meta_trajectories,
            circular_position=self.circular_position,
            num_max=self.num_max,
            image_size=self.image_size,
        )

        return train_gen

    def val_dataloader(
        self,
    ) -> RandomTrajectoryGenerator:
        """Construct validation dataloader."""

        val_gen = RandomTrajectoryGenerator(
            batch_size=self.batch_size,
            context_size=self.context_size,
            target_size=self.target_size,
            target_includes_context=self.target_includes_context,
            merge_context_target=self.merge_context_target,
            shape_function=self.shape_function,
            shape_kwargs=self.shape_kwargs,
            shape_trajectory_identifiers=self.shape_trajectory_identifiers,
            trajectory_function=self.trajectory_function,
            trajectory_kwargs=self.trajectory_kwargs,
            num_objects=self.num_objects,
            min_length=self.min_length,
            meta_trajectories=self.meta_trajectories,
            circular_position=self.circular_position,
            num_max=self.num_max,
            image_size=self.image_size,
        )

        return val_gen

    def test_dataloader(
        self,
    ) -> RandomTrajectoryGenerator:
        """Construct test dataloader."""

        test_gen = RandomTrajectoryGenerator(
            batch_size=self.batch_size_test,
            context_size=self.context_size_test,
            target_size=self.target_size_test,
            target_includes_context=self.target_includes_context_test,
            merge_context_target=self.merge_context_target_test,
            shape_function=self.shape_function_test,
            shape_kwargs=self.shape_kwargs_test,
            shape_trajectory_identifiers=self.shape_trajectory_identifiers_test,
            trajectory_function=self.trajectory_function_test,
            trajectory_kwargs=self.trajectory_kwargs_test,
            num_objects=self.num_objects_test,
            min_length=self.min_length_test,
            meta_trajectories=self.meta_trajectories_test,
            circular_position=self.circular_position_test,
            num_max=self.num_max_test,
            image_size=self.image_size_test,
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
        parser.add_argument("--context_size", type=int, nargs=2, default=(3, 10))
        parser.add_argument("--target_size", type=int, nargs=2, default=(10, 50))
        parser.add_argument("--target_includes_context", type=str2bool, default=False)
        parser.add_argument("--merge_context_target", type=str2bool, default=False)
        parser.add_argument("--shape_function")
        parser.add_argument(
            "--shape_kwargs",
            action=DictArgument,
            nargs="+",
            default=dict(radius=(0.05, 0.25)),
        )
        parser.add_argument(
            "--shape_trajectory_identifiers",
            type=str,
            nargs="+",
            default=("center_x", "center_y"),
        )
        parser.add_argument("--trajectory_function")
        parser.add_argument(
            "--trajectory_kwargs",
            action=DictArgument,
            nargs="+",
            default=dict(
                start_x=(0.0, 1.0),
                start_y=(0.0, 1.0),
                end_x=(0.0, 1.0),
                end_y=(0.0, 1.0),
            ),
        )
        parser.add_argument("--num_objects", type=int, default=1)
        parser.add_argument("--min_length", type=float, default=0.0)
        parser.add_argument("--meta_trajectories", type=str2bool, default=False)
        parser.add_argument("--circular_position", type=str2bool, default=False)
        parser.add_argument("--num_max", type=int, default=100)
        parser.add_argument("--image_size", type=int, default=64)

        parser.add_argument("--batch_size_test", type=int, default=None)

    @classmethod
    def load_from_checkpoint(
        cls: type, checkpoint_path: str, **kwargs
    ) -> SegmentationTrajectoryModule:
        """Load module from checkpoint.

        Args:
            checkpoint_path: Location of the checkpoint.

        Returns:
            The module initialized from the checkpoint.

        """
        checkpoint = pl.utilities.cloud_io.load(checkpoint_path)["hyper_parameters"]
        checkpoint.update(kwargs)
        return cls(**checkpoint)


# for quick debugging :)
if __name__ == "__main__":

    mod = SegmentationTrajectoryModule(
        batch_size=1,
        context_size=5,
        target_size=50,
        shape_function=("circle", "star"),
        shape_kwargs=(
            dict(radius=(0.05, 0.2)),
            dict(size=(0.1, 0.3), t=(0.2, 0.8), rotation=(0.0, 0.5)),
        ),
        shape_trajectory_identifiers=("center_x", "center_y"),
        trajectory_function="line_",
        trajectory_kwargs=dict(
            start_x=(0.0, 1.0),
            start_y=(0.0, 1.0),
            end_x=(0.0, 1.0),
            end_y=(0.0, 1.0),
        ),
        num_objects=2,
        meta_trajectories=True,
        circular_position=False,
        image_size=128,
        shape_function_test=("rectangle", "triangle_square"),
        shape_kwargs_test=(
            dict(
                end_x=(0.0, 1.0),
                end_y=(0.0, 1.0),
            ),
            dict(size=(0.1, 0.3), t=(0.2, 0.8), rotation=(0.0, 0.5)),
        ),
        shape_trajectory_identifiers_test=(
            ("start_x", "start_y"),
            ("center_x", "center_y"),
        ),
        trajectory_function_test=("circle_", "line_"),
        trajectory_kwargs_test=(
            dict(center_x=(0.0, 1.0), center_y=(0.0, 1.0), radius=(0.1, 0.2)),
            dict(
                start_x=(0.0, 1.0),
                start_y=(0.0, 1.0),
                end_x=(0.0, 1.0),
                end_y=(0.0, 1.0),
            ),
        ),
        circular_position_test=(True, False),
    )

    context_train, target_train = next(mod.train_dataloader())
    context_val, target_val = next(mod.val_dataloader())
    context_test, target_test = next(mod.test_dataloader())

    save_gif_grid(
        (
            context_train["data"][0, :, 0],
            target_train["data"][0, :, 0],
            context_val["data"][0, :, 0],
            target_val["data"][0, :, 0],
            context_test["data"][0, :, 0],
            target_test["data"][0, :, 0],
        ),
        "test.gif",
        pad_value=1,
        nrow=2,
        scale_each=False,
        static_overlay=(True, False, True, False, True, False),
        cmap="magma",
    )
