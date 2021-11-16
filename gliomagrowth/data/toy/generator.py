import numpy as np
from collections import defaultdict
import inspect
from typing import Union, Optional, Iterable, Callable, Dict, Any, Tuple

from batchgenerators.dataloading import SlimDataLoaderBase
from gliomagrowth.util.util import matchcall, ct_to_transformable
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
        shape_function: Callable = shape.circle,
        shape_kwargs: Dict[str, Union[float, Any]] = dict(radius=0.25, image_size=64),
        shape_trajectory_identifiers: Tuple[str, str] = ("center_x", "center_y"),
        trajectory_function: Callable = trajectory.line_,
        trajectory_kwargs: Dict[str, Tuple[float, float]] = dict(
            start_x=(0.0, 1.0), start_y=(0.0, 1.0), end_x=(0.0, 1.0), end_y=(0.0, 1.0)
        ),
        num_objects: int = 1,
        min_length: float = 0.0,
        meta_trajectories: bool = False,
        circular_position: bool = False,
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
        self.shape_function = shape_function
        self.shape_kwargs = shape_kwargs.copy()
        self.shape_trajectory_identifiers = shape_trajectory_identifiers
        self.trajectory_function = trajectory_function
        self.trajectory_kwargs = trajectory_kwargs.copy()
        self.num_objects = num_objects
        self.min_length = min_length
        self.meta_trajectories = meta_trajectories
        self.circular_position = circular_position
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
                    if key not in self.trajectory_kwargs:
                        continue
                    val = self.trajectory_kwargs[key]
                    if hasattr(val, "__iter__") and not isinstance(val, str):
                        try:
                            val = np.random.uniform(*val)
                        except ValueError:
                            val = np.random.choice(val)
                    call_dict[key] = val
                if self.min_length > 0 and not self.circular_position:
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
                np.array(self.trajectory_function(num_max, **call_dict)).astype(
                    np.float32
                )
            )  # (N, len(shape_trajectory_identifiers))
            if not self.circular_position:
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

        coordinates = np.concatenate(coordinates, 1)  # (N, T*2)
        positions = np.concatenate(positions, 1)  # (N, T*1/2)

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
                    key not in self.shape_kwargs
                    or key in self.shape_trajectory_identifiers
                ):
                    continue
                val = self.shape_kwargs[key]

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
                        if not self.circular_position:
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
                for n, name in enumerate(self.shape_trajectory_identifiers):
                    call_dict[name] = c[n]
                img = matchcall(
                    self.shape_function,
                    **call_dict,
                    **joint_call_dict,
                ).astype(np.bool)
                data_context[i, 0][img] = 1

            for i in range(coordinates_target.shape[0]):
                call_dict = {}
                for key, val in params_target.items():
                    if key != "position":
                        call_dict[key] = params_target[key][-1][i]
                c = coordinates_target[i, 2 * t : 2 * (t + 1)]
                for n, name in enumerate(self.shape_trajectory_identifiers):
                    call_dict[name] = c[n]
                img = matchcall(
                    self.shape_function,
                    **call_dict,
                    **joint_call_dict,
                ).astype(np.bool)
                data_target[i, 0][img] = 1

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
        shape_params = inspect.signature(self.shape_function).parameters

        # everything we need to look at for trajectory function
        traj_params = inspect.signature(self.trajectory_function).parameters

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
