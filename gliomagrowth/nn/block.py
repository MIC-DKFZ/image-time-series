import torch
import torch.nn as nn
from typing import Callable, Union, Optional, Dict, Iterable, Type

from gliomagrowth.util.util import is_conv, match_to


class ConcatCoords(nn.Module):
    """Module that concatenates pixel coordinates to an input.

    Args:
        centered: if True, coordinates are [-0.5, 0.5] instead of [0, 1].

    """

    def __init__(self, centered: bool = True, **kwargs):

        super().__init__(**kwargs)
        self.centered = centered

    def forward(self, input_: torch.tensor, batch_dims: int = 1) -> torch.tensor:
        """Forward pass that concatenates coordinates to the input.

        Args:
            input_: Input with expected shape (..., C, SPACE).
            batch_dims: How many batch dimensions are there? The first batch_dims
                dimensions will be ignored, from the rest we infer the number of
                spatial dimensions.

        Returns:
            Input with concatenated coordinates, i.e. shape (..., C+d(SPACE), SPACE).

        """

        dim = input_.ndim - 1 - batch_dims
        coord_channels = []
        for i in range(dim):
            view = [
                1,
            ] * dim
            view[i] = -1
            repeat = list(input_.shape[(1 + batch_dims) :])
            repeat[i] = 1
            coord_channels.append(
                torch.linspace(
                    0.0 - 0.5 * int(self.centered),
                    1.0 - 0.5 * int(self.centered),
                    input_.shape[i + 1 + batch_dims],
                )
                .view(*view)
                .repeat(*repeat)
                .to(device=input_.device, dtype=input_.dtype)
            )
        coord_channels = torch.stack(coord_channels).unsqueeze(0)
        repeat = [
            1,
        ] * input_.ndim
        for i in range(batch_dims):
            repeat[i] = input_.shape[i]
        coord_channels = coord_channels.repeat(*repeat).contiguous()

        return torch.cat([input_, coord_channels], batch_dims)


class ConvModule(nn.Module):
    """Wrapper around Module with init methods for weights."""

    def __init__(self, *args, **kwargs):

        super().__init__()

    def init_weights(self, init_fn: Callable, *args, **kwargs):
        """Initialize weights with provided function.

        Args:
            init_fn: Function to initialize with, e.g. nn.init.kaiming_uniform_.

        """

        class init_(object):
            def __init__(self):
                self.fn = init_fn
                self.args = args
                self.kwargs = kwargs

            def __call__(self, module):
                if is_conv(type(module)):
                    module.weight = self.fn(module.weight, *self.args, **self.kwargs)

        _init_ = init_()
        self.apply(_init_)

    def init_bias(self, init_fn: Callable, *args, **kwargs):
        """Initialize bias with provided function.

        Args:
            init_fn: Function to initialize with, e.g. nn.init.constant_.

        """

        class init_(object):
            def __init__(self):
                self.fn = init_fn
                self.args = args
                self.kwargs = kwargs

            def __call__(self, module):
                if is_conv(type(module)) and module.bias is not None:
                    module.bias = self.fn(module.bias, *self.args, **self.kwargs)

        _init_ = init_()
        self.apply(_init_)


class InjectionConvEncoder(ConvModule):
    """A convolutional encoder that allows you to 'inject' another input.

    Obviously this can also be used as a regular conv encoder by setting
    injection_channels=0 (which is the default).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        depth: Number of blocks (separated by pooling).
        injection_depth: At what depth will the second input be injected?
            Integer or "last".
        injection_channels: Number of channels for the second input.
        block_depth: Number of (conv, dropout, norm, activation) in a block.
        num_feature_maps: Initial number of feature maps. Will be multiplied by
            feature_map_multiplier going deeper.
        feature_map_multiplier: Multiply the number of feature maps by this when going
            to the next depth.
        activation_op: Activation operator.
        activation_kwargs: Initialization options for activation.
        norm_op: Normalization operator.
        norm_kwargs: Initialization options for normalization.
        norm_depth: Use normalization only until this depth is reached.
        conv_op: Convolution operator.
        conv_kwargs: Initialization options for convolution.
        pool_op: Pooling operator.
        pool_kwargs: Initialization options for pooling.
        dropout_op: Dropout operator.
        dropout_kwargs: Initialization options for dropout.
        global_pool_op: Global pooling operator at the end.
        global_pool_kwargs: Initialization options for global pooling.
        coords: If this is activated, we concatenate coordinates at the beginning of
            each depth. Can also be an iterable of bools to specify each depth.
        coords_dim: Probably 2 or 3.

    """

    _default_activation_kwargs = dict(inplace=True)
    _default_norm_kwargs = dict()
    _default_conv_kwargs = dict(kernel_size=3, padding=1)
    _default_pool_kwargs = dict(kernel_size=2)
    _default_dropout_kwargs = dict()
    _default_global_pool_kwargs = dict()

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        depth: int = 4,
        injection_depth: Union[int, str] = "last",
        injection_channels: int = 0,
        block_depth: int = 2,
        num_feature_maps: int = 24,
        feature_map_multiplier: int = 2,
        activation_op: Optional[Type[nn.Module]] = nn.LeakyReLU,
        activation_kwargs: Optional[Dict] = None,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm2d,
        norm_kwargs: Optional[Dict] = None,
        norm_depth: Union[int, str] = 0,
        conv_op: Type[nn.Module] = nn.Conv2d,
        conv_kwargs: Optional[Dict] = None,
        pool_op: Type[nn.Module] = nn.AvgPool2d,
        pool_kwargs: Optional[Dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_kwargs: Optional[Dict] = None,
        global_pool_op: Optional[Type[nn.Module]] = nn.AdaptiveAvgPool2d,
        global_pool_kwargs: Optional[Dict] = None,
        coords: Union[bool, Iterable[bool]] = False,
        coords_dim: int = 2,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.injection_depth = (
            depth - 1 if injection_depth == "last" else injection_depth
        )
        self.injection_channels = injection_channels
        self.block_depth = block_depth
        self.num_feature_maps = num_feature_maps
        self.feature_map_multiplier = feature_map_multiplier

        self.activation_op = activation_op
        self.activation_kwargs = self._default_activation_kwargs.copy()
        if activation_kwargs is not None:
            self.activation_kwargs.update(activation_kwargs)

        self.norm_op = norm_op
        self.norm_kwargs = self._default_norm_kwargs.copy()
        if norm_kwargs is not None:
            self.norm_kwargs.update(norm_kwargs)
        self.norm_depth = depth if norm_depth == "full" else norm_depth

        self.conv_op = conv_op
        self.conv_kwargs = self._default_conv_kwargs.copy()
        if conv_kwargs is not None:
            self.conv_kwargs.update(conv_kwargs)

        self.pool_op = pool_op
        self.pool_kwargs = self._default_pool_kwargs.copy()
        if pool_kwargs is not None:
            self.pool_kwargs.update(pool_kwargs)

        self.dropout_op = dropout_op
        self.dropout_kwargs = self._default_dropout_kwargs.copy()
        if dropout_kwargs is not None:
            self.dropout_kwargs.update(dropout_kwargs)

        self.global_pool_op = global_pool_op
        self.global_pool_kwargs = self._default_global_pool_kwargs.copy()
        if global_pool_kwargs is not None:
            self.global_pool_kwargs.update(global_pool_kwargs)

        if not coords:
            self.coords = [
                False,
            ] * self.depth
        elif coords is True:
            self.coords = [
                True,
            ] * self.depth
        else:
            self.coords = coords
        self.coords_dim = coords_dim

        for d in range(self.depth):

            in_ = (
                self.in_channels
                if d == 0
                else self.num_feature_maps * (self.feature_map_multiplier ** (d - 1))
            )
            out_ = self.num_feature_maps * (self.feature_map_multiplier ** d)

            if d == self.injection_depth + 1:
                in_ += self.injection_channels

            layers = []
            if d > 0:
                layers.append(self.pool_op(**self.pool_kwargs))
            if self.coords[d]:
                layers.append(ConcatCoords())
                in_ += coords_dim
            for b in range(self.block_depth):
                current_in = in_ if b == 0 else out_
                layers.append(self.conv_op(current_in, out_, **self.conv_kwargs))
                if self.dropout_op is not None:
                    layers.append(self.dropout_op(**self.dropout_kwargs))
                if self.norm_op is not None and d < self.norm_depth:
                    layers.append(self.norm_op(out_, **self.norm_kwargs))
                if self.activation_op is not None:
                    layers.append(self.activation_op(**self.activation_kwargs))
            if d == self.depth - 1:
                current_conv_kwargs = self.conv_kwargs.copy()
                current_conv_kwargs["kernel_size"] = 1
                current_conv_kwargs["padding"] = 0
                current_conv_kwargs["bias"] = False
                layers.append(self.conv_op(out_, out_channels, **current_conv_kwargs))

            self.add_module("encode_{}".format(d), nn.Sequential(*layers))

        if self.global_pool_op is not None:
            self.add_module(
                "global_pool", self.global_pool_op(1, **self.global_pool_kwargs)
            )

    def forward(
        self, x: torch.tensor, injection: Optional[torch.tensor] = None
    ) -> torch.tensor:
        """Forward pass through encoder.

        Args:
            x: The regular input.
            injection: The input to be injected deeper in the encoder.
                Will automatically be expanded to appropriate shape.

        Returns:
            The encoder output.

        """

        for d in range(self.depth):
            x = self._modules["encode_{}".format(d)](x)
            if d == self.injection_depth and self.injection_channels > 0:
                injection = match_to(injection, x)
                x = torch.cat([x, injection], 1)
        if hasattr(self, "global_pool"):
            x = self.global_pool(x)

        return x


class MultiOutputInjectionConvEncoder(InjectionConvEncoder):
    """InjectionConvEncoder with the option to return outputs at different depths.

    Args:
        return_last: Return outputs from the last return_last depths. 1 means it's a
            regular encoder

    """

    def __init__(self, *args, return_last: int = 1, **kwargs):

        super().__init__(*args, **kwargs)
        self.return_last = return_last

    def forward(
        self, x: torch.tensor, injection: Optional[torch.tensor] = None
    ) -> Union[torch.tensor, Iterable[torch.tensor]]:
        """Forward pass through encoder.

        Args:
            x: The regular input.
            injection: The input to be injected deeper in the encoder.
                Will automatically be expanded to appropriate shape.

        Returns:
            Either a single tensor when return_last=1, otherwise a tuple with the
                last return_last outputs.

        """

        return_ = []

        for d in range(self.depth):
            if d == self.depth - 1:
                x = self._modules["encode_{}".format(d)][:-2](x)
                if self.return_last >= 2:
                    return_.append(x)
                x = self._modules["encode_{}".format(d)][-2:](x)
            else:
                x = self._modules["encode_{}".format(d)](x)
            if d == self.injection_depth and self.injection_channels > 0:
                injection = match_to(injection, x)
                x = torch.cat([x, injection], 1)
            if self.depth - d <= self.return_last - 1 and d < self.depth - 1:
                return_.append(x)
        if hasattr(self, "global_pool"):
            x = self.global_pool(x)
        return_.append(x)

        if len(return_) == 1:
            return return_[0]
        else:
            return tuple(return_)


class MultiOutputInjectionConvEncoder3D(MultiOutputInjectionConvEncoder):
    """MultiOutputInjectionConvEncoder, but default ops are updated to 3D versions."""

    def __init__(self, *args, **kwargs):

        update_kwargs = dict(
            norm_op=nn.InstanceNorm3d,
            conv_op=nn.Conv3d,
            pool_op=nn.AvgPool3d,
            global_pool_op=nn.AdaptiveAvgPool3d,
            coords_dim=3,
        )

        for (arg, val) in update_kwargs.items():
            if arg not in kwargs:
                kwargs[arg] = val

        super().__init__(*args, **kwargs)


class MultiInputConvDecoder(ConvModule):
    """A convolutional decoder that allows you to provide multiple inputs at different
    depths.

    Obviously this can also be used as a regular conv decoder by providing an int for
    in_channels.

    Args:
        in_channels: Input channels. If you provide N numbers, we assume inputs for the
            lowest N depths, starting with the highest. Assume you have a depth 5, so
            4 different resolutions. If you set this to (32, 64), we expect the first
            input (the lowest resolution) to have 64 channels, the following 32.
        out_channels: Output channels.
        depth: Number of blocks (separated by upsampling).
        block_depth: Number of (conv, dropout, norm, activation) in a block.
        num_feature_maps: Initial number of feature maps. Will be multiplied by
            feature_map_multiplier going deeper.
        feature_map_multiplier: Multiply the number of feature maps by this when going
            to the next depth.
        feature_map_multiplier_backwards: If this is active, the number of feature maps
            will decrease from input to output. Use this to combine
            MultiOutputInjectionConvEncoder and this to a U-Net.
        activation_op: Activation operator.
        activation_kwargs: Initialization options for activation.
        norm_op: Normalization operator.
        norm_kwargs: Initialization options for normalization.
        norm_depth: Use normalization only until this depth is reached.
        conv_op: Convolution operator.
        conv_kwargs: Initialization options for convolution.
        dropout_op: Dropout operator.
        dropout_kwargs: Initialization options for dropout.
        upsample_op: Upsampling operator.
        upsample_kwargs: Initialization options for upsampling.
        initial_upsample_op: Initial upsampling operator. Mostly useful if your input
            doesn't have a spatial resolution, but should have one.
        initial_upsample_kwargs: Initialization options for initial upsampling.
        output_activation_op: Final activation operator,
        output_activation_kwargs: Initialization options for final activation.
        coords: If this is activated, we concatenate coordinates at the beginning of
            each depth. Can also be an iterable of bools to specify each depth.
        coords_dim: Probably 2 or 3.

    """

    _default_activation_kwargs = dict(inplace=True)
    _default_norm_kwargs = dict()
    _default_conv_kwargs = dict(kernel_size=3, padding=1)
    _default_upsample_kwargs = dict(scale_factor=2)
    _default_dropout_kwargs = dict()
    _default_initial_upsample_kwargs = dict(size=(8, 8))
    _default_output_activation_kwargs = dict(dim=1)

    def __init__(
        self,
        in_channels: Union[int, Iterable[int]] = 6,
        out_channels: int = 1,
        depth: int = 4,
        block_depth: int = 2,
        num_feature_maps: int = 24,
        feature_map_multiplier: int = 2,
        feature_map_multiplier_backwards: bool = False,
        activation_op: Optional[Type[nn.Module]] = nn.LeakyReLU,
        activation_kwargs: Optional[Dict] = None,
        norm_op: Optional[Type[nn.Module]] = nn.InstanceNorm2d,
        norm_kwargs: Optional[Dict] = None,
        norm_depth: Union[int, str] = 0,
        conv_op: Type[nn.Module] = nn.Conv2d,
        conv_kwargs: Optional[Dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_kwargs: Optional[Dict] = None,
        upsample_op: Type[nn.Module] = nn.Upsample,
        upsample_kwargs: Optional[Dict] = None,
        initial_upsample_op: Optional[Type[nn.Module]] = nn.Upsample,
        initial_upsample_kwargs: Optional[Dict] = None,
        output_activation_op: Optional[Type[nn.Module]] = None,
        output_activation_kwargs: Optional[Dict] = None,
        coords: Union[bool, Iterable[bool]] = False,
        coords_dim: int = 2,
        **kwargs
    ):

        super().__init__(**kwargs)

        if hasattr(in_channels, "__iter__"):
            self.in_channels = [0,] * (
                depth - len(in_channels)
            ) + list(in_channels)
        else:
            self.in_channels = [0,] * (depth - 1) + [
                in_channels,
            ]
        if initial_upsample_op is not None:
            self.in_channels = [
                0,
            ] + self.in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.block_depth = block_depth
        self.num_feature_maps = num_feature_maps
        self.feature_map_multiplier = feature_map_multiplier
        self.feature_map_multiplier_backwards = feature_map_multiplier_backwards

        self.activation_op = activation_op
        self.activation_kwargs = self._default_activation_kwargs.copy()
        if activation_kwargs is not None:
            self.activation_kwargs.update(activation_kwargs)

        self.norm_op = norm_op
        self.norm_kwargs = self._default_norm_kwargs.copy()
        if norm_kwargs is not None:
            self.norm_kwargs.update(norm_kwargs)
        self.norm_depth = depth if norm_depth == "full" else norm_depth

        self.conv_op = conv_op
        self.conv_kwargs = self._default_conv_kwargs.copy()
        if conv_kwargs is not None:
            self.conv_kwargs.update(conv_kwargs)

        self.upsample_op = upsample_op
        self.upsample_kwargs = self._default_upsample_kwargs.copy()
        if upsample_kwargs is not None:
            self.upsample_kwargs.update(upsample_kwargs)

        self.dropout_op = dropout_op
        self.dropout_kwargs = self._default_dropout_kwargs.copy()
        if dropout_kwargs is not None:
            self.dropout_kwargs.update(dropout_kwargs)

        self.initial_upsample_op = initial_upsample_op
        self.initial_upsample_kwargs = self._default_initial_upsample_kwargs.copy()
        if initial_upsample_kwargs is not None:
            self.initial_upsample_kwargs.update(initial_upsample_kwargs)

        self.output_activation_op = output_activation_op
        self.output_activation_kwargs = self._default_output_activation_kwargs.copy()
        if output_activation_kwargs is not None:
            self.output_activation_kwargs.update(output_activation_kwargs)

        if not coords:
            self.coords = [
                False,
            ] * self.depth
        elif coords is True:
            self.coords = [
                True,
            ] * self.depth
        else:
            self.coords = coords
        self.coords_dim = coords_dim

        if self.initial_upsample_op is not None:
            self.add_module(
                "initial_upsample",
                self.initial_upsample_op(**self.initial_upsample_kwargs),
            )

        for d in range(self.depth):

            in_ = self.in_channels[-(d + 1)]
            if self.feature_map_multiplier_backwards:
                if d > 0:
                    in_ += self.num_feature_maps * (
                        self.feature_map_multiplier ** (self.depth - d)
                    )
                out_ = self.num_feature_maps * (
                    self.feature_map_multiplier ** (self.depth - d - 1)
                )
            else:
                if d > 0:
                    in_ += self.num_feature_maps * (
                        self.feature_map_multiplier ** (d - 1)
                    )
                out_ = self.num_feature_maps * (self.feature_map_multiplier ** d)

            layers = []
            if d > 0:
                layers.append(self.upsample_op(**self.upsample_kwargs))
            if self.coords[d]:
                layers.append(ConcatCoords())
                in_ += coords_dim
            for b in range(self.block_depth):
                current_in = in_ if b == 0 else out_
                layers.append(self.conv_op(current_in, out_, **self.conv_kwargs))
                if self.dropout_op is not None:
                    layers.append(self.dropout_op(**self.dropout_kwargs))
                if self.norm_op is not None and d < self.norm_depth:
                    layers.append(self.norm_op(out_, **self.norm_kwargs))
                if self.activation_op is not None:
                    layers.append(self.activation_op(**self.activation_kwargs))
            if d == self.depth - 1:
                current_conv_kwargs = self.conv_kwargs.copy()
                current_conv_kwargs["kernel_size"] = 1
                current_conv_kwargs["padding"] = 0
                current_conv_kwargs["bias"] = False
                layers.append(
                    self.conv_op(out_, self.out_channels, **current_conv_kwargs)
                )

            self.add_module(
                "decode_{}".format(self.depth - 1 - d), nn.Sequential(*layers)
            )

        if self.output_activation_op is not None:
            self.add_module(
                "output_activation",
                self.output_activation_op(**self.output_activation_kwargs),
            )

    def forward(self, *inputs: torch.tensor) -> torch.tensor:
        """Decoder forward pass.

        Args:
            inputs: We expect the same number of inputs as the number of in_channel
            parameters we provided.

        Returns:
            Decoder output.

        """

        x = inputs[-1]

        if hasattr(self, "initial_upsample"):
            x = self.initial_upsample(x)

        for d in range(self.depth):
            if 0 < d < len(inputs):
                x = torch.cat((x, inputs[-(d + 1)]), 1)
            x = self._modules["decode_{}".format(self.depth - 1 - d)](x)

        if hasattr(self, "output_activation"):
            x = self.output_activation(x)

        return x


class MultiInputConvDecoder3D(MultiInputConvDecoder):
    """MultiInputConvDecoder, but default ops are updated to 3D versions."""

    def __init__(self, *args, **kwargs):

        update_kwargs = dict(
            norm_op=nn.InstanceNorm3d,
            conv_op=nn.Conv3d,
            coords_dim=3,
        )

        for (arg, val) in update_kwargs.items():
            if arg not in kwargs:
                kwargs[arg] = val

        super().__init__(*args, **kwargs)