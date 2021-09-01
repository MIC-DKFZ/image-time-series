import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Iterable, Union, Type, Tuple
import warnings

from gliomagrowth.util.util import (
    match_shapes,
    stack_batch,
    unstack_batch,
    tensor_to_loc_scale,
)


class ImageProcess(nn.Module):
    """
    A Neural Process, but the queries can simultaneously be image and vector valued.

    We further allow multi-level representations, e.g. when using
    MultiOutputInjectionConvEncoder and MultiInputConvDecoder.

    Args:
        context_encoder_op: Encoder for the context. This is also the prior encoder
            in a variational model.
        decoder_op: Decoder. Make sure the
            decoder accommodates the correct input size depending on
            availability of context and target representations.
        target_encoder_op: Encoder for the target images.
            Only necessary when working with queries that contain images.
        context_encoder_kwargs: Instantiate context_encoder_op with this.
        decoder_kwargs: Instantiate decoder_op with this.
        target_encoder_kwargs: Instantiate target_encoder_op with this.
        variational: Make the model variational.
        posterior_encoder_op: Encoder for the posterior. You need this if you want to
            have different contexts and targets, e.g. when you only have queries and
            images in the context, but want to predict target segmentations.
            Will automatically be used if provided.
        posterior_encoder_kwargs: Instantiate posterior_encoder_op with this.

    """

    def __init__(
        self,
        context_encoder_op: Type[nn.Module],
        decoder_op: Type[nn.Module],
        target_encoder_op: Optional[Type[nn.Module]] = None,
        context_encoder_kwargs: Optional[dict] = None,
        decoder_kwargs: Optional[dict] = None,
        target_encoder_kwargs: Optional[dict] = None,
        variational: bool = False,
        posterior_encoder_op: Optional[Type[nn.Module]] = None,
        posterior_encoder_kwargs: Optional[dict] = None,
        *args,
        **kwargs
    ):

        super().__init__()

        if context_encoder_kwargs is None:
            context_encoder_kwargs = {}
        self.context_encoder = context_encoder_op(**context_encoder_kwargs)

        if decoder_kwargs is None:
            decoder_kwargs = {}
        self.decoder = decoder_op(**decoder_kwargs)

        if target_encoder_op is not None:
            if target_encoder_kwargs is None:
                target_encoder_kwargs = {}
            self.target_encoder = target_encoder_op(**target_encoder_kwargs)

        if posterior_encoder_op is not None:
            if posterior_encoder_kwargs is None:
                posterior_encoder_kwargs = {}
            self.posterior_encoder = posterior_encoder_op(**posterior_encoder_kwargs)
            if not variational:
                warnings.warn(
                    "Constructed posterior encoder, but the model is deterministic,\
                    so it won't be used."
                )

        self.variational = variational
        self.posterior = None
        self.prior = None

    def aggregate(
        self,
        context_query: torch.Tensor,
        target_query: torch.Tensor,
        context_representation: Union[torch.Tensor, Iterable[torch.Tensor]],
        target_representation: Union[torch.Tensor, Iterable[torch.Tensor], None] = None,
    ) -> torch.Tensor:
        """
        Aggregate representations. This implementation averages along
        first axis.

        Args:
            context_query: Shape (B, N, Cq).
                Not used in this implementation. Will be necessary
                for attention based aggregation.
            target_query: Shape (B, M, Cq).
            context_representation: Shape (B, N, Ci, ...).
                Can also be a list or tuple of tensors with varying
                number of channels and spatial size.
            target_representation: Shape (B, M, Ci, ...).
                Can also be a list or tuple of tensors with varying
                number of channels and spatial size. Only used when
                query images need to be encoded.

        Returns:
            Average representation.
                Will be a list if input is a list or tuple.

        """

        if torch.is_tensor(context_representation):
            context_representation = [
                context_representation,
            ]
        if torch.is_tensor(target_representation):
            target_representation = [
                target_representation,
            ]
        if target_representation is None:
            target_representation = [
                None,
            ] * len(context_representation)

        context_representation = [
            r.mean(1, keepdim=True) for r in context_representation
        ]

        for r, rep in enumerate(context_representation[:-1]):
            concat = match_shapes(
                rep, target_representation[r], target_query, ignore_axes=2
            )
            if target_representation[r] is None:
                concat = concat[:1]
            else:
                concat = concat[:2]
            context_representation[r] = torch.cat(concat, 2)

        context_representation[-1] = torch.cat(
            match_shapes(
                context_representation[-1],
                target_representation[-1],
                target_query,
                ignore_axes=2,
            ),
            2,
        )

        return context_representation

    def encode_context(
        self,
        context_query: torch.Tensor,
        context_image: Optional[torch.Tensor] = None,
        context_seg: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        """
        Use the context encoder (or a provided one) to encode a representation.

        Args:
            context_query: Shape (B, N, Cq)
            context_image: Shape (B, N, Cimg, ...). Note that either context_image or
                context_seg must be different from None.
            context_seg: Shape (B, N, Cs, ...). Note that either context_image or
                context_seg must be different from None.
            encoder: An encoder module. If None, this defaults to self.context_encoder.
                This functionality is mainly here so we can also use the code for
                posterior encoding.

        Returns:
            Shape (B, N, Cr, ...). Can also be a list!

        """

        if context_image is None and context_seg is None:
            raise ValueError("context_image and context_seg must not both be None!")
        if encoder is None:
            encoder = self.context_encoder

        B = context_query.shape[0]

        input_ = torch.cat(
            match_shapes(context_query, context_image, context_seg, ignore_axes=2), 2
        )
        input_ = stack_batch(input_)
        output = encoder(input_)
        if torch.is_tensor(output):
            output = unstack_batch(output, B)
        else:
            output = [unstack_batch(o, B) for o in output]
        return output

    def encode_target(
        self, target_query: torch.Tensor, target_image: torch.Tensor
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        """
        Use the target encoder to encode a representation.

        Args:
            target_query: Shape (B, M, Cq)
            target_image: Shape (B, N, Cimg, ...)

        Returns:
            Shape (B, N, Cr, ...). Can also be a list!

        """

        if self.target_encoder is None:
            raise AttributeError("target_encoder is None, so we can't encode anything!")

        B = target_query.shape[0]

        input_ = torch.cat(match_shapes(target_query, target_image, ignore_axes=2), 2)
        input_ = stack_batch(input_)
        output = self.target_encoder(input_)
        if torch.is_tensor(output):
            output = unstack_batch(output, B)
        else:
            output = [unstack_batch(o, B) for o in output]
        return output

    def decode(
        self, representation: Union[torch.Tensor, Iterable[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Decode an aggregated representation.

        Args:
            representation: Shape (B, M, Cr, ...).

        Returns:
            Output of 'decoder', shape (B, M, Cout, ...).

        """

        if torch.is_tensor(representation):
            representation = [
                representation,
            ]

        B = representation[0].shape[0]
        representation = [stack_batch(r) for r in representation]
        representation = self.decoder(*representation)
        return unstack_batch(representation, B)

    def encode_prior(
        self,
        representation: Union[torch.Tensor, Iterable[torch.Tensor]],
        save: bool = True,
    ) -> torch.distributions.Normal:
        """
        Encode prior from representations.

        Args:
            representation: Sequence or tensor, shape (B, N, C, ...)
            save: Save the prior in self.prior

        Returns:
            The prior, which is also saved in self.prior
        """

        if torch.is_tensor(representation):
            prior = representation
        else:
            prior = representation[-1]
        prior = prior.mean(1, keepdim=True)
        try:
            prior = tensor_to_loc_scale(prior, torch.distributions.Normal, axis=2)
        except ValueError as e:
            print(prior.shape)
            print(torch.any(torch.isnan(prior)))
            print(prior)
            raise e
        if save:
            self.prior = prior

        return prior

    def encode_posterior(
        self,
        representation: Union[torch.Tensor, Iterable[torch.Tensor]],
        save: bool = True,
    ) -> torch.distributions.Normal:
        """
        Encode posterior from representations.

        Args:
            representation: Sequence or tensor, shape (B, N, C, ...)
            save: Save the prior in self.prior

        Returns:
            The posterior, which is also saved in self.posterior

        """

        if torch.is_tensor(representation):
            posterior = representation
        else:
            posterior = representation[-1]
        posterior = posterior.mean(1, keepdim=True)
        posterior = tensor_to_loc_scale(posterior, torch.distributions.Normal, axis=2)
        if save:
            self.posterior = posterior

        return posterior

    def forward(
        self,
        context_query: torch.Tensor,
        target_query: torch.Tensor,
        context_image: Optional[torch.Tensor] = None,
        context_seg: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        target_seg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass in the Segmentation Process.

        Args:
            context_query: Shape (B, N, Cq).
            target_query: Shape (B, M, Cq).
            context_image: Shape (B, N, Cimg, ...). Note that either context_image or
                context_seg must be different from None.
            context_seg: Shape (B, N, Cs, ...). Note that either context_image or
                context_seg must be different from None.
            target_image: Shape (B, M, Cimg, ...).
            target_seg: Shape (B, M, Cs, ...).

        Returns:
            Output of 'decoder', shape (B, M, Cout, ...)

        """

        # encode context
        # returns a tuple of tensors with shape (B, N, Ci, ...)
        context_representation = self.encode_context(
            context_query, context_image, context_seg
        )

        # encode target if there's something to encode
        # returns a tuple of tensors with shape (B, M, Ci, ...)
        if target_image is not None and hasattr(self, "target_encoder"):
            target_representation = self.encode_target(target_query, target_image)
        else:
            target_representation = None

        if self.variational:

            prior = self.encode_prior(context_representation)

            if self.training:

                if target_seg is None:
                    raise ValueError(
                        "target_seg must not be None for training in variational mode."
                    )

                encoder = getattr(self, "posterior_encoder", None)
                posterior = self.encode_context(
                    target_query, target_image, target_seg, encoder=encoder
                )
                posterior = self.encode_posterior(posterior)
                sample = posterior.rsample()

            else:

                sample = prior.loc

            # expand again, because aggregate will take mean
            if torch.is_tensor(context_representation):
                sample = torch.repeat_interleave(
                    sample, context_representation.shape[1], 1
                )
                context_representation = sample
            else:
                sample = torch.repeat_interleave(
                    sample, context_representation[-1].shape[1], 1
                )
                context_representation[-1] = sample

        # aggregate (i.e. create the input for the decoder)
        # in variational mode, this writes into the prior attribute
        context_representation = self.aggregate(
            context_query,
            target_query,
            context_representation,
            target_representation,
        )

        # decode
        return self.decode(context_representation)

    def sample(
        self,
        n_samples,
        context_query: torch.Tensor,
        target_query: torch.Tensor,
        context_image: Optional[torch.Tensor] = None,
        context_seg: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sampling pass in the Segmentation Process.

        Args:
            context_query: Shape (B, N, Cq).
            target_query: Shape (B, M, Cq).
            context_image: Shape (B, N, Cimg, ...). Note that either context_image or
                context_seg must be different from None.
            context_seg: Shape (B, N, Cs, ...). Note that either context_image or
                context_seg must be different from None.
            target_image: Shape (B, M, Cimg, ...).

        Returns:
            Output of 'decoder', shape (B, M, Cout, ...)

        """

        if not self.variational:
            raise AttributeError("Can only sample in variational mode.")

        # encode context
        # returns a tuple of tensors with shape (B, N, Ci, ...)
        context_representation = self.encode_context(
            context_query, context_image, context_seg
        )

        # encode target if there's something to encode
        # returns a tuple of tensors with shape (B, M, Ci, ...)
        if target_image is not None:
            target_representation = self.encode_target(target_query, target_image)
        else:
            target_representation = None

        prior = self.encode_prior(context_representation, save=False)

        samples = []
        while len(samples) < n_samples:

            sample = prior.sample()
            sample = torch.repeat_interleave(sample, context_query.shape[1], 1)

            if torch.is_tensor(context_representation):

                samples.append(
                    self.decode(
                        self.aggregate(
                            context_query,
                            target_query,
                            sample,
                            target_representation,
                        )
                    )
                )

            else:

                samples.append(
                    self.decode(
                        self.aggregate(
                            context_query,
                            target_query,
                            context_representation[:-1]
                            + [
                                sample,
                            ],
                            target_representation,
                        )
                    )
                )

        return torch.stack(samples)


class AttentiveImageProcess(ImageProcess):
    """ImageProcess that uses attention for aggregation.

    We're generally assuming representations at multiple scales/levels, otherwise the
    arguments wouldn't make much sense ;) If you provide fewer attention mechanisms than
    there are representation scales, attention will be used on the deepest/last ones.

    Args:
        attention_op: Attention module, e.g. torch.nn.MultiheadAttention. You can also
            specify different modules for different levels.
        attention_kwargs: Initialization arguments for attention. If you want different
            arguments for different levels, each keyword can be a tuple or list of
            arguments.
        num_attention: Number of attention modules. attention_op will be repeated this
            many times if it's not iterable.
        global_sum: Use summation for the deepest level (like a normal Neural Process).
        scaleup_query: Queries/keys will be scaled up to appropriate spatial resolution
            if this is active.
        downsample: Downsample representations. This saves memory for attention.
        upsample: Upsample again after downsampling.
        interpolation_mode: Interpolation mode for downsampling and upsampling.
        downsample_size: Downsample to this size. Can be iterable to use different sizes
            at different representation scales.
        higher_level_attention_op: Another attention mechanism to be used for the
            highest scales (i.e. largest spatial resolution). Example: You want to use
            full spatio-temporal attention for lower resolutions, but only temporal
            at high resolutions. If this is None, averaging will be used instead (not
            recommended).
        higher_level_attention_kwargs: Initialization arguments for
            higher_level_attention.

    """

    def __init__(
        self,
        attention_op: Union[Type[nn.Module], Iterable[Type[nn.Module]]],
        attention_kwargs: Optional[dict] = None,
        num_attention: int = 1,
        global_sum: bool = True,
        scaleup_query: bool = False,
        downsample: bool = False,
        upsample: bool = False,
        interpolation_mode: str = "bilinear",
        downsample_size: Union[int, Iterable[int]] = 8,
        higher_level_attention_op: Optional[Type[nn.Module]] = None,
        higher_level_attention_kwargs: Optional[dict] = None,
        **kwargs
    ):

        super().__init__(**kwargs)

        # first we create the individual dicts to instantiate the attention ops with.
        # items in attention_kwargs that are lists/tuples will be split!
        if isinstance(attention_op, type):
            attention_op = [
                attention_op,
            ] * num_attention
        if attention_kwargs is None:
            attention_kwargs = {}
        attention_kwargs_list = []
        for i in range(len(attention_op)):
            kw = {}
            for key, val in attention_kwargs.items():
                if isinstance(val, (list, tuple)):
                    kw[key] = val[i]
                else:
                    kw[key] = val
            attention_kwargs_list.append(kw)

        self.attention = [
            op(**kw) for op, kw in zip(attention_op, attention_kwargs_list)
        ]
        self.num_attention = len(self.attention)
        self.global_sum = global_sum
        self.scaleup_query = scaleup_query
        self.downsample = downsample
        self.upsample = upsample
        self.interpolation_mode = interpolation_mode
        self.downsample_size = downsample_size

        # so .to(device) works
        for m, module in enumerate(self.attention):
            self.add_module("attention_{}".format(m), module)

        if higher_level_attention_op is not None:
            if higher_level_attention_kwargs is None:
                higher_level_attention_kwargs = {}
            self.add_module(
                "higher_level_attention",
                higher_level_attention_op(**higher_level_attention_kwargs),
            )

    def optional_downsample(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Optionally downsample input, depending on configuration.

        Args:
            x: Shape (B, N, C, spatial_dims).

        Returns:
            The resampled input and and indicator if resampling actually took place.

        """

        if not self.downsample:

            return x, False

        else:

            spatial_shape = x.shape[3:]
            if not hasattr(self.downsample_size, "__iter__"):
                downsample_size = [self.downsample_size] * len(spatial_shape)
            else:
                downsample_size = self.downsample_size

            target_shape = []
            need_to_downsample = False
            for s, ss in enumerate(spatial_shape):
                if ss > downsample_size[s]:
                    need_to_downsample = True
                target_shape.append(min(ss, downsample_size[s]))

            if need_to_downsample:

                B = x.shape[0]
                x = stack_batch(x)
                x = F.interpolate(
                    x,
                    size=target_shape,
                    mode=self.interpolation_mode,
                    align_corners=False,
                )
                x = unstack_batch(x, B)

            return x, need_to_downsample

    def aggregate(
        self,
        context_query: torch.Tensor,
        target_query: torch.Tensor,
        context_representation: Union[torch.Tensor, Iterable[torch.Tensor]],
        target_representation: Union[torch.Tensor, Iterable[torch.Tensor], None] = None,
    ) -> Iterable[torch.Tensor]:
        """
        Aggregate representations. This implementation uses attention over queries/keys.

        Args:
            context_query: Shape (B, N, Cq).
                These are the keys for the attention mechanism.
            target_query: Shape (B, M, Cq).
                These are the queries for the attention mechanism.
            context_representation: Shape (B, N, Ci, ...).
                These are the values for the attention mechanism.
                Can also be a list or tuple of tensors with varying
                number of channels and spatial size.
            target_representation: Shape (B, M, Ci, ...).
                Will just be concatenated to the output in this implementation.

        Returns:
            Average representation. Will be a list.

        """

        if torch.is_tensor(context_representation):
            context_representation = [
                context_representation,
            ]
        if torch.is_tensor(target_representation):
            target_representation = [
                target_representation,
            ]
        if target_representation is None:
            target_representation = [
                None,
            ] * len(context_representation)

        attention = list(self.attention)
        while len(attention) < len(context_representation) - int(self.global_sum):
            attention = [None] + attention

        for r, rep in enumerate(context_representation):

            trep = target_representation[r]
            try:
                att = attention[r]
            except IndexError:
                att = None

            # regular summation in the deepest level if desired
            # also used if there is no attention mechanism
            if (r == len(context_representation) - 1 and self.global_sum) or (
                att is None and not hasattr(self, "higher_level_attention")
            ):

                rep = rep.mean(1, keepdim=True)
                concat = [rep]
                if trep is not None:
                    concat.append(trep)
                concat.append(target_query)
                concat = match_shapes(*concat, ignore_axes=2)
                context_representation[r] = torch.cat(concat, 2)

            elif att is None:

                context_representation[r] = self.higher_level_attention(
                    target_query, context_query, rep
                )[0]
                context_representation[r] = torch.cat(
                    match_shapes(
                        context_representation[r], target_query, ignore_axes=2
                    ),
                    2,
                )

            else:

                # downsample if necessary
                rep_original_shape = rep.shape
                rep, did_downsample = self.optional_downsample(rep)

                # if there is no target_representation, we use target_query and
                # context_query for attention
                if trep is None and not self.scaleup_query:

                    context_representation[r] = att(target_query, context_query, rep)[0]

                # if there is a target_representation, we concatenate target_query and
                # context_query to the respective representations and use the resulting
                # tensors for attention
                else:

                    # if there is no target representation, we just scale up the query
                    if trep is None:

                        trep = match_shapes(rep, target_query, ignore_axes=(1, 2))[1]

                    else:

                        trep, _ = self.optional_downsample(trep)
                        trep = torch.cat(
                            match_shapes(trep, target_query, ignore_axes=2), 2
                        )

                    rep = torch.cat(match_shapes(rep, context_query, ignore_axes=2), 2)
                    rep = att(trep, rep, rep)[0]

                    if self.upsample and did_downsample:
                        rep = stack_batch(rep)
                        rep = F.interpolate(
                            rep,
                            size=rep_original_shape[3:],
                            mode=self.interpolation_mode,
                            align_corners=False,
                        )
                        rep = unstack_batch(rep, rep_original_shape[0])

                    context_representation[r] = rep

        return context_representation