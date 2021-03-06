import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from typing import Optional, Any, Tuple, Dict, Union, List, Iterable

import gliomagrowth as gg
from gliomagrowth.util.util import (
    nn_module_lookup,
    make_onehot,
    stack_batch,
    unstack_batch,
)
from gliomagrowth.util.lightning import (
    VisdomLogger,
    make_default_parser,
    run_experiment,
    str2bool,
    DictArgument,
)
from gliomagrowth.nn import loss as customloss
from gliomagrowth.nn.block import MultiOutputInjectionConvEncoder, MultiInputConvDecoder
from gliomagrowth.nn.attention import MultiheadAttention
from gliomagrowth.nn.neuralprocess import AttentiveImageProcess
from gliomagrowth.eval.metrics import dice


class ContinuousSegmentation(pl.LightningModule):
    """Toy experiment to test segmentation interpolation.

    Args:
        num_classes: Number of classes including background.
        criterion_task: Loss for the output predictions.
        criterion_task_reduction: Reduction argument for criterion_task.
        criterion_task_onehot: Activate this if the target segmentation needs to be
            onehot encoded.
        criterion_latent: Loss for the latent space.
        criterion_latent_reduction: Reduction argument for latent loss.
        criterion_latent_weight: Weight for latent loss.
        optimizer: All modules are wrapped into one object and use the same optimizer.
        learning_rate: Initial learning rate.
        step_lr: Use StepLR scheduler. Mutually exclusive with reduce_lr_on_plateau.
        step_lr_gamma: Gamma argument for StepLR. Multiplies learning rate by this
            factor every epoch.
        reduce_lr_on_plateau: Use ReduceLROnPlateau scheduler. Mutually exclusive with
            StepLR.
        reduce_lr_factor: Learning rate factor for ReduceLROnPlateau.
        reduce_lr_patience: Patience for ReduceLROnPlateau.
        representation_channels: Size of the latent representation, i.e. the deepest
            output of the encoder.
        model_global_sum: Averages over deepest representations (instead of attention).
        model_spatial_attention: Use this many spatio-temporal attention mechanisms.
            These are used in the lowest possible levels with the lowest spatial
            resolution.
        model_temporal_attention: Use "regular" attention (over time). Applied to
            representations with higher spatial resolution. If model_spatial_attention
            and model_temporal_attention together are smaller than the number of
            representations returned by the encoder, the highest spatial resolutions
            are averaged instead.
        model_att_embed_dim: Embedding dimension for attention.
        model_att_heads: Number of attention heads.
        model_depth: Encoder and decoder depth (= number of blocks).
        model_block_depth: Encoder and decoder block depth.
        model_feature_maps: Initial number of feature maps. Multiplied by
            model_feature_map_multiplier after each block.
        model_feature_map_multiplier: Multiply feature map size by this after a block.
        model_activation: Activation function.
        model_activation_kwargs: Initialization arguments for activation.
        model_output_activation: Output activation function.
        model_output_activation_kwargs: Initialization arguments for output activation.
        model_norm: Normalization operator.
        model_norm_kwargs: Initialization arguments for normalization.
        model_norm_depth: Only use normalization in the first N encoder blocks.
        model_norm_depth_decoder: Only use normalization in the first N decoder blocks.
        model_pool: Pooling operator
        model_pool_kwargs: Initialization arguments for pooling.
        model_upsample: Upsampling operator.
        model_upsample_kwargs: Initialization arguments for upsampling.
        model_initial_upsample: Initial upsampling operator.
        model_initial_upsample_kwargs: Initialization arguments for initial upsampling.
        model_dropout: Dropout operator.
        model_dropout_kwargs: Initialization arguments for dropout.
        model_global_pool: Global pooling operator. Applied before last encoder output.
        model_global_pool_kwargs: Initialization arguments for global pooling.
        model_use_coords: Concatenate coordinates at each depth.

    """

    def __init__(
        self,
        num_classes: int = 2,  # incl. background
        criterion_task: str = "crossentropydiceloss",
        criterion_task_reduction: str = "mean",
        criterion_task_onehot: str = True,
        criterion_latent: str = "kldivergence",
        criterion_latent_reduction: str = "mean",
        criterion_latent_weight: float = 0.01,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        step_lr: bool = False,
        step_lr_gamma: float = 0.99,  # every epoch
        reduce_lr_on_plateau: bool = False,
        reduce_lr_factor: float = 0.1,
        reduce_lr_patience: int = 10,
        representation_channels: int = 128,
        model_global_sum: bool = True,
        model_spatial_attention: int = 2,
        model_temporal_attention: int = 0,
        model_att_embed_dim: int = 64,
        model_att_heads: int = 4,
        model_depth: int = 5,
        model_block_depth: int = 1,
        model_feature_maps: int = 12,
        model_feature_map_multiplier: int = 2,
        model_activation: str = "leakyrelu",
        model_activation_kwargs: Optional[dict] = None,
        model_output_activation: str = "softmax",
        model_output_activation_kwargs: Optional[dict] = None,
        model_norm: str = "instancenorm",
        model_norm_kwargs: Optional[dict] = None,
        model_norm_depth: int = 1,
        model_norm_depth_decoder: int = 0,
        model_pool: str = "avgpool",
        model_pool_kwargs: Optional[dict] = None,
        model_upsample: str = "convtranspose",
        model_upsample_kwargs: Optional[dict] = dict(
            kernel_size=4, stride=2, padding=1
        ),
        model_initial_upsample: str = "upsample",
        model_initial_upsample_kwargs: Optional[dict] = None,
        model_dropout: Optional[str] = None,
        model_dropout_kwargs: Optional[dict] = None,
        model_global_pool: str = "adaptiveavgpool",
        model_global_pool_kwargs: Optional[dict] = None,
        model_use_coords: bool = True,
        **kwargs
    ):

        super().__init__()

        self.save_hyperparameters()
        self.model = self.make_model(self.hparams)
        self.criterion_task = nn_module_lookup(criterion_task, 2, customloss)(
            reduction=self.hparams.criterion_task_reduction
        )
        self.criterion_latent = nn_module_lookup(criterion_latent, 2, customloss)()

        self.learning_rate = self.hparams.learning_rate

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]],
        Dict,
    ]:

        optimizer = nn_module_lookup(self.hparams.optimizer, 0, torch.optim)
        optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        if self.hparams.step_lr and self.hparams.reduce_lr_on_plateau:
            raise ValueError("Please turn on only one scheduler!")
        elif self.hparams.step_lr:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 1, self.hparams.step_lr_gamma
            )
            return [optimizer], [scheduler]
        elif self.hparams.reduce_lr_on_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.hparams.reduce_lr_factor,
                patience=self.hparams.reduce_lr_patience,
                verbose=True,
            )
            return dict(
                optimizer=optimizer, lr_scheduler=scheduler, monitor="val_loss_total"
            )
        else:
            return optimizer

    @staticmethod
    def make_model(hparams: argparse.Namespace) -> AttentiveImageProcess:
        """Construct model from hparams object.

        Args:
            hparams: mymodule.save_hyperparameters() -> mymodule.hparams

        Returns:
            An instance of AttentiveImageProcess.

        """

        # start with some basic checks
        if (
            hparams.model_spatial_attention + hparams.model_temporal_attention
            > hparams.model_depth
        ):
            raise ValueError(
                "Model depth ({}) must be at least the sum of spatial and temporal attention connections ({} + {}).".format(
                    hparams.model_depth,
                    hparams.model_spatial_attention,
                    hparams.model_temporal_attention,
                )
            )

        # look up ops from strings
        activation_op = nn_module_lookup(hparams.model_activation, 2)
        output_activation_op = nn_module_lookup(hparams.model_output_activation, 2)
        norm_op = nn_module_lookup(hparams.model_norm, 2)
        pool_op = nn_module_lookup(hparams.model_pool, 2)
        upsample_op = nn_module_lookup(hparams.model_upsample, 2)
        initial_upsample_op = nn_module_lookup(hparams.model_initial_upsample, 2)
        global_pool_op = nn_module_lookup(hparams.model_global_pool, 2)
        dropout_op = nn_module_lookup("dropout", 2) if hparams.model_dropout else None
        conv_op = nn_module_lookup("conv", 2)

        # configure encoder
        context_encoder_op = MultiOutputInjectionConvEncoder
        context_encoder_kwargs = dict(
            return_last=1
            + hparams.model_spatial_attention
            + hparams.model_temporal_attention,
            in_channels=2 * hparams.num_classes - 1,
            out_channels=2 * hparams.representation_channels,
            depth=hparams.model_depth,
            block_depth=hparams.model_block_depth,
            num_feature_maps=hparams.model_feature_maps,
            feature_map_multiplier=hparams.model_feature_map_multiplier,
            activation_op=activation_op,
            activation_kwargs=hparams.model_activation_kwargs,
            norm_op=norm_op,
            norm_kwargs=hparams.model_norm_kwargs,
            norm_depth=hparams.model_norm_depth,
            conv_op=conv_op,
            conv_kwargs=None,
            pool_op=pool_op,
            pool_kwargs=hparams.model_pool_kwargs,
            dropout_op=dropout_op,
            dropout_kwargs=hparams.model_dropout_kwargs,
            global_pool_op=global_pool_op,
            global_pool_kwargs=hparams.model_global_pool_kwargs,
            coords=hparams.model_use_coords,
            coords_dim=2,
        )

        # configure decoder
        in_channels = [hparams.representation_channels + hparams.num_classes - 1]
        for i in range(hparams.model_spatial_attention):
            in_channels.append(hparams.model_att_embed_dim)
        for i in range(hparams.model_temporal_attention):
            fmaps = (
                hparams.model_feature_maps
                * hparams.model_feature_map_multiplier
                ** (hparams.model_depth - hparams.model_spatial_attention - i - 1)
                + hparams.num_classes
                - 1
            )
            in_channels.append(fmaps)
        in_channels = list(reversed(in_channels))
        if hparams.model_initial_upsample_kwargs is None:
            initial_upsample_kwargs = dict(size=(2 ** (7 - hparams.model_depth),) * 2)
        else:
            initial_upsample_kwargs = hparams.model_initial_upsample_kwargs
        decoder_op = MultiInputConvDecoder
        decoder_kwargs = dict(
            in_channels=in_channels,
            out_channels=hparams.num_classes,
            depth=hparams.model_depth,
            block_depth=hparams.model_block_depth,
            num_feature_maps=hparams.model_feature_maps,
            feature_map_multiplier=hparams.model_feature_map_multiplier,
            feature_map_multiplier_backwards=True,
            activation_op=activation_op,
            activation_kwargs=hparams.model_activation_kwargs,
            norm_op=norm_op,
            norm_kwargs=hparams.model_norm_kwargs,
            norm_depth=hparams.model_norm_depth_decoder,
            conv_op=conv_op,
            conv_kwargs=None,
            upsample_op=upsample_op,
            upsample_kwargs=hparams.model_upsample_kwargs,
            dropout_op=dropout_op,
            dropout_kwargs=hparams.model_dropout_kwargs,
            initial_upsample_op=initial_upsample_op,
            initial_upsample_kwargs=initial_upsample_kwargs,
            output_activation_op=output_activation_op,
            output_activation_kwargs=hparams.model_output_activation_kwargs,
            coords=hparams.model_use_coords,
            coords_dim=2,
        )

        # configure attention
        kdim = [hparams.representation_channels + hparams.num_classes - 1]
        vdim = [hparams.representation_channels + hparams.num_classes - 1]
        for i in range(hparams.model_spatial_attention):
            fmaps = (
                hparams.model_feature_maps
                * hparams.model_feature_map_multiplier ** (hparams.model_depth - i - 1)
                + hparams.num_classes
                - 1
            )
            kdim.append(fmaps)
            vdim.append(fmaps)
        kdim = list(reversed(kdim))
        vdim = list(reversed(vdim))
        attention_op = MultiheadAttention
        attention_kwargs = dict(
            embed_dim=hparams.model_att_embed_dim,
            num_heads=hparams.model_att_heads,
            spatial_attention=True,
            concat_coords=2,
            bias=True,
            batch_first=True,
            embed_v=True,
            qdim=hparams.num_classes - 1,
            kdim=kdim,
            vdim=vdim,
        )
        temporal_attention_kwargs = dict(
            embed_dim=hparams.model_att_embed_dim,
            num_heads=hparams.model_att_heads,
            spatial_attention=False,
            concat_coords=0,
            bias=True,
            batch_first=True,
            embed_v=False,
            qdim=hparams.num_classes - 1,
            kdim=hparams.num_classes - 1,
            vdim=None,
        )

        # put everything together
        model_op = AttentiveImageProcess
        model_kwargs = dict(
            num_attention=hparams.model_spatial_attention,
            global_sum=hparams.model_global_sum,
            scaleup_query=True,
            downsample=False,
            upsample=False,
            variational=True,
            higher_level_attention_op=attention_op
            if hparams.model_temporal_attention > 0
            else None,
            higher_level_attention_kwargs=temporal_attention_kwargs,
            target_encoder_op=None,
            target_encoder_kwargs=None,
            context_encoder_op=context_encoder_op,
            context_encoder_kwargs=context_encoder_kwargs,
            decoder_op=decoder_op,
            decoder_kwargs=decoder_kwargs,
            attention_op=attention_op,
            attention_kwargs=attention_kwargs,
            posterior_encoder_op=None,
            posterior_encoder_kwargs=None,
        )

        return model_op(**model_kwargs)

    def forward(self, *x: Any, **xx: Any):
        """Forward pass through the module is just a forward pass through the model."""

        return self.model(*x, **xx)

    def step(
        self,
        batch: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        batch_idx: int,
        return_all: bool = False,
        loss_aggregate: bool = True,
    ) -> Tuple[torch.Tensor, dict, Optional[dict]]:
        """Base step that is used by train, val and test.

        Args:
            batch: A dictionary that should at least have "position", "data".
            batch_idx: Not used.
            return_all: Also return a dict with input and output tensors. Use this to
                get the data for visualization and similar.
            loss_aggregate: If True, aggregate the loss over the batch. This is for
                testing purposes.

        Returns:
            (
                Total loss,
                Dictionary of scalar tensors for logging,
                None or dictionary of higher-dimensional tensors
            )

        """

        context, target = batch

        context_query = context["position"]  # (B, N, P)
        context_seg = context["data"]  # (B, N, 1, SPACE)
        target_query = target["position"]  # (B, M, P)
        target_seg = target["data"]  # (B, M, 1, SPACE)
        context_query = torch.from_numpy(context_query).to(self.device, torch.float)
        context_seg = torch.from_numpy(context_seg).to(self.device, torch.float)
        target_query = torch.from_numpy(target_query).to(self.device, torch.float)
        target_seg = torch.from_numpy(target_seg).to(self.device, torch.float)

        context_seg = make_onehot(context_seg, range(self.hparams.num_classes), axis=2)
        target_seg = make_onehot(target_seg, range(self.hparams.num_classes), axis=2)

        prediction = self.model(
            context_query=context_query,
            target_query=target_query,
            context_image=None,
            context_seg=context_seg,
            target_image=None,
            target_seg=target_seg,
        )

        # prediction/task loss
        if not self.hparams.criterion_task_onehot:
            loss_task = self.criterion_task(
                stack_batch(prediction),
                stack_batch(torch.argmax(target_seg, 2, keepdim=False)),
            )
        else:
            loss_task = self.criterion_task(
                stack_batch(prediction), stack_batch(target_seg)
            )
        if loss_task.ndim > 1:
            loss_task = unstack_batch(loss_task, prediction.shape[0])

        # when reduction is none, we do sum and batch/point average
        while loss_task.ndim > 2:
            loss_task = loss_task.sum(-1)
        if loss_aggregate:
            loss_task = loss_task.mean()
        else:
            loss_task = loss_task.mean(1)

        log = {"loss_task": loss_task}

        # during validation we need to manually encode the posterior
        if not self.training:
            _ = self.model.encode_posterior(
                self.model.encode_context(
                    target_query,
                    None,
                    target_seg,
                )
            )

        loss_latent = self.criterion_latent(self.model.posterior, self.model.prior)
        while loss_latent.ndim > 1:
            loss_latent = loss_latent.sum(-1)
        if loss_aggregate:
            loss_latent = loss_latent.mean()
        loss_total = loss_task + self.hparams.criterion_latent_weight * loss_latent
        log["loss_latent"] = loss_latent
        log["loss_total"] = loss_total

        # doesn't hurt to monitor memory usage
        log["gpu_memory"] = torch.cuda.max_memory_allocated() // (1024 ** 2)

        log_tensor = {}
        if return_all:
            log_tensor["context_positions"] = context["position"]
            log_tensor["target_positions"] = target["position"]
            log_tensor["context_query"] = context_query
            log_tensor["target_query"] = target_query
            log_tensor["context_seg"] = context_seg
            log_tensor["target_seg"] = target_seg
            log_tensor["prediction"] = prediction

        return loss_total, log, log_tensor

    def log_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        to_disk: bool = False,
        subdir: Optional[str] = None,
        ntotal: int = 64,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        range_: Optional[Iterable[float]] = None,
        scale_each: bool = False,
        pad_value: float = 0.0,
        split_channels: bool = True,
        point_average: bool = False,
    ):
        """Log a tensor.

        We're assuming a shape of (B, N, C, ...) and try to create image grids from it.
        At the moment only logging to Visdom and to disk is supported.

        Args:
            tensor: The data. We take the last element along the second axis.
            name: Should describe what you're trying to log.
            epoch: Prepends "epoch{epoch}_" to the name.
            batch_idx: Prepends "step{batch_idx}_" to the name.
            to_disk: Save to disk.
            subdir: Optional subdirectory in logging folder.
            ntotal: Take this many elements from the batch axis.
            nrow: Number of items per row in grid.
            padding: Padding between items.
            normalize: Normalize by range_.
            range_: Normalize to this range.
            scale_each: Normalize each item separately.
            pad_value: Fill value for padding.
            split_channels: Make separate image grid for each channel.
            point_average: Average over context or target points. Otherwise the last one
                will be used.

        """

        # Assume (B, N, C, H, W) shape. In 3D we just take the center along the first
        # axis. Shouldn't matter, because we usually work with random rotations anyway.
        if point_average:
            tensor = tensor[:ntotal].float().mean(1)
        else:
            tensor = tensor[:ntotal, -1].float()
        image_grid = make_grid(
            tensor=tensor,
            nrow=nrow,
            padding=padding,
            normalize=normalize,
            value_range=range_,
            scale_each=scale_each,
            pad_value=pad_value,
        )

        for logger in self.logger:
            if isinstance(logger, VisdomLogger):
                if split_channels:
                    for c in range(image_grid.shape[0]):
                        logger.experiment.image(
                            image_grid[c],
                            name + "_c{}".format(c),
                            opts=dict(title=name + "_c{}".format(c)),
                        )
                else:
                    logger.experiment.image(
                        image_grid,
                        name,
                        opts=dict(title=name),
                    )

        if to_disk:
            prefix = ""
            if epoch is not None:
                format_ = "epoch{:0" + str(len(str(self.trainer.max_epochs))) + "d}_"
                prefix += format_.format(epoch)
            if batch_idx is not None:
                format_ = "step{}_"
                prefix += format_.format(batch_idx)
            name = prefix + name
            if subdir is not None:
                name = os.path.join(subdir, name)
            if split_channels:
                for c in range(image_grid.shape[0]):
                    fp = os.path.join(
                        self.trainer._default_root_dir, name + "_c{}.png".format(c)
                    )
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    save_image(image_grid[c], fp)
            else:
                fp = os.path.join(self.trainer._default_root_dir, name + ".png")
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                save_image(image_grid, fp)

    def training_step(
        self, batch: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: A dictionary that should at least have "position", "data".
            batch_idx: The batch index.

        Returns:
            The total loss

        """

        return_all = batch_idx % self.trainer.log_every_n_steps == 0 and batch_idx > 0
        loss, log, log_tensor = self.step(batch, batch_idx, return_all=return_all)

        self.log_dict(
            {"train_{}".format(k): v for k, v in log.items()},
            on_step=True,
            on_epoch=False,
        )

        if return_all:
            for key in ("context_seg", "target_seg", "prediction"):
                if key not in log_tensor:
                    continue
                val = log_tensor[key]
                if val is not None:
                    self.log_tensor(
                        tensor=torch.argmax(val, 2, keepdim=True),
                        name="train_" + key,
                        epoch=self.current_epoch,
                        batch_idx=batch_idx,
                        to_disk=False,
                        subdir="train",
                        nrow=8,
                        padding=2,
                        normalize=True,
                        range_=(0, self.hparams.num_classes - 1),
                        scale_each=True,
                        pad_value=1,
                        split_channels=False,
                        point_average="context" in key,
                    )

        return loss

    def validation_step(
        self, batch: Dict[str, np.ndarray], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: A dictionary that should at least have "position", "data".
            batch_idx: The batch index.

        Returns:
            The total loss

        """

        return_all = batch_idx == 0
        loss, log, log_tensor = self.step(batch, batch_idx, return_all=return_all)

        self.log_dict({"val_{}".format(k): v for k, v in log.items()})

        if return_all:

            # regular images
            for key in ("context_seg", "target_seg", "prediction"):
                if key not in log_tensor:
                    continue
                val = log_tensor[key]
                if val is not None:
                    self.log_tensor(
                        tensor=torch.argmax(val, 2, keepdim=True),
                        name="val_" + key,
                        epoch=self.current_epoch,
                        batch_idx=batch_idx,
                        to_disk=True,
                        subdir="val",
                        nrow=8,
                        padding=2,
                        normalize=True,
                        range_=(0, self.hparams.num_classes - 1),
                        scale_each=True,
                        pad_value=1,
                        split_channels=False,
                        point_average="context" in key,
                    )

            # samples
            samples = self.model.sample(
                8,
                log_tensor["context_query"][:8],
                log_tensor["target_query"][:8, -1:],  # want just 1 target point
                None,
                log_tensor["context_seg"][:8],
                None,
            )
            samples = torch.argmax(samples, 3, keepdim=True)
            samples = stack_batch(samples)
            self.log_tensor(
                tensor=samples,
                name="val_samples",
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                to_disk=True,
                subdir="val",
                nrow=samples.shape[0] // 8,
                padding=2,
                normalize=True,
                range_=(0, self.hparams.num_classes - 1),
                scale_each=False,
                pad_value=1,
                split_channels=False,
            )

        return loss

    def test_step(
        self, batch: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]], batch_idx: int
    ) -> Tuple[str, float, float, float, float, float, float, float]:
        """Test step.

        Args:
            batch: A dictionary that should at least have "position", "data".
            batch_idx: The batch index.

        Returns:
            Several metrics for logging

        """

        old_reduction_task = self.criterion_task.reduction
        old_reduction_latent = self.criterion_latent.reduction
        self.criterion_task.reduction = "none"
        self.criterion_latent.reduction = "none"

        loss, log, log_tensor = self.step(
            batch, batch_idx, return_all=True, loss_aggregate=False
        )

        self.criterion_task.reduction = old_reduction_task
        self.criterion_latent.reduction = old_reduction_latent

        prediction = log_tensor["prediction"]
        prediction = torch.argmax(prediction, 2, keepdim=True)
        prediction = make_onehot(prediction, range(self.hparams.num_classes), axis=2)
        prediction = prediction[:, :, 1:]  # remove background
        target_seg = log_tensor["target_seg"][:, :, 1:]
        scores = dice(prediction, target_seg, batch_axes=(0, 1, 2))
        scores = scores.mean(1)  # average along trajectory

        num_context = log_tensor["context_positions"].shape[1]
        result = np.concatenate(
            (
                np.array([num_context] * scores.shape[0])[:, None],
                log["loss_task"][:, None].cpu().numpy(),
                log["loss_latent"][:, None].cpu().numpy(),
                log["loss_total"][:, None].cpu().numpy(),
                scores.cpu().numpy(),
            ),
            1,
        )

        return result

    def test_epoch_end(self, outputs):

        columns = [
            "Context Size",
            "Loss Task",
            "Loss Latent",
            "Loss",
        ]
        for c in range(1, self.hparams.num_classes):
            columns.append("Dice Class " + str(c))

        arr = pd.DataFrame(np.concatenate(outputs, 0), columns=columns)
        arr.to_csv(os.path.join(self.trainer._default_root_dir, "test.csv"))

        arr = arr.mean().to_dict()

        self.log_dict(arr, logger=False)

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument(
            "--criterion_task", type=str, default="crossentropydiceloss"
        )
        parser.add_argument("--criterion_task_reduction", type=str, default="mean")
        parser.add_argument("--criterion_task_onehot", type=str2bool, default=True)
        parser.add_argument("--criterion_latent", type=str, default="kldivergence")
        parser.add_argument("--criterion_latent_reduction", type=str, default="mean")
        parser.add_argument("--criterion_latent_weight", type=float, default=0.01)
        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--step_lr", type=str2bool, default=False)
        parser.add_argument("--step_lr_gamma", type=float, default=0.99)
        parser.add_argument("--reduce_lr_on_plateau", type=str2bool, default=False)
        parser.add_argument("--reduce_lr_factor", type=float, default=0.1)
        parser.add_argument("--reduce_lr_patience", type=int, default=10)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--representation_channels", type=int, default=128)
        parser.add_argument("--model_spatial_attention", type=int, default=2)
        parser.add_argument("--model_temporal_attention", type=int, default=0)
        parser.add_argument("--model_att_embed_dim", type=int, default=128)
        parser.add_argument("--model_att_heads", type=int, default=8)
        parser.add_argument("--model_depth", type=int, default=5)
        parser.add_argument("--model_block_depth", type=int, default=1)
        parser.add_argument("--model_feature_maps", type=int, default=12)
        parser.add_argument("--model_activation", type=str, default="leakyrelu")
        parser.add_argument(
            "--model_activation_kwargs", action=DictArgument, nargs="+", default=None
        )
        parser.add_argument("--model_output_activation", type=str, default="softmax")
        parser.add_argument(
            "--model_output_activation_kwargs",
            action=DictArgument,
            nargs="+",
            default=None,
        )
        parser.add_argument("--model_upsample", type=str, default="convtranspose")
        parser.add_argument(
            "--model_upsample_kwargs",
            action=DictArgument,
            nargs="+",
            default=dict(kernel_size=4, stride=2, padding=1),
        )
        parser.add_argument("--model_initial_upsample", type=str, default="upsample")
        parser.add_argument(
            "--model_initial_upsample_kwargs",
            action=DictArgument,
            nargs="+",
            default=None,
        )
        parser.add_argument("--model_norm", type=str, default="instancenorm")
        parser.add_argument(
            "--model_norm_kwargs", action=DictArgument, nargs="+", default=None
        )
        parser.add_argument("--model_norm_depth", type=int, default=1)
        parser.add_argument("--model_norm_depth_decoder", type=int, default=0)
        parser.add_argument("--model_pool", type=str, default="avgpool")
        parser.add_argument(
            "--model_pool_kwargs", action=DictArgument, nargs="+", default=None
        )
        parser.add_argument("--model_dropout", type=str, default=None)
        parser.add_argument(
            "--model_dropout_kwargs", action=DictArgument, nargs="+", default=None
        )

        return parser


if __name__ == "__main__":

    ExperimentModule = ContinuousSegmentation

    parser = make_default_parser()
    parser.add_argument(
        "--name", type=str, default=os.path.basename(__file__).split(".")[0]
    )
    parser = pl.Trainer.add_argparse_args(parser)

    # select function type
    parser.add_argument("--data", type=str, default="toy")
    temp_args, _ = parser.parse_known_args()

    # try to find a module in gg.data.lightning that matches the name
    module_name = temp_args.data.capitalize() + "Module"
    DataModule = getattr(gg.data, module_name, None)
    if DataModule is None:
        raise ValueError("Unknown module type: {}".format(temp_args.data))

    # add parser args
    parser = DataModule.add_data_specific_args(parser)
    parser = ExperimentModule.add_model_specific_args(parser)
    args = parser.parse_args()

    # manual changes
    if args.mlflow is None:
        args.mlflow = "file://" + os.path.join(args.base_dir, "mlruns")
    if "--max_epochs" not in sys.argv:
        args.max_epochs = 200
    args.num_classes = args.num_objects + 1

    experiment_name = "{}".format(os.path.basename(__file__).split(".")[0])
    if hasattr(args, "dim") and args.dim != 2:
        experiment_name += "_{}d".format(args.dim)
    run_experiment(
        ExperimentModule,
        DataModule,
        args,
        experiment_name,
        globs=globals(),
    )