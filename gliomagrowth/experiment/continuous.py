import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl

import gliomagrowth as gg
from gliomagrowth.util.util import (
    nn_module_lookup,
    str2bool,
    make_onehot,
    stack_batch,
)
from gliomagrowth.util.lightning import (
    VisdomLogger,
    make_default_parser,
    run_experiment,
)

from deeputil import loss as customloss
from deeputil.model import MultiOutputInjectionConvEncoder, MultiInputConvDecoder
from deeputil.model.attention import MultiheadAttention
from deeputil.model.neuralprocess import AttentiveSegmentationProcess

from deeputil.metrics import dice


class ContinuousTumorGrowth(pl.LightningModule):
    def __init__(
        self,
        use_images=True,
        in_channels=4,  # image channels
        num_classes=4,  # incl. background
        dim=2,
        reconstruct_context=True,
        variational=True,
        criterion_task="crossentropydiceloss",
        criterion_task_reduction="mean",
        criterion_task_onehot=True,
        criterion_latent="kldivergence",
        criterion_latent_weight=1.0,
        optimizer="adam",
        learning_rate=0.0001,
        step_lr=False,
        step_lr_gamma=0.99,  # every epoch
        reduce_lr_on_plateau=False,
        reduce_lr_factor=0.1,
        reduce_lr_patience=10,
        representation_channels=128,
        normalize_date_factor=1.0,
        model_global_sum=True,
        model_spatial_attention=2,
        model_temporal_attention=2,
        model_att_embed_dim=128,
        model_att_heads=8,
        model_depth=5,
        model_block_depth=2,
        model_feature_maps=24,
        model_feature_map_multiplier=2,
        model_activation="leakyrelu",
        model_activation_kwargs=None,
        model_output_activation="softmax",
        model_output_activation_kwargs=None,
        model_norm="instancenorm",
        model_norm_kwargs=None,
        model_norm_depth=1,
        model_norm_depth_decoder=0,
        model_pool="avgpool",
        model_pool_kwargs=None,
        model_upsample="upsample",
        model_upsample_kwargs=None,
        model_initial_upsample_kwargs=None,
        model_dropout=False,
        model_dropout_kwargs=None,
        model_global_pool="adaptiveavgpool",
        model_global_pool_kwargs=None,
        model_use_coords=True,
        test_dice_samples=100,
        **kwargs
    ):

        super().__init__()

        self.save_hyperparameters()
        self.model = self.make_model(self.hparams)
        self.criterion_task = nn_module_lookup(criterion_task, dim, customloss)(
            reduction=self.hparams.criterion_task_reduction
        )
        # weights hack so we can resume
        self.criterion_task.weight = torch.tensor((1,) * num_classes).to(
            self.device, torch.float
        )
        if variational:
            self.criterion_latent = nn_module_lookup(
                criterion_latent, dim, customloss
            )()

        self.learning_rate = self.hparams.learning_rate

    def configure_optimizers(self):
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
    def make_model(hparams):

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
        activation_op = nn_module_lookup(hparams.model_activation, hparams.dim)
        output_activation_op = nn_module_lookup(
            hparams.model_output_activation, hparams.dim
        )
        norm_op = nn_module_lookup(hparams.model_norm, hparams.dim)
        pool_op = nn_module_lookup(hparams.model_pool, hparams.dim)
        upsample_op = nn_module_lookup(hparams.model_upsample, hparams.dim)
        global_pool_op = nn_module_lookup(hparams.model_global_pool, hparams.dim)
        dropout_op = (
            nn_module_lookup("dropout", hparams.dim) if hparams.model_dropout else None
        )
        conv_op = nn_module_lookup("conv", hparams.dim)

        # only necessary when target image inputs are available
        target_encoder_op = None
        target_encoder_kwargs = None

        # configure encoder
        context_encoder_op = MultiOutputInjectionConvEncoder
        context_encoder_kwargs = dict(
            return_last=1
            + hparams.model_spatial_attention
            + hparams.model_temporal_attention,
            in_channels=1
            + hparams.num_classes
            + int(hparams.use_images) * hparams.in_channels,
            out_channels=(1 + int(hparams.variational))
            * hparams.representation_channels,
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
            coords_dim=hparams.dim,
        )

        # configure decoder
        in_channels = [hparams.representation_channels + 1]
        for i in range(hparams.model_spatial_attention):
            in_channels.append(hparams.representation_channels)
        for i in range(hparams.model_temporal_attention):
            fmaps = (
                hparams.model_feature_maps
                * hparams.model_feature_map_multiplier
                ** (hparams.model_depth - hparams.model_spatial_attention - i - 1)
                + 1
            )
            in_channels.append(fmaps)
        in_channels = list(reversed(in_channels))
        if hparams.model_initial_upsample_kwargs is None:
            initial_upsample_kwargs = dict(
                size=(
                    2 ** (7 - hparams.model_depth),
                    2 ** (7 - hparams.model_depth),
                )
            )
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
            initial_upsample_op=upsample_op,
            initial_upsample_kwargs=initial_upsample_kwargs,
            output_activation_op=output_activation_op,
            output_activation_kwargs=hparams.model_output_activation_kwargs,
            coords=hparams.model_use_coords,
            coords_dim=hparams.dim,
        )

        # configure attention
        kdim = [hparams.representation_channels + 1]
        vdim = [hparams.representation_channels + 1]
        for i in range(hparams.model_spatial_attention):
            fmaps = (
                hparams.model_feature_maps
                * hparams.model_feature_map_multiplier ** (hparams.model_depth - i - 1)
                + 1
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
            concat_coords=hparams.dim,
            bias=True,
            batch_first=True,
            embed_v=True,
            qdim=1,
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
            qdim=1,
            kdim=1,
            vdim=None,
        )

        # put everything together
        model_op = AttentiveSegmentationProcess
        model_kwargs = dict(
            num_attention=hparams.model_spatial_attention,
            global_sum=hparams.model_global_sum,
            scaleup_query=True,
            downsample=False,
            upsample=False,
            variational=hparams.variational,
            higher_level_attention_op=attention_op
            if hparams.model_temporal_attention > 0
            else None,
            higher_level_attention_kwargs=temporal_attention_kwargs,
            target_encoder_op=target_encoder_op,
            target_encoder_kwargs=target_encoder_kwargs,
            context_encoder_op=context_encoder_op,
            context_encoder_kwargs=context_encoder_kwargs,
            decoder_op=decoder_op,
            decoder_kwargs=decoder_kwargs,
            attention_op=attention_op,
            attention_kwargs=attention_kwargs,
        )

        return model_op(**model_kwargs)

    def forward(self, *x, **xx):

        return self.model(*x, **xx)

    def step(self, batch, batch_idx, return_all=False):

        context, target = transformable_to_ct(batch)

        context_query = context["scan_days"]  # (B, N, 1)
        context_seg = context["seg"]  # (B, N, 1, H, W)
        target_query = target["scan_days"]  # (B, M, 1)
        target_seg = target["seg"]  # (B, M, 1, H, W)
        context_query = torch.from_numpy(context_query).to(self.device, torch.float)
        context_seg = torch.from_numpy(context_seg).to(self.device, torch.float)
        target_query = torch.from_numpy(target_query).to(self.device, torch.float)
        target_seg = torch.from_numpy(target_seg).to(self.device, torch.float)
        if self.hparams.use_images:
            context_image = context["data"]  # (B, N, C, H, W)
            context_image = torch.from_numpy(context_image).to(self.device, torch.float)
            # target images only for posterior
            target_image = target["data"]  # (B, M, C, H, W)
            target_image = torch.from_numpy(target_image).to(self.device, torch.float)
        else:
            context_image = None
            target_image = None

        context_seg = make_onehot(context_seg, range(self.hparams.num_classes), axis=2)
        target_seg = make_onehot(target_seg, range(self.hparams.num_classes), axis=2)

        context_query *= self.hparams.normalize_date_factor
        target_query *= self.hparams.normalize_date_factor

        if self.hparams.reconstruct_context:
            target_query = torch.cat((context_query, target_query), 1)
            target_seg = torch.cat((context_seg, target_seg), 1)
            if target_image is not None:
                target_image = torch.cat((context_image, target_image), 1)

        prediction = self.model(
            context_query=context_query,
            context_seg=context_seg,
            target_query=target_query,
            context_image=context_image,
            target_image=target_image,
            target_seg=target_seg,
        )

        if not self.hparams.criterion_task_onehot:
            loss_task = (
                self.criterion_task(
                    stack_batch(prediction),
                    stack_batch(torch.argmax(target_seg, 2, keepdim=False)),
                )
                .mean(0)
                .sum()
            )
        else:
            loss_task = (
                self.criterion_task(stack_batch(prediction), stack_batch(target_seg))
                .mean(0)
                .sum()
            )
        log = {"loss_task": loss_task}
        if self.hparams.variational:
            # during validation we need to manually encode the posterior
            if not self.training:
                _ = self.model.encode_posterior(
                    self.model.encode_context(target_query, target_seg, target_image)
                )

            loss_latent = (
                self.criterion_latent(self.model.posterior, self.model.prior)
                .mean(0)
                .sum()
            )
            loss_total = loss_task + self.hparams.criterion_latent_weight * loss_latent
            log["loss_latent"] = loss_latent
            log["loss_total"] = loss_total
        else:
            loss_total = loss_task
        log["gpu_memory"] = torch.cuda.max_memory_allocated() // (1024 ** 2)

        log_tensor = {}
        if return_all:
            log_tensor["subjects"] = context["subjects"]
            log_tensor["timesteps"] = context["timesteps"]
            log_tensor["slices"] = context["slices"]
            log_tensor["context_query"] = context_query
            log_tensor["target_query"] = target_query
            log_tensor["context_image"] = context_image
            log_tensor["context_seg"] = context_seg
            log_tensor["target_seg"] = target_seg
            log_tensor["prediction"] = prediction

        return loss_total, log, log_tensor

    def log_tensor(
        self,
        tensor,
        name,
        epoch=None,
        batch_idx=None,
        to_disk=False,
        subdir=None,
        nrow=8,
        padding=2,
        normalize=True,
        range_=None,
        scale_each=True,
        pad_value=0,
        split_channels=True,
    ):

        # assume (B, N, C, H, W) shape
        image_grid = make_grid(
            tensor=tensor[:64, -1].float(),
            nrow=nrow,
            padding=padding,
            normalize=normalize,
            range=range_,
            scale_each=scale_each,
            pad_value=pad_value,
        )

        for logger in self.logger:
            if isinstance(logger, VisdomLogger):
                if split_channels:
                    for c in range(image_grid.shape[0]):
                        logger.experiment.add_image(
                            image_grid[c],
                            name + "_c{}".format(c),
                            opts=dict(title=name + "_c{}".format(c)),
                        )
                else:
                    logger.experiment.add_image(
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

    def training_step(self, batch, batch_idx):

        return_all = batch_idx % self.trainer.log_every_n_steps == 0 and batch_idx > 0
        loss, log, log_tensor = self.step(batch, batch_idx, return_all=return_all)

        self.log_dict(
            {"train_{}".format(k): v for k, v in log.items()},
            on_step=True,
            on_epoch=False,
        )

        if return_all:
            for key in ("context_image", "context_seg", "target_seg", "prediction"):
                if key not in log_tensor:
                    continue
                val = log_tensor[key]
                if val is not None:
                    if "image" in key:
                        self.log_tensor(
                            tensor=val,
                            name="train_" + key,
                            epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            to_disk=False,
                            subdir="train",
                            nrow=8,
                            padding=2,
                            normalize=True,
                            range_=None,
                            scale_each=True,
                            pad_value=1,
                            split_channels=True,
                        )
                    else:
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
                        )

        return loss

    def validation_step(self, batch, batch_idx):

        return_all = batch_idx == 0
        loss, log, log_tensor = self.step(batch, batch_idx, return_all=return_all)

        self.log_dict({"val_{}".format(k): v for k, v in log.items()})

        if return_all:
            for key in ("context_image", "context_seg", "target_seg", "prediction"):
                if key not in log_tensor:
                    continue
                val = log_tensor[key]
                if val is not None:
                    if "image" in key:
                        self.log_tensor(
                            tensor=val,
                            name="val_" + key,
                            epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            to_disk=True,
                            subdir="val",
                            nrow=8,
                            padding=2,
                            normalize=True,
                            range_=None,
                            scale_each=True,
                            pad_value=1,
                            split_channels=True,
                        )
                    else:
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
                        )

        return loss

    def test_step(self, batch, batch_idx):

        gt_volume = np.sum(batch["seg"][0, -1] > 0)
        if gt_volume == 0:
            return

        old_reconstruct_context = self.hparams.reconstruct_context
        self.hparams.reconstruct_context = False

        loss, log, log_tensor = self.step(batch, batch_idx, return_all=True)

        self.hparams.reconstruct_context = old_reconstruct_context

        gt_segmentation = log_tensor["target_seg"][0, -1]
        gt_segmentation = torch.argmax(gt_segmentation, 0, keepdim=False) > 0
        prediction = log_tensor["prediction"][0, -1]
        prediction = torch.argmax(prediction, 0, keepdim=False) > 0
        predicted_dice = dice(prediction, gt_segmentation)

        # make samples
        samples = self.model.sample(
            self.hparams.test_dice_samples,
            log_tensor["context_query"],
            log_tensor["context_seg"],
            log_tensor["target_query"][:, -1:],
            log_tensor["context_image"],
            None,
        )[:, 0, -1]
        samples = torch.argmax(samples, 1, keepdim=False) > 0

        # get dice scores
        tp = samples * gt_segmentation[None, ...]
        fp = samples * torch.logical_not(gt_segmentation[None, ...])
        fn = torch.logical_not(samples) * gt_segmentation[None, ...]
        while len(tp.shape) > 1:
            tp = tp.sum(-1)
            fp = fp.sum(-1)
            fn = fn.sum(-1)
        dice_scores = (2 * tp.float()) / (2 * tp.float() + fp.float() + fn.float())
        dice_scores[torch.isnan(dice_scores)] = 0
        best_dice = torch.max(dice_scores)

        # get volumes
        sum_axes = tuple(range(1, len(samples.shape)))
        pred_volumes = torch.sum(samples, sum_axes)
        best_volume_index = torch.argmin(torch.abs(pred_volumes - gt_volume))
        best_volume_dice = dice_scores[best_volume_index]

        subject_info = "_".join(
            [
                log_tensor["subjects"][0],
                str(log_tensor["timesteps"][0] + log_tensor["context_query"].shape[1]),
                str(log_tensor["slices"][0]),
                "it" + str(log_tensor["context_query"].shape[1]),
            ]
        )

        return (
            subject_info,
            gt_volume,
            log["loss_task"].item(),
            log["loss_latent"].item(),
            log["loss_total"].item(),
            predicted_dice,
            best_volume_dice.item(),
            best_dice.item(),
        )

    def test_epoch_end(self, outputs):

        arr = np.array(outputs, dtype=object)
        arr = pd.DataFrame(
            arr,
            columns=[
                "Subject and Timestep",
                "GT Volume",
                "Loss Task",
                "Loss Latent",
                "Loss",
                "Dice",
                "Prior Best Volume Dice",
                "Prior Best Dice",
            ],
        )
        arr = arr.set_index("Subject and Timestep")
        arr.to_csv(os.path.join(self.trainer._default_root_dir, "test.csv"))

        return arr.mean().to_dict()

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )

        parser.add_argument("--use_images", type=str2bool, default=True)
        parser.add_argument("--reconstruct_context", type=str2bool, default=True)
        parser.add_argument(
            "--criterion_task", type=str, default="crossentropydiceloss"
        )
        parser.add_argument("--criterion_task_reduction", type=str, default="mean")
        parser.add_argument("--criterion_task_onehot", type=str2bool, default=True)
        parser.add_argument("--criterion_latent", type=str, default="kldivergence")
        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--step_lr", type=str2bool, default=False)
        parser.add_argument("--step_lr_gamma", type=float, default=0.99)
        parser.add_argument("--reduce_lr_on_plateau", type=str2bool, default=False)
        parser.add_argument("--reduce_lr_factor", type=float, default=0.1)
        parser.add_argument("--reduce_lr_patience", type=int, default=10)
        parser.add_argument("--in_channels", type=int, default=4)
        parser.add_argument("--num_classes", type=int, default=4)
        parser.add_argument("--dim", type=int, default=2)
        parser.add_argument("--variational", type=str2bool, default=True)
        parser.add_argument("--criterion_latent_weight", type=float, default=0.001)
        parser.add_argument("--representation_channels", type=int, default=128)
        parser.add_argument("--normalize_date_factor", type=float, default=1.0)
        parser.add_argument("--model_spatial_attention", type=int, default=2)
        parser.add_argument("--model_temporal_attention", type=int, default=0)
        parser.add_argument("--model_att_embed_dim", type=int, default=128)
        parser.add_argument("--model_att_heads", type=int, default=8)
        parser.add_argument("--model_depth", type=int, default=4)
        parser.add_argument("--model_feature_maps", type=int, default=12)
        parser.add_argument("--model_activation", type=str, default="leakyrelu")
        parser.add_argument("--model_output_activation", type=str, default="softmax")
        parser.add_argument("--model_norm", type=str, default="instancenorm")
        parser.add_argument("--model_norm_depth", type=int, default=1)
        parser.add_argument("--model_norm_depth_decoder", type=int, default=0)
        parser.add_argument("--model_dropout", type=str2bool, default=False)
        parser.add_argument("--test_dice_samples", type=int, default=100)

        return parser


if __name__ == "__main__":

    ExperimentModule = ContinuousTumorGrowth

    parser = make_default_parser()
    parser.add_argument(
        "--name", type=str, default=os.path.basename(__file__).split(".")[0]
    )
    parser = pl.Trainer.add_argparse_args(parser)

    # select function type
    parser.add_argument("--data", type=str, default="glioma")
    temp_args, _ = parser.parse_known_args()
    if temp_args.data == "glioma":
        DataModule = gg.data.lightning.GliomaModule
    else:
        # try to find a module in gg.data.lightning that matches the name
        module_name = temp_args.data.capitalize() + "Module"
        DataModule = getattr(gg.data.lightning, module_name, None)
        if DataModule is None:
            raise ValueError("Unknown module type: {}".format(temp_args.data))

    # add parser args
    parser = DataModule.add_data_specific_args(parser)
    parser = ExperimentModule.add_model_specific_args(parser)
    args = parser.parse_args()

    # manual changes
    if args.mlflow is None:
        args.mlflow = "file://" + os.path.join(args.base_dir, "mlruns")
    if args.whole_tumor:
        args.num_classes = 2
    if "--max_epochs" not in sys.argv:
        args.max_epochs = 300
    if not args.use_images:
        args.model_norm_depth = 0
        args.transform_gamma = False
        args.transform_blur = False
        args.transform_brightness = False

    run_experiment(
        ExperimentModule,
        DataModule,
        args,
        os.path.basename(__file__).split(".")[0],
        globs=globals(),
    )