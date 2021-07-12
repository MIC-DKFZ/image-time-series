import torch
from torch import nn
from typing import Union, Optional, Iterable

from gliomagrowth.util.util import make_onehot as make_onehot_segmentation


class FScoreLoss(nn.modules.loss._WeightedLoss):
    """Uses the 1 - F-score as a loss.

    .. math::
        F = \frac{ (1 + \beta^2) TP }{ (1 + \beta^2) TP + \beta^2 FN + FP }

    Args:
        beta: The beta in the above formula.
        eps: Epsilon for numerical stability.
        tp_bias: This is added to the TP count. Should add a little stability for
            very small structures.
        make_onehot: Convert the target segmentation to onehot internally. Turn this
            off if the target is already onehot.
        make_onehot_newaxis: 'newaxis' argument for the onehot conversion.
        ignore_index: These classes will not contribute to the loss. Has no effect if
            reduction is "none".
        weight: Weights for the different classes. Has no effect if reduction is "none".
        reduction: "mean", "sum", or "none".

    """

    def __init__(
        self,
        beta: float = 1.0,
        eps: float = 1e-6,
        tp_bias: Union[int, float] = 0,
        make_onehot: bool = False,
        make_onehot_newaxis: bool = False,
        ignore_index: Optional[Union[int, Iterable[int]]] = None,
        weight: Optional[torch.tensor] = None,
        reduction="mean",
        **kwargs
    ):

        super(FScoreLoss, self).__init__(weight=weight, reduction=reduction)
        self.beta = beta
        self.eps = eps
        self.tp_bias = tp_bias
        self.make_onehot = make_onehot
        self.make_onehot_newaxis = make_onehot_newaxis
        self.ignore_index = ignore_index
        if isinstance(ignore_index, int):
            self.ignore_index = [
                ignore_index,
            ]

    def forward(self, input_: torch.tensor, target: torch.tensor) -> torch.tensor:
        """Forward pass through the loss.

        Args:
            input_: Input with expected shape (B, C, ...) with C the number of classes.
            target: Target, either onehot with shape (B, C, ...), or not with shape
                either (B, ...) or (B, 1, ...). Make sure to set the make_onehot and
                make_onehot_newaxis arguments appropriately.

        Returns:
            The loss.

        """

        assert not target.requires_grad

        if self.make_onehot:
            target = make_onehot_segmentation(
                target, range(input_.shape[1]), newaxis=self.make_onehot_newaxis
            )

        target = target.float()

        tp = input_ * target
        fn = (1 - input_) * target
        fp = input_ * (1 - target)

        while tp.dim() > 2:
            tp = tp.sum(-1, keepdim=False)
            fn = fn.sum(-1, keepdim=False)
            fp = fp.sum(-1, keepdim=False)

        tp += self.tp_bias

        result = 1 - ((1 + self.beta * self.beta) * tp) / (
            (1 + self.beta * self.beta) * tp
            + self.beta * self.beta * fn
            + fp
            + self.eps
        )

        if self.reduction != "none":

            if self.weight is not None:

                if not torch.is_tensor(self.weight):
                    self.weight = torch.tensor(self.weight)
                self.weight = self.weight.float()
                self.weight = self.weight.to(device=input_.device)

                weight = self.weight.expand_as(result)

                result = result * weight

            if self.ignore_index is not None:

                for cls in sorted(self.ignore_index, reverse=True):
                    if cls == result.shape[1] - 1:
                        result = result[:, :-1]
                    elif cls == 0:
                        result = result[:, 1:]
                    else:
                        result = torch.cat([result[:, :cls], result[:, cls + 1 :]], 1)

            if self.reduction == "mean":
                result = torch.mean(result)
            elif self.reduction == "sum":
                result = torch.sum(result)
            else:
                raise ValueError(
                    "reduction must be 'none', 'mean' or 'sum', but is {}".format(
                        self.reduction
                    )
                )

        return result


class DiceLoss(FScoreLoss):
    """FScoreLoss with beta=1."""

    def __init__(self, **kwargs):
        kwargs["beta"] = 1.0
        super(DiceLoss, self).__init__(**kwargs)

    def forward(self, input_, target):
        return super(DiceLoss, self).forward(input_, target)


class CrossEntropyDiceLoss(DiceLoss):
    """Weighted sum of CE and Dice losses.

    Expects softmax inputs!

    Args:
        ce_weight: The weight for the CE loss. Weight for the Dice loss will be
            1 - ce_weight.

    """

    def __init__(self, ce_weight: float = 0.5, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.ce_weight = ce_weight
        self.ce = nn.NLLLoss(weight=kwargs.get("weight", None))

    def forward(self, input_: torch.tensor, target: torch.tensor) -> torch.tensor:
        """Forward pass through the loss.

        Args:
            input_: Input with expected shape (B, C, ...) with C the number of classes.
            target: Target, either onehot with shape (B, C, ...), or not with shape
                either (B, ...) or (B, 1, ...). Make sure to set the make_onehot and
                make_onehot_newaxis arguments appropriately.

        Returns:
            The loss.

        """

        dice = super().forward(input_, target)

        if self.make_onehot:
            if self.make_onehot_newaxis:
                # already correct format
                pass
            else:
                # remove channel axis
                target = target.squeeze()
        else:
            target = torch.argmax(target, 1, keepdim=False)
        target = target.long()

        ce = self.ce(torch.log(input_ + self.eps), target)

        return (1 - self.ce_weight) * dice + self.ce_weight * ce