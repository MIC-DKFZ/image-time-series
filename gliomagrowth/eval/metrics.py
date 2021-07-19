import numpy as np
import torch
from typing import Union, Optional, Iterable


class ConfusionMatrix:
    """A confusion matrix object.

    The main idea is to store TP, FP, TN, FN values, so we can compute multiple metrics.

    Args:
        test: The segmentation to be tested.
        reference: The reference segmentation.
        batch_axes: Elements along these axes will be treated separately.

    """

    def __init__(
        self,
        test: Optional[Union[torch.Tensor, np.ndarray]] = None,
        reference: Optional[Union[torch.Tensor, np.ndarray]] = None,
        batch_axes: Optional[Union[int, Iterable[int]]] = None,
    ):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)
        if type(batch_axes) == int:
            self.batch_axes = [
                batch_axes,
            ]
        elif batch_axes is None:
            self.batch_axes = []
        else:
            self.batch_axes = batch_axes

    @staticmethod
    def assert_shape(
        test: Union[torch.Tensor, np.ndarray],
        reference: Union[torch.Tensor, np.ndarray],
    ):
        """Convenience method to assert that test and reference have same shapes.

        Args:
            test: The segmentation to be tested.
            reference: The reference segmentation.

        """

        assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
            test.shape, reference.shape
        )

    def set_test(self, test: Union[torch.Tensor, np.ndarray]):
        """Set the test segmentation.

        Args:
            test: The segmentation to be tested.

        """

        self.test = test
        self.reset()

    def set_reference(self, reference: Union[torch.Tensor, np.ndarray]):
        """Set the reference segmentation.

        Args:
            reference: The reference segmentation.

        """

        self.reference = reference
        self.reset()

    def reset(self):
        """Remove all stored values."""

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):
        """Fill stored TP, FP, TN, FN and existence values.

        We can also work with fuzzy values. The segmentations are just expected to have
        values between 0 and 1. "not x" is calculated as "1 - x".
        """

        if self.test is None or self.reference is None:
            raise ValueError(
                "'test' and 'reference' must both be set to compute confusion matrix."
            )

        self.assert_shape(self.test, self.reference)

        aggregate_axes = tuple(
            i for i in range(self.reference.ndim) if i not in self.batch_axes
        )
        self.size = np.product([self.reference.shape[i] for i in aggregate_axes])

        if isinstance(self.test, torch.Tensor):

            self.test_empty = self.test
            self.test_full = self.test
            self.reference_empty = self.reference
            self.reference_full = self.reference
            for i in sorted(aggregate_axes, reverse=True):
                self.test_empty = self.test_empty.any(i)
                self.test_full = self.test_full.all(i)
                self.reference_empty = self.reference_empty.any(i)
                self.reference_full = self.reference_full.all(i)
            self.test_empty = torch.logical_not(self.test_empty)
            self.reference_empty = torch.logical_not(self.reference_empty)

            self.tp = torch.logical_and(self.test, self.reference).sum(aggregate_axes)
            self.fp = torch.logical_and(
                self.test, torch.logical_not(self.reference)
            ).sum(aggregate_axes)
            self.tn = torch.logical_and(
                torch.logical_not(self.test), torch.logical_not(self.reference)
            ).sum(aggregate_axes)
            self.fn = torch.logical_and(
                torch.logical_not(self.test), self.reference
            ).sum(aggregate_axes)

        else:

            self.test_empty = ~self.test.any(axis=aggregate_axes)
            self.test_full = self.test.all(axis=aggregate_axes)
            self.reference_empty = ~self.reference.any(axis=aggregate_axes)
            self.reference_full = self.reference.all(axis=aggregate_axes)

            self.tp = np.logical_and(self.test, self.reference).sum(aggregate_axes)
            self.fp = np.logical_and(self.test, np.logical_not(self.reference)).sum(
                aggregate_axes
            )
            self.tn = np.logical_and(
                np.logical_not(self.test), np.logical_not(self.reference)
            ).sum(aggregate_axes)
            self.fn = np.logical_and(np.logical_not(self.test), self.reference).sum(
                aggregate_axes
            )

    def get_matrix(self) -> Iterable[Union[torch.Tensor, np.ndarray, np.number]]:
        """Get the TP, FP, TN, FN values."""

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self) -> np.int64:
        """Get the total number of elements in the reference segmentation."""

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self) -> Iterable[Union[torch.Tensor, np.ndarray, np.bool_]]:
        """Get existence values, i.e. emptiness und fullness of segmentations."""

        for case in (
            self.test_empty,
            self.test_full,
            self.reference_empty,
            self.reference_full,
        ):
            if case is None:
                self.compute()
                break

        return (
            self.test_empty,
            self.test_full,
            self.reference_empty,
            self.reference_full,
        )


def dice(
    test: Optional[Union[torch.Tensor, np.ndarray]] = None,
    reference: Optional[Union[torch.Tensor, np.ndarray]] = None,
    batch_axes: Optional[Union[int, Iterable[int]]] = None,
    confusion_matrix: Optional[ConfusionMatrix] = None,
    nan_for_nonexisting: bool = True,
    **kwargs
) -> Union[torch.Tensor, np.ndarray, np.number]:
    """Compute Dice score: 2TP / (2TP + FP + FN).

    Args:
        test: The segmentation to be tested.
        reference: The reference segmentation.
        batch_axes: Elements along these axes will be treated separately.
        confusion_matrix: Provide a confusion matrix instead of segmentations. If it
            exists and has existing .test and .reference attributes, separately provided
            test and reference will be ignored.
        nan_for_nonexisting: Return NaN if both test and reference are empty.
            Will return 0 otherwise.

    Returns:
        The Dice score(s).

    """

    if (
        confusion_matrix is None
        or confusion_matrix.test is None
        or confusion_matrix.reference is None
    ):
        confusion_matrix = ConfusionMatrix(test, reference, batch_axes=batch_axes)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    result = 2.0 * tp / (2 * tp + fp + fn)

    if not nan_for_nonexisting:
        if isinstance(result, torch.Tensor):
            result[torch.isnan(result)] = 0.0
        elif isinstance(result, np.ndarray):
            result[np.isnan(result)] = 0.0
        else:
            result = 0.0

    return result