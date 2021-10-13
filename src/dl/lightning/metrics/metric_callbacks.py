import torch
from torchmetrics import Metric
from typing import Optional, Any, Callable

from .functional import accuracy, iou


class Accuracy(Metric):
    def __init__(
            self,
            compute_on_step: bool=True,
            dist_sync_on_step: bool=False,
            progress_grouo: Optional[Any]=None,
            dist_sync_func: Optional[Callable]=None
        ) -> None:
        """
        Simple pytorch lightning accuracy metric callback. 
        uses the custom accuracy func which utilizes the conf mat
        and takes in an activation function to convert to probs.

        Args:
        --------
            compute_on_step (bool, default=True):
                 Forward only calls update() and returns None if this is
                  et to False. default: True
            dist_sync_on_step (bool, default=False):
                Synchronize computed values in distributed setting
            process_group (any, optional, default=None):
                Specify the process group on which synchronization is
                called. default: None (which selects the entire world)
            dist_sync_func (Callable, optional, default=None):
                Callback that performs the allgather operation on the
                metric state. When None, DDP will be used to perform the
                allgather.
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=progress_grouo,
            dist_sync_fn=dist_sync_func
        )

        self.add_state(
            "batch_accuracies", 
            default=torch.tensor(0.), 
            dist_reduce_fx="sum"
        )

        self.add_state(
            "n_batches", 
            default=torch.tensor(0), 
            dist_reduce_fx="sum"
        )

    def update(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            activation: Optional[str]="sofmax"
        ) -> None:
        """
        Update the batch accuracy list with one batch accuracy value

        Args:
        --------
            pred (torch.Tensor):
                Predicted output from the model. Shape (B, C, H, W).
            target (torch.Tensor):
                The ground truth segmentation tensor. Shape (B, H, W).
            activation (str, optional, default="softmax"):
                The activation function. One of: "softmax", "sigmoid" or
                None
        """
        batch_acc = accuracy(pred, target, activation)
        self.batch_accuracies += batch_acc
        self.n_batches += 1 

    def compute(self) -> torch.Tensor:
        """
        Compute the accuracy of one batch and normalize accordingly.

        Returns:
        ---------
            torch.Tensor: The accuracy value. Shape (1).
        """
        return self.batch_accuracies / self.n_batches


class MeanIoU(Metric):
    def __init__(
            self,
            compute_on_step: bool=True,
            dist_sync_on_step: bool=False,
            progress_grouo: Optional[Any]=None,
            dist_sync_func: Optional[Callable]=None
        ) -> None:
        """
        Simple pytorch lightning mIoU metric callback. 
        uses the custom accuracy func which utilizes the conf mat
        and takes in an activation function to convert to probs.

        Args:
        --------
            compute_on_step (bool, default=True):
                 Forward only calls update() and returns None if this is
                  et to False. default: True
            dist_sync_on_step (bool, default=False):
                Synchronize computed values in distributed setting
            process_group (any, optional, default=None):
                Specify the process group on which synchronization is
                called. default: None (which selects the entire world)
            dist_sync_func (Callable, optional, default=None):
                Callback that performs the allgather operation on the
                metric state. When None, DDP will be used to perform the
                allgather.
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=progress_grouo,
            dist_sync_fn=dist_sync_func
        )

        self.add_state(
            "batch_ious",
            default=torch.tensor(0.),
            dist_reduce_fx="sum"
        )

        self.add_state(
            "n_batches",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(
            self, 
            pred: torch.Tensor, 
            target: torch.Tensor, 
            activation: Optional[str]="sofmax"
        ) -> None:
        """
        Update the batch IoU list with one batch IoU matrix

        Args:
        --------
            pred (torch.Tensor):
                Predicted output from the model. Shape (B, C, H, W).
            target (torch.Tensor):
                The ground truth segmentation tensor. Shape (B, H, W).
            activation (str, optional, default="softmax"):
                The activation function. One of: "softmax", "sigmoid" or
                None
        """
        batch_iou = iou(pred, target, activation)
        self.batch_ious += batch_iou.mean()
        self.n_batches += 1

    def compute(self) -> torch.Tensor:
        """
        Normalize the batch IoU values.

        Returns:
        ---------
            torch.Tensor: The IoU mat. Shape (B, n_classes, n_classes).
        """
        return self.batch_ious / self.n_batches