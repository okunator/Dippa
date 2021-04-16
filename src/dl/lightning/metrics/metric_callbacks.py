import torch
import torch.nn as nn
import pytorch_lightning as pl

from .functional import accuracy, iou


class Accuracy(pl.metrics.Metric):
    def __init__(self, dist_sync_on_step: bool=False) -> None:
        """
        Simple pytorch lightning accuracy metric callback. 
        uses the custom accuracy func which utilizes the conf mat
        and takes in an activation function to convert to probs.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("batch_accuracies", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor, activation: str="sofmax") -> None:
        """
        Compute the accuracy of the batch and add to 
        """
        batch_acc = accuracy(pred, target, activation)
        self.batch_accuracies += batch_acc
        self.n_batches += 1 

    def compute(self) -> torch.Tensor:
        return self.batch_accuracies / self.n_batches


class MeanIoU(pl.metrics.Metric):
    def __init__(self, dist_sync_on_step=False) -> None:
        """
        Simple pytorch lightning mIoU metric callback. 
        uses the custom accuracy func which utilizes the conf mat
        and takes in an activation function to convert to probs.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("batch_ious", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor, activation: str="sofmax") -> torch.Tensor:
        """
        Compute 
        """
        batch_iou = iou(pred, target, activation)
        self.batch_ious += batch_iou.mean()
        self.n_batches += 1

    def compute(self) -> torch.Tensor:
        return self.batch_ious / self.n_batches