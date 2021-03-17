import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def confusion_mat(yhat: torch.Tensor, 
                  target: torch.Tensor, 
                  activation: Optional[str] = None) -> torch.Tensor:
    """
    Computes confusion matrix from the soft mask and target tensor

    Args:
    ----------
        yhat (torch.Tensor): 
            the soft mask from the network of shape (B, C, H, W)
        target (torch.Tensor): 
            the target matrix of shape (B, H, W)
        activation (str, optional, default=1): 
            apply sigmoid or softmax activation before taking argmax

    Returns:
    ----------
        torch.Tensor of shape (B, num_classes, num_classes)
    """

    yhat_soft = yhat
    if activation is not None:
        assert activation in ("sigmoid", "softmax"), f"activation: {activation} sigmoid and softmax allowed."
        if activation == "sigmoid":
            yhat_soft = torch.sigmoid(yhat)
        elif activation == "softmax":
            yhat_soft = F.softmax(yhat, dim=1)
            
    
    n_classes = yhat_soft.shape[1]
    batch_size = yhat_soft.shape[0]
    bins = target + torch.argmax(yhat_soft, dim=1)*n_classes
    bins_vec = bins.view(batch_size, -1)

    confusion_list = []
    for iter_id in range(batch_size):
        pb = bins_vec[iter_id]
        bin_count = torch.bincount(pb, minlength=n_classes**2)
        confusion_list.append(bin_count)

    confusion_vec = torch.stack(confusion_list)
    confusion_mat = confusion_vec.view(batch_size, n_classes, n_classes).to(torch.float32)

    return confusion_mat


def iou(yhat: torch.Tensor, 
        target: torch.Tensor,
        activation: Optional[str] = None,
        eps: Optional[float] = 1e-7) -> torch.Tensor:
    """
    Compute the per class intersection over union for dense predictions

    Args:
    -----------
        yhat (torch.Tensor): 
            the soft mask from the network of shape (B, C, H, W)
        target (torch.Tensor): 
            the target matrix of shape (B, H, W)
        activation (str, optional, default=None): 
            apply sigmoid or softmax activation before taking argmax
        eps (float, optional, default=1e-7)
            small constant to avoid zero div error

    Returns:
    ------------
        torch.Tensor of shape (B, num_classes, num_classes)
    """
    conf_mat = confusion_mat(yhat, target, activation)
    rowsum = torch.sum(conf_mat, dim=1) # [(TP + FP), (FN + TN)]
    colsum = torch.sum(conf_mat, dim=2) # [(TP + FN), (FP + TN)]
    diag = torch.diagonal(conf_mat, dim1=-2, dim2=-1) # [TP, TN]
    denom = rowsum + colsum - diag # [(TP + FN + FP), (TN + FN + FP)]
    ious = (diag + eps) / (denom + eps) # [(TP/(TP + FN + FP)), (TN/(TN + FN + FP))]
    return ious


def accuracy(yhat: torch.Tensor,
             target: torch.Tensor,
             activation: Optional[str] = None,
             eps: float = 1e-7) -> torch.Tensor:
    """
    Compute the per class accuracy for dense predictions

    Args:
    -----------
        yhat (torch.Tensor): 
            the soft mask from the network of shape (B, C, H, W)
        target (torch.Tensor): 
            the target matrix of shape (B, H, W)
        activation (str, optional, default=None): 
            apply sigmoid or softmax activation before taking argmax

    Returns:
    ------------
        torch.Tensor of shape (B, num_classes, num_classes)
    """
    conf_mat = confusion_mat(yhat, target, activation)
    diag = torch.diagonal(conf_mat, dim1=-2, dim2=-1) # batch diagonal
    denom = conf_mat.sum()
    accuracies = (eps + diag) / (eps + denom)
    return accuracies
