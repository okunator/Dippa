import torch
from typing import List
from torch import nn
from torch.nn.modules.loss import _Loss

# adapted from: pytorch-toolbelt losses.joint_loss.py
class WeightedLoss(_Loss):
    """
    Wrapper class around CE loss function that applies weights with fixed factor.
    This class adds nuclei border weights to the final computed loss on a feature map generated
    from a H&E image.
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, yhat, y, edge_weight, y_weight):
        loss_matrix = self.loss(yhat, y)
        loss = (loss_matrix * (edge_weight**y_weight)).mean()
        return loss * self.weight


class JointCELoss(_Loss):
    """
    Wrap two CE loss functions into one. The first computes CE for binary instance segmentation task
    from inputs 'yhat_inst' and 'y_inst' and the second computes CE for semantic segmentation task
    from inputs 'yhat_type' and 'y_type'. Then a weighted sum of two losses is computed.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.bCE = WeightedLoss(first, first_weight)
        self.mCE = WeightedLoss(second, second_weight)

    def forward(self, yhat_inst, yhat_type, y_inst, y_type, edge_weight, y_weight):
        return self.bCE(yhat_inst, y_inst, edge_weight, y_weight) + \
            self.mCE(yhat_type, y_type, edge_weight, y_weight)
