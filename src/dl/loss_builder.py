
import torch
import src.dl.losses as losses

from torch import nn
from typing import List, Dict, Optional
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss, _WeightedLoss


class JointPanopticLoss(_WeightedLoss):
    def __init__(self,
                 inst_loss: nn.Module,
                 type_loss: nn.Module,
                 aux_loss: Optional[nn.Module] = None,
                 loss_weights: Optional[List[float]] = [1.0, 1.0]) -> None:
        """
        Adds two weighted CE losses to one joint loss. When a classification decoder branch is added
        to any network, this loss will take in the instance branch and classification branch outputs
        and computes a weighted joint loss for the whole network to ensure convergence
        Args:
            inst_loss (nn.Module): loss function for the instance segmentation head
            type_loss (nn.Module): loss function for the semantic segmentation head
            aux_loss (nn.Module): loss function for the auxilliary regression head
            loss_weights (List[float]): List of weights for loss functions of instance,
                                        semantic and auxilliary branches in this order.
                                        If there is no auxilliary branch such as HoVer-branch
                                        then only two weights are needed.
        """
        assert 1 < len(loss_weights) <= 3, f"Too many weights in the loss_weights list: {loss_weights}" 
        super().__init__()
        self.inst_loss = inst_loss
        self.type_loss = type_loss
        self.aux_loss = aux_loss
        self.weights = loss_weights

    def forward(self,
                yhat_inst: torch.Tensor,
                yhat_type: torch.Tensor,
                target_inst: torch.Tensor,
                target_type: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None,
                edge_weight: Optional[float] = 1.1,
                **kwargs) -> torch.Tensor:

        iw = self.weights[0]
        tw = self.weights[1]
        l1 = self.inst_loss(yhat_inst, target_inst, edge_weight, target_weight, **kwargs)*iw
        l2 = self.type_loss(yhat_type, target_type, edge_weight, target_weight, **kwargs)*tw
        return l1 + l2


class LossBuilder:
    def __init__(self,
                 class_types: str,
                 edge_weights: bool = True) -> None:
        """
        Initializes the loss function for instance or panoptic segmentation.
        This uses the loss functions available in the losses.py.

        Args:
            class_types: (str): one of "instance" or "panoptic"
            edge_weight: (float): weight to be applied to nuclei edges
        """
        self.class_types = class_types
        self.edge_weights = edge_weights
        self.loss_lookup = {
            "wCE": "WeightedCELoss",
            "symmetric_wCE": "WeightedSCELoss",
            "IoU_wCE": "TODO",
            "IoU_symmetric_wCE": "TODO"
        }

    @classmethod
    def set_loss(cls,
                 loss_name: str,
                 class_types: str,
                 edge_weights: bool,
                 loss_weights: List[float] = [1.0, 1.0],
                 binary_weights: Optional[torch.Tensor] = None,
                 type_weights: Optional[torch.Tensor] = None,
                 **kwargs) -> nn.Module:
        """
        Initialize the loss function.

        Args:
            loss_name (str) : one of ("wCE", "symmetric_wCE", "IoU_wCE", "IoU_symmetric_wCE")
            class_types (str): one of "instance" or "panoptic"
            edge_weight: (float): weight to be applied to nuclei edges
            loss_weights (List[float]): List of weights for loss functions of instance,
                            semantic and auxilliary branches in this order.
                            If there is no auxilliary branch such as HoVer-branch
                            then only two weights are needed.
            binary_weights (Optional[torch.Tensor]): Tensor of size 2. Weights for background
                                                     and foreground.
            type_weights (Optional[torch.Tensor]): Tensor of size C. Each slot indicates
                                                   theweight to be applied for each class
        """
        c = cls(class_types, edge_weights)

        kwargs = kwargs.copy()
        kwargs.setdefault("edge_weights", edge_weights)
        kwargs.setdefault("class_weights", binary_weights)
        loss_key = c.loss_lookup[loss_name]
        if c.class_types == "panoptic":
            loss_inst = losses.__dict__[loss_key](**kwargs)
            kwargs["class_weights"] = type_weights
            loss_type = losses.__dict__[loss_key](**kwargs)
            loss = JointPanopticLoss(loss_inst, loss_type, loss_weights)
        elif c.class_types == "instance":
            loss = losses.__dict__[loss_key](**kwargs)
        return loss
