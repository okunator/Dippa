
import torch
import src.dl.losses as losses

from torch import nn
from typing import List, Dict, Optional
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss, _WeightedLoss


class JointLoss(nn.Module):
    def __init__(self,
                 losses: List[nn.Module],
                 weights: Optional[List[float]] = None) -> None:
        """
        Takes in a list of nn.Module losses and computes the loss for each loss
        in the list and at the end sums the outputs together as one joint loss.

        Args:
            losses (List[nn.Module]): List of losses found in losses.py
            weights (List[float]): List of weights for each loss

        Returns:
            torch.Tensor: computed joint loss from summed from losses in the input List
        """
        super().__init__()
        if weights is not None:
            assert all(
                0 <= val <= 1.0 for val in weights), "Weights need to be 0 <= weight <= 1"
        self.losses = losses
        self.weights = weights

    def forward(self, **kwargs):
        if self.weights is not None:
            l = list(zip(self.losses, self.weigths))
        else:
            l = list(zip(self.losses, [1.0]*len(self.losses)))
        losses = torch.stack([loss(**kwargs)*weight for loss, weight in l])
        return torch.sum(losses)


class JointPanopticLoss(_WeightedLoss):
    def __init__(self,
                 inst_loss: nn.Module,
                 type_loss: nn.Module,
                 aux_loss: Optional[nn.Module] = None,
                 loss_weights: Optional[List[float]] = [1.0, 1.0]) -> None:
        """
        Combines two losses: one from instance segmentation branch and another 
        from semantic segmentation branch to one joint loss.

        Args:
            inst_loss (nn.Module): loss function for the instance segmentation head
            type_loss (nn.Module): loss function for the semantic segmentation head
            aux_loss (nn.Module): loss function for the auxilliary regression head
            loss_weights (List[float]): List of weights for loss functions of instance,
                                        semantic and auxilliary branches in this order.
                                        If there is no auxilliary branch such as HoVer-branch
                                        then only two weights are needed.
        """
        assert 1 < len(
            loss_weights) <= 3, f"Too many weights in the loss_weights list: {loss_weights}"
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
        l1 = self.inst_loss(
            yhat=yhat_inst, target=target_inst, target_weight=target_weight,
            edge_weight=edge_weight, **kwargs
        )
        l2 = self.type_loss(
            yhat=yhat_type, target=target_type, target_weight=target_weight,
            edge_weight=edge_weight, **kwargs
        )
        return tw*l1 + iw*l2


class LossBuilder:
    def __init__(self,
                 class_types: str,
                 edge_weights: bool = True) -> None:
        """
        Initializes the loss function for instance or panoptic segmentation.
        This uses the loss functions available in the losses.py and parses the
        args in config.py to build a loss functions from the args specified for
        the segmentation task.

        Args:
            class_types: (str): one of "instance" or "panoptic"
            edge_weight: (float): weight to be applied to nuclei edges
        """
        self.class_types = class_types
        self.edge_weights = edge_weights
        self.loss_lookup = {
            "Iou": "IoULoss",
            "DICE": "DiceLoss",
            "Tversky": "TverskyLoss",
            "wCE": "WeightedCELoss",
            "wSCE": "WeightedSCELoss",
            "wFocal": "WeightedFocalLoss"
        }
        self.joint_losses = [
            "IoU_wCE",
            "IoU_wSCE",
            "DICE_wCE",
            "DICE_wSCE",
            "DICE_wFocal",
            "Tversky_wCE",
            "Tversky_wSCE",
            "Tversky_wFocal",
        ]

    @classmethod
    def set_loss(cls,
                 loss_name_inst: str,
                 class_types: str,
                 edge_weights: bool = True,
                 loss_name_type: Optional[str] = None,
                 loss_weights: Optional[List[float]] = [1.0, 1.0],
                 binary_weights: Optional[torch.Tensor] = None,
                 type_weights: Optional[torch.Tensor] = None,
                 **kwargs) -> nn.Module:
        """
        Initialize the loss function.

        Args:
            loss_name_inst (str): one of "wCE", "symmetric_wCE", "IoU_wCE", "IoU_symmetric_wCE", 
                                  "DICE_wCE", "DICE_symmetric_wCE"
            class_types (str): one of "instance" or "panoptic"
            edge_weight: (float): weight to be applied to nuclei edges
            loss_name_type (str): one of one of "wCE", "symmetric_wCE", "IoU_wCE", "IoU_symmetric_wCE", 
                                  "DICE_wCE", "DICE_symmetric_wCE". Optionally you can set the type_loss
                                   to be a different loss func than the `loss_name_inst`. If this arg is
                                   not provided then the type loss will be the same as the inst_loss if
                                   panoptic segmentation is the segmentation task.
            loss_weights (List[float]): List of weights for loss functions of instance,
                            semantic and auxilliary branches in this order.
                            If there is no auxilliary branch such as HoVer-branch
                            then only two weights are needed.
            binary_weights (Optional[torch.Tensor]): Tensor of size 2. Weights for background
                                                     and foreground.
            type_weights (Optional[torch.Tensor]): Tensor of size C. Each slot indicates
                                                   the weight to be applied for each class
        """
        def solve_loss_key(key: str, joint_losses: List[str]) -> List[str]:
            if key in joint_losses:
                loss_keys = key.split("_")
            else:
                loss_keys = [key]
            return loss_keys

        c = cls(class_types, edge_weights)

        # set up kwargs
        kwargs = kwargs.copy()
        kwargs.setdefault("edge_weights", edge_weights)
        kwargs.setdefault("class_weights", binary_weights)

        # get losses that are mentioned in the inst_loss_name
        loss_keys_inst = solve_loss_key(loss_name_inst, c.joint_losses)
        loss_names_inst = [c.loss_lookup[key] for key in loss_keys_inst]

        if c.class_types == "panoptic":

            # Set instance segmentation branch loss
            loss_inst = JointLoss([losses.__dict__[cl_key](**kwargs)
                                   for cl_key in loss_names_inst])

            # Take off the nuclei edge weights from the type loss
            kwargs["edge_weights"] = False

            # Set the class weights. Can be None
            kwargs["class_weights"] = type_weights

            # set type loss if it was defined
            if loss_name_type is not None:
                loss_keys_type = solve_loss_key(loss_name_type, c.joint_losses)
                loss_names_type = [c.loss_lookup[key]
                                   for key in loss_keys_type]
            else:
                loss_names_type = loss_names_inst

            # Use JointLoss to define the loss
            loss_type = JointLoss([losses.__dict__[cl_key](**kwargs)
                                   for cl_key in loss_names_type])
            loss = JointPanopticLoss(loss_inst, loss_type, loss_weights)
        elif c.class_types == "instance":
            loss = JointLoss([losses.__dict__[cl_key](**kwargs)
                              for cl_key in loss_names_inst])
        return loss
