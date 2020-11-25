
import torch
import torch.nn as nn
import src.dl.losses.losses as losses
from typing import List, Dict, Optional
from src.dl.losses.joint_losses import JointLoss, JointInstLoss, JointPanopticLoss


class LossBuilder:
    def __init__(self,
                 class_types: str,
                 edge_weights: bool = True,
                 aux_branch: Optional[str] = None) -> None:
        """
        Initializes the loss function for instance or panoptic segmentation.
        This uses the loss functions available in the losses.py and parses the
        args in config.py to build a loss functions from the args specified for
        the segmentation task.

        Args:
            class_types (str): one of "instance" or "panoptic"
            edge_weight (float): weight to be applied to nuclei edges
            aux_branch (str, optional): one of ("hover", "micro", None)
        """
        self.class_types = class_types
        self.edge_weights = edge_weights
        self.aux_branch = aux_branch

        self.loss_lookup = {
            "Iou": "IoULoss",
            "DICE": "DiceLoss",
            "Tversky": "TverskyLoss",
            "wCE": "WeightedCELoss",
            "wSCE": "WeightedSCELoss",
            "wFocal": "WeightedFocalLoss",
            "hover":"HoVerLoss",
            "MS-SSIM": "MSSSIM"
        }

        self.joint_losses = [
            "IoU_wCE",
            "IoU_wSCE",
            "DICE_wCE",
            "DICE_wSCE",
            "DICE_wFocal",
            "DICE_wFocal_MS-SSIM",
            "Tversky_wCE",
            "Tversky_wSCE",
            "Tversky_wFocal",
            "Tversky_wFocal_MS-SSIM",
        ]

    @classmethod
    def set_loss(cls,
                 loss_name_inst: str,
                 class_types: str,
                 edge_weights: bool = True,
                 loss_name_type: Optional[str] = None,
                 loss_weights: Optional[List[float]] = [1.0, 1.0, 1.0],
                 binary_weights: Optional[torch.Tensor] = None,
                 type_weights: Optional[torch.Tensor] = None,
                 aux_branch: Optional[str] = None,
                 **kwargs) -> nn.Module:
        """
        Initialize the loss function.

        Args:
            loss_name_inst (str): one of "wCE", "symmetric_wCE", "IoU_wCE", "IoU_symmetric_wCE", "DICE_wCE",
                                 "DICE_symmetric_wCE", "DICE_wFocal_MS-SSIM", "Tversky_wFocal_MS-SSIM"
            class_types (str): one of "instance" or "panoptic"
            edge_weight: (float): weight to be applied to nuclei edges
            loss_name_type (str): one of the elements in self.joint_losses. Optionally you can set
                                  the type_loss to be a different loss func than the `loss_name_inst`.
                                  If this arg is not provided then the type loss will be the same as
                                  the inst_loss if panoptic segmentation is the segmentation task.
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

        c = cls(class_types, edge_weights, aux_branch)

        # set up kwargs
        kwargs = kwargs.copy()
        kwargs.setdefault("edge_weights", edge_weights)
        kwargs.setdefault("class_weights", binary_weights)

        # get losses that are mentioned in the inst_loss_name
        loss_keys_inst = solve_loss_key(loss_name_inst, c.joint_losses)
        loss_names_inst = [c.loss_lookup[key] for key in loss_keys_inst]

        if c.class_types == "panoptic":

            # Set instance segmentation branch loss
            loss_inst = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_inst])

            # Take off the nuclei edge weights from the type loss
            kwargs["edge_weights"] = False

            # Set the class weights. Can be None
            kwargs["class_weights"] = type_weights

            # set semantic loss if it was defined
            if loss_name_type is not None:
                loss_keys_type = solve_loss_key(loss_name_type, c.joint_losses)
                loss_names_type = [c.loss_lookup[key] for key in loss_keys_type]
            else:
                loss_names_type = loss_names_inst

            # Set semantic seg branch loss
            loss_type = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_type])

            # set auxilliary loss if that aux branch is used
            loss_aux = None
            if c.aux_branch == "hover":
                loss_aux = losses.HoVerLoss(**kwargs)

            loss = JointPanopticLoss(loss_inst, loss_type, loss_aux, loss_weights)

        elif c.class_types == "instance":

            # set instance seg loss
            loss_inst = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_inst])

            # set auxilliary branch loss
            loss_aux = None
            if c.aux_branch == "hover":
                loss_aux = losses.HoVerLoss(**kwargs)

            loss = JointInstLoss(loss_inst, loss_aux)

        return loss
