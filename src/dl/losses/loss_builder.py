
import torch
import torch.nn as nn
import src.dl.losses as losses
from typing import List, Dict, Optional
from src.dl.losses.joint_losses import JointLoss, JointInstLoss, JointPanopticLoss


loss_lookup = {
    "iou": "IoULoss",
    "dice": "DiceLoss",
    "tversky": "TverskyLoss",
    "ce": "WeightedCELoss",
    "sce": "WeightedSCELoss",
    "focal": "WeightedFocalLoss",
    "mse": "MSE",
    "gmse": "GradMSE",
    "ssim": "SSIM",
    "msssim": "MSSSIM"
}

joint_seg_losses = [
    "iou",
    "dice",
    "tversky",
    "ce",
    "sce",
    "focal",
    "iou_ce",
    "iou_sce",
    "iou_focal",
    "dice_ce",
    "dice_sce",
    "dice_focal",
    "tversky_ce",
    "tversky_sce",
    "tversky_focal",
    "iou_ce_ssim",
    "iou_sce_ssim",
    "iou_focal_ssim",
    "dice_ce_ssim",
    "dice_sce_ssim",
    "dice_focal_ssim",
    "tversky_ce_ssim",
    "tversky_sce_ssim",
    "tversky_focal_ssim",
    "iou_ce_msssim",
    "iou_sce_msssim",
    "iou_focal_msssim",
    "dice_ce_msssim",
    "dice_sce_msssim",
    "dice_focal_msssim",
    "tversky_ce_msssim",
    "tversky_sce_msssim",
    "tversky_focal_msssim",
]

joint_aux_losses = [
    "mse",
    "gmse",
    "ssim",
    "msssim",
    "mse_ssim",
    "mse_gmse",
    "mse_msssim",
    "gmse_ssim",
    "gmse_msssim",
    "ssim_msssim",
    "mse_gmse_ssim",
    "mse_gmse_msssim"
]


class LossBuilder:
    def __init__(self,
                 class_types: str,
                 aux_branch_name: Optional[str] = None) -> None:
        """
        Initializes the loss function for instance or panoptic segmentation.
        This uses the loss functions available in the src/dl/losses and parses the
        args in config.py to build a loss functions from the args specified for
        the segmentation task.

        Args:
            class_types (str): one of "instance" or "panoptic"
            aux_branch_name (str, optional): one of ("hover", "micro", None)
        """
        self.class_types = class_types
        self.aux_branch = aux_branch_name


    def solve_loss_key(self, key: str, joint_losses: List[str]) -> List[str]:
        if key in joint_losses:
            loss_keys = key.split("_")
        else:
            loss_keys = [key]
        return loss_keys


    @classmethod
    def set_loss(cls,
                 class_types: str,
                 loss_name_inst: str,
                 loss_name_type: Optional[str] = None,
                 loss_name_aux: Optional[str] = None,
                 loss_weights: Optional[List[float]] = [1.0, 1.0, 1.0],
                 binary_weights: Optional[torch.Tensor] = None,
                 type_weights: Optional[torch.Tensor] = None,
                 edge_weight: Optional[float] = None,
                 aux_branch_name: Optional[str] = None,
                 **kwargs) -> nn.Module:
        """
        Initialize the joint loss function.

        Args:
            class_type (str): One of ("panoptic", "instance") adds type branch loss to joint loss if "panoptic"
            loss_name_inst (str): the inst branch loss name. This is defined in config.py
            loss_name_type (str): the type branch loss name. This is defined in config.py
            loss_weights (List[float]): List of weights for loss functions of instance,
                                        semantic and auxilliary branches in this order.
            binary_weights (torch.Tensor): Tensor of size (2, ). Weights for background and foreground.
            type_weights (torch.Tensor): Tensor of size (C, ). Each slot indicates
                                         the weight to be applied for each class
            edge_weight (float): weight given at the nuclei edges
            aux_branch_name (str): one of ("hover", "micro", None)
        """
        assert class_types in ("instance", "panoptic")
        assert loss_name_inst in joint_seg_losses, f"loss_name_inst need to be one of {joint_seg_losses}"
        assert loss_name_type in joint_seg_losses, f"loss_name_type need to be one of {joint_seg_losses}"
        assert loss_name_aux in joint_aux_losses, f"loss_name_aux need to be one of {joint_aux_losses}"
        assert aux_branch_name in (None, "hover", "micro"), "aux_branch name need to be one of (None, 'hover', 'micro')"

        c = cls(class_types, aux_branch_name)

        # set up kwargs
        kwargs = kwargs.copy()
        kwargs.setdefault("class_weights", binary_weights)
        kwargs.setdefault("edge_weight", edge_weight)

        # get losses that are mentioned in the inst_loss_name
        loss_keys_inst = c.solve_loss_key(loss_name_inst, joint_seg_losses)
        loss_names_inst = [loss_lookup[key] for key in loss_keys_inst]

        if c.class_types == "instance":
            # set instance seg loss
            loss_inst = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_inst])

            # set auxilliary branch loss
            loss_aux = None
            if aux_branch_name is not None:
                loss_keys_aux = c.solve_loss_key(loss_name_aux, joint_aux_losses)
                loss_names_aux = [loss_lookup[key] for key in loss_keys_aux]
                loss_aux = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_aux])

            loss = JointInstLoss(loss_inst, loss_aux)

        elif c.class_types == "panoptic":
        
            # Set instance segmentation branch loss
            loss_inst = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_inst])

            # Take off the nuclei edge weights from the type loss
            kwargs["edge_weight"] = None

            # Set the class weights.
            kwargs["class_weights"] = type_weights

            # set semantic loss
            loss_keys_type = c.solve_loss_key(loss_name_type, joint_seg_losses)
            loss_names_type = [loss_lookup[key] for key in loss_keys_type]

            # Set semantic seg branch loss
            loss_type = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_type])

            # set auxilliary loss if that aux branch is used
            loss_aux = None
            if aux_branch_name is not None:
                loss_keys_aux = c.solve_loss_key(loss_name_aux, joint_aux_losses)
                loss_names_aux = [loss_lookup[key] for key in loss_keys_aux]
                loss_aux = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_aux])

            loss = JointPanopticLoss(loss_inst, loss_type, loss_aux, loss_weights)

        return loss

