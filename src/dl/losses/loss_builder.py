
import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import List, Dict, Optional

import src.dl.losses as losses
from src.dl.losses.joint_losses import JointLoss, MultiTaskLoss


class LossBuilder:
    def __init__(self,
                 decoder_branches: DictConfig,
                 loss_args: DictConfig) -> None: 
        """
        Initializes the multi-task loss function for instance segmentation.
        This uses the loss functions available in the src/dl/losses.Builds 
        a loss functions from the args specified for the segmentation task
        Makes use of the experiment.yml

        Args:
            decoder_branches (omegaconf.DictConfig):
                Arguments related to multi-task architecture:
                specifically how to set up model deocder branches
            loss_args (omegaconf.DictConfig):
                Arguments related to the different loss functions
                at different branches 
        """
        self.type_branch: bool = decoder_branches.type
        self.aux_branch: bool = decoder_branches.aux
        self.aux_branch_type: str = decoder_branches.aux_type
        self.inst_branch_loss: str = loss_args.inst_branch_loss
        self.type_branch_loss: str = loss_args.type_branch_loss
        self.aux_branch_loss: str = loss_args.aux_branch_loss
        self.edge_weight: bool = loss_args.edge_weight
        self.class_weights: bool = loss_args.class_weights

    def solve_loss_key(self, key: str, joint_losses: List[str]) -> List[str]:
        if key in joint_losses:
            loss_keys = key.split("_")
        else:
            loss_keys = [key]
        return loss_keys

    @classmethod
    def set_loss(cls,
                 decoder_branches: DictConfig,
                 loss_args: DictConfig,
                 loss_weights: Optional[List[float]] = None,
                 binary_weights: Optional[torch.Tensor] = None,
                 type_weights: Optional[torch.Tensor] = None,
                 edge_weight: Optional[float] = 1.1,
                 **kwargs) -> nn.Module:
        """
        Initialize the joint loss function.
        
        Args:
            decoder_branches (omegaconf.DictConfig):
                Arguments related to multi-task architecture:
                specifically how to set up model deocder branches
            loss_args (omegaconf.DictConfig):
                Arguments related to the different loss functions
                at different branches 
            loss_weights (List[float], optional): 
                List of weights for loss functions of instance, 
                semantic and auxilliary branches in this order.
            binary_weights (torch.Tensor): 
                Tensor of size (2, ). Weights for background
                 and foreground.
            type_weights (torch.Tensor): 
                Tensor of size (C, ). Each slot indicates the 
                weight to be applied for each class
            edge_weight (float): 
                Weight given at the nuclei edges
        """
        c = cls(decoder_branches, loss_args)

        # set inst branch loss
        kwargs = kwargs.copy()
        kwargs["edge_weight"] = edge_weight if c.edge_weight else None
        kwargs["class_weights"] = binary_weights if c.class_weights else None
        
        assert c.inst_branch_loss in losses.JOINT_SEG_LOSSES, f"inst_branch_loss need to be one of {losses.JOINT_SEG_LOSSES}"
        loss_keys_inst = c.solve_loss_key(c.inst_branch_loss, losses.JOINT_SEG_LOSSES)
        loss_names_inst = [losses.LOSS_LOOKUP[key] for key in loss_keys_inst]
        loss_inst = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_inst])

        # set auxilliary branch loss
        loss_aux = None
        if c.aux_branch:
            assert c.aux_branch_loss in losses.JOINT_AUX_LOSSES, f"aux_branch_loss need to be one of {losses.JOINT_AUX_LOSSES}"
            loss_keys_aux = c.solve_loss_key(c.aux_branch_loss, losses.JOINT_AUX_LOSSES)
            loss_names_aux = [losses.LOSS_LOOKUP[key] for key in loss_keys_aux]
            loss_aux = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_aux])
    
        # set type branch loss
        loss_type = None
        if c.type_branch:
            kwargs["edge_weight"] =  None
            kwargs["class_weights"] = type_weights if c.class_weights else None
            assert c.type_branch_loss in losses.JOINT_SEG_LOSSES, f"type_branch_loss need to be one of {losses.JOINT_SEG_LOSSES}"
            loss_keys_type = c.solve_loss_key(c.type_branch_loss, losses.JOINT_SEG_LOSSES)
            loss_names_type = [losses.LOSS_LOOKUP[key] for key in loss_keys_type]
            loss_type = JointLoss([losses.__dict__[cl_key](**kwargs) for cl_key in loss_names_type])

        loss = MultiTaskLoss(loss_inst, loss_type, loss_aux, loss_weights)
        return loss

