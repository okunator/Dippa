import torch
import torch.nn as nn
from typing import List, Optional

import src.dl.losses as losses
loss_vars = vars(losses)


class LossBuilder:
    """
    Class used to initialize the multi-task segmentation loss function.
    This uses the losses available in the src/dl/losses and builds 
    a multi-tasks loss function from the given arguments
    """

    def solve_loss_key(self, key: str, joint_losses: List[str]) -> List[str]:
        if key in joint_losses:
            loss_keys = key.split("_")
        else:
            loss_keys = [key]
        return loss_keys

    @classmethod
    def set_loss(
            cls,
            decoder_type_branch: bool,
            decoder_sem_branch: bool,
            decoder_aux_branch: str,
            inst_branch_loss: str,
            type_branch_loss: str,
            sem_branch_loss: str,
            aux_branch_loss: str,
            loss_weights: Optional[List[float]]=None,
            binary_weights: Optional[torch.Tensor]=None,
            class_weights: Optional[torch.Tensor]=None,
            edge_weight: Optional[float]=None,
            **kwargs
        ) -> nn.Module:
        """
        Initialize the joint loss function.
        
        Args:
        ------------
            decoder_type_branch (bool):
                Flag to specify if the network contains a type 
                classification branch
            decoder_sem_branch (bool):
                Flag to specify if the network contains a type 
            decoder_aux_branch (str):
                One of ("hover", "contour", "dist", None). Specifies the
                aux branch type of the network. If None, network does 
                not contain an aux branch.
                classification branch
            inst_branch_loss (str):
                A string specifying the loss funcs used in the binary
                segmentation branch of the network. Loss names separated
                with underscores e.g. "ce_dice"
            type_branch_loss (str):
                A string specifying the loss funcs used in the cell type
                segmentation branch of the network. Loss names separated
                with underscores e.g. "ce_dice"
            aux_branch_loss (str):
                A string specifying the loss funcs used in the auxiliary
                regression branch of the network. Loss names separated 
                with underscores e.g. "mse_ssim"
            sem_branch_loss (str):
                A string specifying the loss funcs used in the semantic
                segmentation branch of the network. Loss names separated
                with underscores e.g. "ce_dice"
            loss_weights (List[float], optional, default=None): 
                List of weights for loss functions of instance, semantic
                and auxilliary branches in this order.
            binary_weights (torch.Tensor, optional, default=None): 
                Tensor of size (2, ). Weights for background and 
                foreground.
            class_weights (torch.Tensor, optional, default=None): 
                Tensor of size (C, ). Each slot indicates the weight to 
                be applied for each class
            edge_weight (float, optional, default=1.1): 
                Weight given at the nuclei edges

        Returns:
        ------------
            nn.Module: Initialized multitask loss function
        """
        c = cls()

        # set inst branch loss
        kwargs = kwargs.copy()
        kwargs["edge_weight"] = edge_weight
        kwargs["class_weights"] = binary_weights
        
        assert inst_branch_loss in losses.JOINT_SEG_LOSSES, (
            f"inst_branch_loss need to be one of {losses.JOINT_SEG_LOSSES}"
        )

        loss_keys_inst = c.solve_loss_key(
            inst_branch_loss, loss_vars["JOINT_SEG_LOSSES"]
        )
        loss_names_inst = [
            loss_vars["LOSS_LOOKUP"][key] for key in loss_keys_inst
        ]
        loss_list = [
            loss_vars[cl_key](**kwargs)
            for cl_key in loss_names_inst
        ]

        loss_inst = losses.JointLoss(loss_list)
        
        # set auxilliary branch loss
        loss_aux = None
        if decoder_aux_branch:
            assert aux_branch_loss in loss_vars["JOINT_AUX_LOSSES"], (
                "aux_branch_loss need to be one of:", 
                f"{loss_vars['JOINT_AUX_LOSSES']}"
            )

            loss_keys_aux = c.solve_loss_key(
                aux_branch_loss, loss_vars["JOINT_AUX_LOSSES"]
            )

            loss_names_aux = [
                loss_vars["LOSS_LOOKUP"][key]
                for key in loss_keys_aux
            ]

            loss_list = [
                loss_vars[cl_key](**kwargs) 
                for cl_key in loss_names_aux
            ]

            loss_aux = losses.JointLoss(loss_list)
    
        # set type branch loss
        loss_type = None
        if decoder_type_branch:
            kwargs["edge_weight"] =  None # Take off edge weights
            kwargs["class_weights"] = class_weights
            assert type_branch_loss in loss_vars["JOINT_SEG_LOSSES"], (
                "type_branch_loss need to be one of:", 
                f"{loss_vars['JOINT_SEG_LOSSES']}"
            )

            loss_keys_type = c.solve_loss_key(
                type_branch_loss, loss_vars["JOINT_SEG_LOSSES"]
            )
            loss_names_type = [
                loss_vars["LOSS_LOOKUP"][key] for key in loss_keys_type
            ]
            loss_list = [
                loss_vars[cl_key](**kwargs) 
                for cl_key in loss_names_type
            ]

            loss_type = losses.JointLoss(loss_list)

        # set semantic branch loss
        loss_sem = None
        if decoder_sem_branch:
            assert sem_branch_loss in loss_vars["JOINT_SEG_LOSSES"], (
                "sem_branch_loss need to be one of: ", 
                f"{loss_vars['JOINT_SEG_LOSSES']}"
            )

            loss_keys_sem = c.solve_loss_key(
                sem_branch_loss, loss_vars["JOINT_SEG_LOSSES"]
            )
            loss_names_sem = [
                loss_vars["LOSS_LOOKUP"][key] for key in loss_keys_sem
            ]
            loss_list = [
                loss_vars[cl_key](**kwargs) 
                for cl_key in loss_names_sem
            ]

            loss_sem = losses.JointLoss(loss_list)

        loss = losses.MultiTaskLoss(
            loss_inst, loss_type, loss_aux, loss_sem, loss_weights
        )

        return loss
