import torch
import torch.nn as nn
from typing import List, Dict, Optional

from .criterions.utils import loss_func
from . import *


mtl = vars()


def multitaskloss_func(
        names: Dict[str, str],
        edge_weights: Optional[Dict[str, float]]=None,
        class_weights: Optional[Dict[str, torch.Tensor]]=None,
        **kwargs
    ) -> nn.Module:
    """
    Initialize the multitask loss function. 

    Args:
    -----------
        names (Dict[str, str]):
            A Dict of branch names mapped to a string specifying a
            jointloss. e.g. {"inst": "tversky_ce_ssim", "aux": "mse"}
        edge_weights (Dict[str, float], optional, default=None):
            A dictionary of baranch names mapped to floats that are used
            to weight nuclei edges in CE-based losses. e.g.
            {"inst": 1.1, "aux": None}
        class_weights (Dict[str, torch.Tensor], optional, default=None):
            A dictionary of branch names mapped to class weight tensors
            of shape (n_classes_branch, ).

    Returns:
    -----------
        nn.Module: Initialized nn.Module.
    """
    
    if edge_weights is not None:
        assert len(names) == len(edge_weights), (
            "Got differing number of branches for `edge_weights and `names`.",
            f"Got: {edge_weights.keys()} and {names.keys()}"
        )
        
    if class_weights is not None:
        assert len(names) == len(class_weights), (
            "Got differing number of branches for `class_weights and `names`.",
            f"Got: {class_weights.keys()} and {names.keys()}"
        )
        
    
    allowed_seg = mtl['JOINT_SEG_LOSSES']
    allowed_reg = mtl['JOINT_AUX_LOSSES']

    branch_losses = {}
    for branch, loss_name in names.items():
        allowed = allowed_seg if "aux" not in branch else allowed_reg
        
        assert loss_name in allowed, (
            f"Max number of losses in one JointLoss is currently 4. "
            f"Illegal loss given for branch: {branch}. Got: {loss_name}",
            f"Allowed losses: {allowed}."
        )
    
        try:
            losses = loss_name.split("_")
        except:
            losses = [loss_name]
        
        ew = None
        if edge_weights is not None: 
            ew = edge_weights[branch]
            
        cw = None
        if class_weights:
            cw = class_weights[branch]
            
        # all losses weighted uniformly
        branch_losses[branch] = JointLoss(
            [loss_func(l, edge_weight=ew, class_weights=cw) for l in losses]
        )    
    
    mtl_f = MultiTaskLoss(branch_losses, **kwargs)

    return mtl_f


__all__ = ["multitaskloss_func"]
