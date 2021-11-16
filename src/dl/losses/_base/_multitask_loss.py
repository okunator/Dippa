import torch
import torch.nn as nn
from typing import Dict, Optional

from ._joint_loss import JointLoss


class MultiTaskLoss(nn.ModuleDict):
    def __init__(
            self,
            branch_losses: Dict[str, JointLoss],
            loss_weights: Optional[Dict[str, float]]=None,
            **branch_kwargs
        ) -> None:
        """
        Combines losses for different branches to a single multi-taks 
        loss function.

        Args:
        ----------
            branch_losses (Dict[str, JointLoss]):
                Dictionary of branch names mapped to JointLoss modules.
                e.g. {"inst": JointLoss(MSE(), Dice())}
            loss_weights (Dict[str, float], optional, default=None): 
                Dictionary of branch names mapped to the weight used for
                that branch loss
        """
        super().__init__()
        
        self.weights = {k: 1.0 for k in branch_losses.keys()}
        if loss_weights is not None:
            if len(loss_weights) != len(branch_losses):
                raise ValueError(f"""
                    Got {len(loss_weights)} loss weights and {len(branch_losses)}
                    branches. Need to have the same length."""
            )
            if not all(k in branch_losses.keys() for k in loss_weights.keys()):
                raise ValueError(f"""
                    Mismatching keys in `loss_weights` and `branch_losses`"""
                )
            else:
                self.weights = loss_weights
                
        for branch, loss in branch_losses.items():
            self.add_module(f"{branch}_loss", loss) 

    def forward(
            self,
            yhats: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            **kwargs
        ) -> torch.Tensor:
        """
        Computes the joint loss of the multi-task network

        Args:
        ------------
            yhats (Dict,[str, torch.Tensor]):
                Dictionary of branch name mapped to the predicted mask
                of that branch.
            targets (Dict,[str, torch.Tensor]):
                Dictionary of branch name mapped to the GT mask of that
                branch.

        Returns:
        ------------
            torch.Tensor: computed multi-task loss (Scalar).
        """
        weight_map = None
        if "weight_map" in targets.keys():
            weight_map = targets["weight_map"]
            
        multitask_loss = 0.0
        for branch, loss in self.items():
            branch = branch.split("_")[0]
            branch_loss = loss(
                yhat=yhats[f"{branch}_map"],
                target=targets[f"{branch}_map"],
                target_weight=weight_map,
            )
            multitask_loss += branch_loss*self.weights[branch]

        return multitask_loss
    
    def extra_repr(self) -> str:
        s = ('branch_loss_weights={weights}')
        return s.format(**self.__dict__)