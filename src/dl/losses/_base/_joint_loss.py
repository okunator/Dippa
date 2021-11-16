import torch
import torch.nn as nn
from typing import List, Optional


class JointLoss(nn.ModuleDict):
    def __init__(
            self,
            losses: List[nn.Module],
            weights: Optional[List[float]]=None,
        ) -> None:
        """
        Takes in a list of nn.Module losses and computes the loss for 
        each loss in the list and at the end sums the outputs together 
        as one joint loss.

        Args:
        ----------
            losses (List[nn.Module]):
                List of initialized nn.Module losses
            weights (List[float], optional, default=None):
                List of weights for each loss            
        """
        super().__init__()
        if len(losses) > 4:
            raise ValueError(f"""
                Currently the max number of losses in one JointLoss is 4.
                Got: {len(losses)}. {losses}""" 
            )
        
        self.weights = [1.0]*len(losses)
        if weights is not None:
            if not all(0 <= val <= 1.0 for val in weights):
                raise ValueError(
                f"Weights need to be 0 <= weight <= 1. Got: {weights}"
            )
            self.weights = weights
        
        for i in range(len(losses)):
            self.add_module(f"loss{i + 1}", losses[i])

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Computes the joint-loss

        Returns:
        ----------
            torch.Tensor: computed joint loss from summed losses in the
            input List
        """
        lw = zip(self.values(), self.weights)
        losses = torch.stack([loss(**kwargs)*weight for loss, weight in lw])
        return torch.sum(losses)
    
    def extra_repr(self) -> str:
        s = ('loss_weights={weights}')
        return s.format(**self.__dict__)