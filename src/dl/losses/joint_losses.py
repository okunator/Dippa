import torch
import torch.nn as nn
from typing import List, Dict, Optional


class JointLoss(nn.ModuleDict):
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
            torch.Tensor: computed joint loss from summed losses in the input List
        """
        super().__init__()
        if weights is not None:
            assert all(0 <= val <= 1.0 for val in weights), "Weights need to be 0 <= weight <= 1"

        self.weights = weights
        for i in range(len(losses)):
            self.add_module('loss%d' % (i + 1), losses[i])

    def forward(self, **kwargs):
        if self.weights is not None:
            l = list(zip(self.values(), self.weigths))
        else:
            l = list(zip(self.values(), [1.0]*len(self.values())))

        losses = torch.stack([loss(**kwargs)*weight for loss, weight in l])
        return torch.sum(losses)


class MultiTaskLoss(nn.Module):
    def __init__(self,
                 inst_loss: nn.Module,
                 type_loss: Optional[nn.Module] = None,
                 aux_loss: Optional[nn.Module] = None,
                 loss_weights: Optional[List[float]] = None) -> None:
        """
        Combines losses for different branches to  a single multi-taks loss function

        Args:
            inst_loss (nn.Module): 
                Loss function for the instance segmentation head
            type_loss (nn.Module, optional, default=None): 
                Loss function for the semantic segmentation head
            aux_loss (nn.Module, optional, default=None): 
                Loss function for the auxilliary regression head
            loss_weights (List[float], optional, default=None): 
                List of weights for loss functions of instance,
                type and auxilliary branches in this order. If there is no auxilliary
                branch such as HoVer-branch then only two weights are needed.
        """
        super().__init__()
        self.inst_loss = inst_loss
        self.type_loss = type_loss
        self.aux_loss = aux_loss
        
        self.weights = [1.0, 1.0, 1.0]
        if loss_weights is not None:
            assert 1 < len(loss_weights) <= 3, f"Too many weights in the loss_weights list: {loss_weights}"
            self.weights = loss_weights

    def forward(self,
                yhat_inst: torch.Tensor,
                target_inst: Optional[torch.Tensor] = None,
                yhat_type: Optional[torch.Tensor] = None,
                target_type: Optional[torch.Tensor] = None,
                yhat_aux: Optional[torch.Tensor] = None,
                target_aux: Optional[torch.Tensor] = None,
                target_weight: Optional[torch.Tensor] = None,
                edge_weight: Optional[float] = 1.1,
                **kwargs) -> torch.Tensor:
        """
        Computes the joint loss when objective is to do panoptic segmentation

        Args:
            yhat_inst (torch.Tensor): 
                output of instance segmentation decoder branch. Shape: (B, 2, H, W) 
            target_inst (torch.Tensor, optional, default=None): 
                ground truth annotations for instance segmentation. Shape: (B, H, W)
            yhat_type (torch.Tensor, optional, default=None): 
                output of semantic segmetnation decoder branch. Shape: (B, C, H, W)
            target_type (torch.Tensor, optional, default=None): 
                ground truth annotaions for semantic segmentation. Shape: (B, H, W)
            yhat_aux (torch.Tensor, optional, default=None): 
                aux predictions. Shape: (B, 2, H, W)
            target_aux (torch.Tensor, optional, default=None): 
                aux maps ground truth. Shape: (B, H, W)
            target_weight (torch.Tensor, optional, default=None): 
                weight map for nuclei borders. Shape (B, H, W)
            edge_weight (float, optional, default=1.1): 
                weight applied at the nuclei borders. (edge_weight**target_weight)
        """

        iw = self.weights[0]
        tw = self.weights[1]
        aw = self.weights[2]

        loss = self.inst_loss(
            yhat=yhat_inst, target=target_inst, target_weight=target_weight, edge_weight=edge_weight, **kwargs
        )
        loss *= iw

        if self.type_loss:
            l2 = self.type_loss(
                yhat=yhat_type, target=target_type, target_weight=target_weight, edge_weight=edge_weight, **kwargs
            )
            loss += tw*l2

        if self.aux_loss:
            l3 = self.aux_loss(yhat=yhat_aux, target=target_aux, target_inst=target_inst, **kwargs)
            loss += l3*aw

        return loss