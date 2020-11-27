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
            torch.Tensor: computed joint loss from summed from losses in the input List
        """
        super().__init__()
        if weights is not None:
            assert all(
                0 <= val <= 1.0 for val in weights), "Weights need to be 0 <= weight <= 1"

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


class JointInstLoss(nn.Module):
    def __init__(self,                
                 inst_loss: nn.Module,
                 aux_loss: Optional[nn.Module] = None,
                 loss_weights: Optional[List[float]] = [1.0, 1.0]) -> None:
        """
        Abstraction for instance segmentation loss. If aux branch is used 
        then aux branch loss is also included to the joint loss.

        Args:
            inst_loss (nn.Module): loss function for the instance segmentation head
            aux_loss (nn.Module): loss function for the auxilliary regression head
            loss_weights (List[float]): List of weights for loss functions of instance,
                                        semantic and auxilliary branches in this order.
                                        If there is no auxilliary branch such as HoVer-branch
                                        then only two weights are needed.
        """
        super().__init__()
        assert 1 < len(loss_weights) <= 2, f"Too many weights in the loss_weights list: {loss_weights}"
        self.inst_loss = inst_loss
        self.aux_loss = aux_loss
        self.weights = loss_weights

    def forward(self,
                yhat_inst: torch.Tensor,
                target_inst: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None,
                edge_weight: Optional[float] = 1.1,
                yhat_aux: Optional[torch.Tensor] = None,
                target_aux: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Computes the joint loss when objective is to do instance segmentation

        Args:
            yhat_inst (torch.Tensor): output of instance segmentation decoder branch. Shape: (B, 2, H, W)
            target_inst (torch.Tensor): ground truth annotations for instance segmentation. Shape: (B, H, W)
            target_weight (torch.Tensor, optional): weight map for nuclei borders. Shape (B, H, W)
            edge_weight (float, optional): weight applied at the nuclei borders. (edge_weight**target_weight)
            yhat_aux (torch.Tensor, optional): aux predictions. Shape: (B, 2, H, W)
            target_aux (torch.Tensor, optional): aux maps ground truth. Shape: (B, H, W)
        """

        iw = self.weights[0]
        aw = self.weights[1]

        loss = self.inst_loss(
            yhat=yhat_inst, target=target_inst, target_weight=target_weight, edge_weight=edge_weight, **kwargs
        )
        loss = loss*iw

        if self.aux_loss is not None:
            aux_loss = self.aux_loss(yhat=yhat_aux, target=target_aux, target_inst=target_inst, **kwargs)
            loss += aux_loss*aw
        
        return loss


class JointPanopticLoss(nn.Module):
    def __init__(self,
                 inst_loss: nn.Module,
                 type_loss: nn.Module,
                 aux_loss: Optional[nn.Module] = None,
                 loss_weights: Optional[List[float]] = [1.0, 1.0, 1.0]) -> None:
        """
        Combines two losses: one from instance segmentation branch and another 
        from semantic segmentation branch to one joint loss. If aux branch is used 
        then third aux branch loss is also included

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
                yhat_aux: Optional[torch.Tensor] = None,
                target_aux: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Computes the joint loss when objective is to do panoptic segmentation

        Args:
            yhat_inst (torch.Tensor): output of instance segmentation decoder branch. Shape: (B, 2, H, W) 
            yhat_type (torch.Tensor): output if semantic segmetnation decoder branch. Shape: (B, C, H, W)
            target_inst (torch.Tensor): ground truth annotations for instance segmentation. Shape: (B, H, W)
            target_type (torch.Tensor): ground truth annotaions for semantic segmentation. Shape: (B, H, W)
            target_weight (torch.Tensor, optional): weight map for nuclei borders. Shape (B, H, W)
            edge_weight (float, optional): weight applied at the nuclei borders. (edge_weight**target_weight)
            yhat_aux (torch.Tensor, optional): aux predictions. Shape: (B, 2, H, W)
            target_aux (torch.Tensor, optional): aux maps ground truth. Shape: (B, H, W)
        """

        iw = self.weights[0]
        tw = self.weights[1]
        aw = self.weights[2]

        l1 = self.inst_loss(
            yhat=yhat_inst, target=target_inst, target_weight=target_weight, edge_weight=edge_weight, **kwargs
        )

        l2 = self.type_loss(
            yhat=yhat_type, target=target_type, target_weight=target_weight, edge_weight=edge_weight, **kwargs
        )

        loss = tw*l1 + iw*l2

        if self.aux_loss is not None:
            l3 = self.aux_loss(yhat=yhat_aux, target=target_aux, target_inst=target_inst, **kwargs)
            loss += l3*aw

        return loss