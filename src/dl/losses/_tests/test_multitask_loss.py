import pytest
import torch
from typing import Tuple, Dict
from collections import OrderedDict
import scipy.ndimage as ndi

from src.utils import get_weight_map
from src.dl.losses.utils import multitaskloss_func


def _get_dummy_pair(n_classes: int) -> Tuple[torch.Tensor]:
    dummy_yhat = torch.zeros([1, n_classes, 64, 64]).float()
    dummy_target = torch.zeros([1, 64, 64]).long()
    
    for i, c in enumerate(range(dummy_yhat.shape[1]), 1):  
        dummy_yhat[..., c, (i*c)+10:(i*c)+20, (i*c)+30:(i*c)+40] += 1
        dummy_yhat[..., c, (i*c)+10:(i*c)+20, (i*c)+30:(i*c)+40] += 1
        dummy_target[..., (i*c*2)+10:(i*c*2)+20, (i*c*2)+30:(i*c*2)+40] += 1
        dummy_target[..., (i*c*2)+15:(i*c*2)+25, (i*c*2)+15:(i*c*2)+25] += 1
        dummy_target[dummy_target > c] = c
        
    return dummy_yhat, dummy_target


def _get_dummy_weight_map(target: torch.Tensor) -> torch.Tensor:   
    wmap = get_weight_map(ndi.label(target)[0][0])
    wmap = torch.tensor(wmap).unsqueeze(0)
    return wmap


def _get_dummy_class_weights(n_classes: int) -> torch.Tensor:
    w = torch.zeros(1, n_classes)
    w[0][0] += 0.1
    
    for i in range(1, n_classes):
        w[0][i] += 0.9+i*0.01
    
    return w[0, ...]

@torch.no_grad()
@pytest.mark.parametrize("n_classes", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("class_weights", [False, True])
@pytest.mark.parametrize("edge_weights", [False, True])
@pytest.mark.parametrize("losses", [
    {"inst": "tversky_ce"}, {"inst": "tversky_ce", "type": "tversky_focal"},
    {"inst": "tversky_ce", "type": "tversky_focal", "aux": "mse_ssim"},
    {"type": "tversky_focal_iou_dice", "aux": "mse_ssim_ce"},
    {"type": "tversky_focal_iou_dice", "aux": "mse_ssim_ce", "sem": "tversky"},
])
def test_multitask_loss(
        n_classes: int,
        losses: Dict[str, str],
        edge_weights: Dict[str, float],
        class_weights: Dict[str, bool]
    ) -> None:
    """
    Test the mutlitask losses with different joint losses
    """
    losses = OrderedDict(losses)
    
    yhats = {}
    targets = {}
    class_wd = {} if class_weights else None
    edge_wd = {} if edge_weights else None
    for i, br in enumerate(losses.keys(), 1):
        yhats[f"{br}_map"], targets[f"{br}_map"] = _get_dummy_pair(n_classes+i)
        
        if edge_weights:
            edge_wd[br] = 1.1
            
        if class_weights:
            class_wd[br] = _get_dummy_class_weights(n_classes+i)
    
    if edge_weights:
        targets["weight_map"] = _get_dummy_weight_map(targets[f"{br}_map"])
        
    mtl = multitaskloss_func(
        losses, edge_weights=edge_wd, class_weights=class_wd
    )
        
    mtl(yhats, targets)
    
    