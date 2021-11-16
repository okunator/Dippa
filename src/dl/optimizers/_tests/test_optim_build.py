import pytest
import torch
from typing import Tuple

from src.config import TEST_CONFIG
from src.dl.losses.utils import multitaskloss_func
from src.dl.models._base._multitask_model import MultiTaskSegModel
from src.dl.optimizers.utils import OptimizerBuilder


def _get_dummy_img(input_size: int):
    return torch.ones([1, 3, input_size, input_size])


def _get_dummy_target(
        n_classes: int,
        input_size: int
    ) -> Tuple[torch.Tensor]:
    
    dummy_target = torch.zeros([1, input_size, input_size]).long()
    
    for i, c in enumerate(range(n_classes)):  
        dummy_target[..., (i*c*2)+10:(i*c*2)+20, (i*c*2)+30:(i*c*2)+40] += 1
        dummy_target[..., (i*c*2)+15:(i*c*2)+25, (i*c*2)+15:(i*c*2)+25] += 1
        dummy_target[dummy_target > c] = c
        
    return dummy_target


@pytest.mark.parametrize("lookahead", [True, False])
@pytest.mark.parametrize("bias_wd", [True, False])
@pytest.mark.parametrize(
    "name", [
        "adam", "adamp", "rmsprop", "adadelta", "sgd", "apollo", "adabelief",
        "adagrad", "adamax", "adamw", "asgd", "accsgd", "adabound", "adamod",
        "diffgrad", "lamb", "novograd", "pid", "qhadam", "qhm", "radam",
        "sgdw", "yogi", "ranger", "rangerqh", "rangerva"
    ])
def test_optim_build(
        name: str,
        lookahead: bool,
        bias_wd: float
    ) -> None:
    """
    Test the optimizer builder and optimizer step
    """
    model = MultiTaskSegModel.from_conf(TEST_CONFIG)
    decoder_lr = 0.0003
    encoder_lr = 0.0003
    decoder_wd = 0.0003
    encoder_wd = 0.0003
    
    optimizer = OptimizerBuilder.set_optimizer(
        model=model,
        optimizer_name=name,
        lookahead=lookahead,
        decoder_learning_rate=decoder_lr,
        encoder_learning_rate=encoder_lr,
        decoder_weight_decay=decoder_wd,
        encoder_weight_decay=encoder_wd,
        bias_weight_decay=bias_wd
    )
    
    img = _get_dummy_img(TEST_CONFIG.model.input_size)
    targets = {}
    for br, n in TEST_CONFIG.model.decoder.branches.items():
        targets[f"{br}_map"] = _get_dummy_target(n, TEST_CONFIG.model.input_size)
     
    losses = TEST_CONFIG.training.loss.branch_losses   
    mtl = multitaskloss_func(losses, edge_weights=None, class_weights=None)
    
    # Run one training step
    optimizer.zero_grad()
    output = model(img)
    loss = mtl(output, targets)
    loss.backward()
    optimizer.step()
    
    