import torch
import torch.nn as nn
from src.dl.models.modules import Mish

def convert_relu_to_mish(model: nn.Module):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish(inplace=False))
        else:
            convert_relu_to_mish(child)