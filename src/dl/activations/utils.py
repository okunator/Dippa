import torch
import torch.nn as nn
import src.dl.activations as act


def convert_relu_to_mish(model: nn.Module):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, act.Mish(inplace=False))
        else:
            convert_relu_to_mish(child)


def convert_relu_to_swish(model: nn.Module):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, act.Swish(inplace=False))
        else:
            convert_relu_to_swish(child)