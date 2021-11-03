import torch
import torch.nn as nn

from src.dl.models.modules.conv.blocks import *
import pytest


def _get_sample(in_channels: int) -> torch.Tensor:
    return torch.ones([1, in_channels, 32, 32])
    

@pytest.mark.parametrize("preactivate", [True, False])
@pytest.mark.parametrize("weight_standardize", [True, False])
@pytest.mark.parametrize("normalization", ["bn", "bcn"])
@pytest.mark.parametrize("in_channels", [32, 64, 32])
@pytest.mark.parametrize("out_channels", [64, 32, 32])
@pytest.mark.parametrize("n_blocks", [1, 2, 3])
@pytest.mark.parametrize(
    "block", [
        ResidualBlock, DenseBlock, BasicBlock, BottleneckResidualBlock,
        DepthWiseSeparableResidualBlock, DepthWiseSeparableBasicBlock,
        MobileInvertedResidualBlock, FusedMobileInvertedResidualBlock
    ]
)
def test_forward(
        block: nn.Module,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        preactivate: bool,
        weight_standardize: bool,
        normalization: str,
        **kwargs
    ) -> None:
    """
    Test the forward pass of the conv blocks with different param sets
    """
    sample = _get_sample(in_channels=in_channels)

    conv_block = block(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=n_blocks,
        preactivate=preactivate,
        weight_standardize=weight_standardize,
        normalization=normalization,
        **kwargs
    )

    conv_block.eval()

    with torch.no_grad():
        out = conv_block(sample)

    assert out.shape[1] == conv_block.out_channels
    assert out.shape[2:] == sample.shape[2:]


@pytest.mark.parametrize("preactivate", [True, False])
@pytest.mark.parametrize("weight_standardize", [True, False])
@pytest.mark.parametrize("normalization", ["bn", "bcn"])
@pytest.mark.parametrize("in_channels", [32, 64, 32])
@pytest.mark.parametrize("out_channels", [64, 32, 32])
@pytest.mark.parametrize("n_blocks", [1, 2, 3])
@pytest.mark.parametrize(
    "block", [
        ResidualBlock, DenseBlock, BasicBlock, BottleneckResidualBlock,
        DepthWiseSeparableResidualBlock, DepthWiseSeparableBasicBlock,
        MobileInvertedResidualBlock, FusedMobileInvertedResidualBlock
    ]
)
def test_forward_backward(
        block: nn.Module,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        preactivate: bool,
        weight_standardize: bool,
        normalization: str,
        **kwargs
    ) -> None:
    """
    Test the model forward + backward methods
    """
    sample = _get_sample(in_channels=in_channels)

    conv_block = block(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=n_blocks,
        preactivate=preactivate,
        weight_standardize=weight_standardize,
        normalization=normalization,
        **kwargs
    )

    out = conv_block(sample)
    out.mean().backward()

    assert out.shape[1] == conv_block.out_channels
    assert out.shape[2:] == sample.shape[2:]

