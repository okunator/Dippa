import torch
import pytest

from typing import List

from src.dl.models.modules import *
from src.dl.models.decoders.long_skips.utils import long_skip_func


up = FixedUnpool(scale_factor=2)


def _get_in_sample(in_channels: int, in_dim: int) -> torch.Tensor:
    """
    Get a dummy tensor sample emulating the output of previous decoder 
    stage 
    """

    return torch.ones([1, in_channels, in_dim, in_dim])


def _get_encoder_skip_samples(
        skip_chls: List[int],
        out_dims: List[int]
    ) -> List[torch.Tensor]:
    """
    Get a dummy sample of encoder side skip tensors
    """
    skip_dims = out_dims[1:-1]
    encoder_skips = [torch.ones([1, c, d, d]) for c, d in zip(skip_chls, skip_dims)]

    return encoder_skips


def _get_extra_skip_samples(
        ix: int,
        long_skip: str,
        channels: List[int],
        dims: List[int],
    ) -> List[torch.Tensor]:
    """
    Get a dummy sample of the possible extra skips that are needed in
    the skip modules.
    """
    if long_skip == "unet3+":
        dec_dims = dims[:-1]
        dec_dims[0] *= 2
        dec_channels = channels[:-1]
        extra_skips = [torch.ones([1, dec_channels[i], dec_dims[i], dec_dims[i]]) for i in range(ix)]
    elif long_skip == "unet++":
        pass
    else:
        extra_skips = None

    return extra_skips


@pytest.mark.parametrize("stage_ix", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dec_channels", [[64, 32, 16, 8, 8, 2]])
@pytest.mark.parametrize("dec_out_dims", [[8, 16, 32, 64, 128, 256]])
@pytest.mark.parametrize("skip_channels", [[224, 96, 64, 32]])
@pytest.mark.parametrize("short_skip", ["residual", None])
@pytest.mark.parametrize("conv_block_type", ["basic", "dws", "mbconv", "fusedmbconv", "bottleneck"])
@pytest.mark.parametrize("long_skip", ["unet", "unet3+"])
@pytest.mark.parametrize("merge_policy", ["summation", "concatenate"])
@pytest.mark.parametrize("normalization", ["bn", "bcn"])
@pytest.mark.parametrize("activation", ["leaky-relu"])
@pytest.mark.parametrize("preactivate", [True, False])
@pytest.mark.parametrize("weight_standardize", [True, False])
@pytest.mark.parametrize("attention", [None, "se"])
def test_forward(
        stage_ix: int,
        dec_channels: List[int],
        skip_channels: List[int],
        dec_out_dims: List[int],
        short_skip: str,
        conv_block_type: str,
        long_skip: str,
        merge_policy: str,
        normalization: str,
        activation: str,
        weight_standardize: bool,
        preactivate: bool,
        attention: str
    ) -> None:
    """
    Test the forward method of one skip block.

    For each decoder stage the skip connection gets an upsampled input
    from the previous decoder stage.
    """

    in_channels = dec_channels[stage_ix]
    in_dim = dec_out_dims[stage_ix]
    sample = up(_get_in_sample(in_channels, in_dim))
    skips = _get_encoder_skip_samples(skip_channels, dec_out_dims)
    extra_skips = _get_extra_skip_samples(stage_ix, long_skip, dec_channels, dec_out_dims)

    skip = long_skip_func(
        name=long_skip,
        stage_ix=stage_ix,
        in_channels=in_channels,
        dec_channels=dec_channels,
        skip_channels=skip_channels,
        dec_out_dims=dec_out_dims,
        short_skip=short_skip,
        conv_block_type=conv_block_type,
        long_skip=long_skip,
        merge_policy=merge_policy,
        normalization=normalization,
        activation=activation,
        weight_standardize=weight_standardize,
        preactivate=preactivate,
        attention=attention
    )

    with torch.no_grad():
        out, extras = skip(x=sample, ix=stage_ix, skips=skips, extra_skips=extra_skips)

    assert out.shape[1] == skip.out_channels
    assert out.shape[2:] == sample.shape[2:]

    if extras:
        assert all([s.shape == t.shape for s, t in zip(extra_skips, extras)])


@pytest.mark.parametrize("stage_ix", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dec_channels", [[64, 32, 16, 8, 8, 2]])
@pytest.mark.parametrize("dec_out_dims", [[8, 16, 32, 64, 128, 256]])
@pytest.mark.parametrize("skip_channels", [[224, 96, 64, 32]])
@pytest.mark.parametrize("short_skip", ["residual", None])
@pytest.mark.parametrize("conv_block_type", ["basic", "dws", "mbconv", "fusedmbconv", "bottleneck"])
@pytest.mark.parametrize("long_skip", ["unet3+"]) # no need to test unet skip 
@pytest.mark.parametrize("merge_policy", ["summation", "concatenate"])
@pytest.mark.parametrize("normalization", ["bn", "bcn"])
@pytest.mark.parametrize("activation", ["leaky-relu"])
@pytest.mark.parametrize("preactivate", [True, False])
@pytest.mark.parametrize("weight_standardize", [True, False])
@pytest.mark.parametrize("attention", [None, "se"])
def test_forward_backward(
        stage_ix: int,
        dec_channels: List[int],
        skip_channels: List[int],
        dec_out_dims: List[int],
        short_skip: str,
        conv_block_type: str,
        long_skip: str,
        merge_policy: str,
        normalization: str,
        activation: str,
        weight_standardize: bool,
        preactivate: bool,
        attention: str
    ) -> None:
    """
    Test the forward and backward method of one skip block.

    For each decoder stage the skip connection gets an upsampled input
    from the previous decoder stage.
    """

    in_channels = dec_channels[stage_ix]
    in_dim = dec_out_dims[stage_ix]
    sample = up(_get_in_sample(in_channels, in_dim))
    skips = _get_encoder_skip_samples(skip_channels, dec_out_dims)
    extra_skips = _get_extra_skip_samples(stage_ix, long_skip, dec_channels, dec_out_dims)

    skip = long_skip_func(
        name=long_skip,
        stage_ix=stage_ix,
        in_channels=in_channels,
        dec_channels=dec_channels,
        skip_channels=skip_channels,
        dec_out_dims=dec_out_dims,
        short_skip=short_skip,
        conv_block_type=conv_block_type,
        long_skip=long_skip,
        merge_policy=merge_policy,
        normalization=normalization,
        activation=activation,
        weight_standardize=weight_standardize,
        preactivate=preactivate,
        attention=attention
    )

    out, _ = skip(x=sample, ix=stage_ix, skips=skips, extra_skips=extra_skips)
    out.mean().backward()
