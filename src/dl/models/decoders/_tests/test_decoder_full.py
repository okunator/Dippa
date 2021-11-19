import torch
import pytest

from typing import List, Tuple

from src.dl.models.modules import *
from src.dl.models.decoders import Decoder
from src.dl.models.encoders import TimmUniversalEncoder


def _get_enc_feature_samples(
        encoder_name: str,
        input_size: int
    ) -> Tuple[List[torch.Tensor], int]:
    """
    Get a dummy tensors emulating the output features of an encoder
    """
    enc = TimmUniversalEncoder(
        name=encoder_name,
        pretrained=False,
        features_only=False
    )
    
    feats = [
        torch.ones([1, enc.out_channels[i], input_size // 2**i, input_size // 2**i])
        for i in range(len(enc.out_channels))
    ]
    
    return feats, enc.out_channels


@pytest.mark.slow
@pytest.mark.parametrize("encoder_name", ["tf_efficientnetv2_s", "resnet50"])
@pytest.mark.parametrize("model_input_size", [64])
@pytest.mark.parametrize("dec_channels", [[96, 64, 32, 16, 8]])
@pytest.mark.parametrize("n_blocks", [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("short_skip", ["residual", None])
@pytest.mark.parametrize("conv_block_types", [["basic", "dws", "mbconv", "fusedmbconv", "bottleneck"]])
@pytest.mark.parametrize("long_skip", ["unet", "unet3+"])
@pytest.mark.parametrize("merge_policy", ["summation", "concatenate"])
@pytest.mark.parametrize("normalization", ["bn"])
@pytest.mark.parametrize("activation", ["leaky-relu"])
@pytest.mark.parametrize("weight_standardize", [True, False])
@pytest.mark.parametrize("attention", [None, "se"])
def test_forward(
        encoder_name: str,
        model_input_size: int,
        dec_channels: List[int],
        conv_block_types: List[str],
        n_blocks: List[int],
        n_layers: int,
        short_skip: str,
        long_skip: str,
        merge_policy: str,
        normalization: str,
        activation: str,
        weight_standardize: bool,
        attention: str
    ) -> None:
    """
    Test the forward method of the decoder.
    """

    features, enc_channels = _get_enc_feature_samples(encoder_name, model_input_size)

    decoder = Decoder(
        enc_channels=enc_channels,
        model_input_size=model_input_size,
        dec_channels=dec_channels,
        short_skip=short_skip,
        conv_block_types=conv_block_types,
        n_blocks=n_blocks,
        n_layers=n_layers,
        long_skip=long_skip,
        long_skip_merge_policy=merge_policy,
        normalization=normalization,
        activation=activation,
        weight_standardize=weight_standardize,
        attention=attention
    )

    with torch.no_grad():
        out = decoder(*features)

    assert out.shape[1] == decoder.out_channels
    assert tuple(out.shape[2:]) == (model_input_size, model_input_size)
    
    
@pytest.mark.slow
@pytest.mark.parametrize("encoder_name", ["tf_efficientnetv2_s", "resnet50"])
@pytest.mark.parametrize("model_input_size", [64])
@pytest.mark.parametrize("dec_channels", [None])
@pytest.mark.parametrize("n_blocks", [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("short_skip", ["residual", None])
@pytest.mark.parametrize("conv_block_types", [["basic", "dws", "mbconv", "fusedmbconv", "bottleneck"]])
@pytest.mark.parametrize("long_skip", ["unet", "unet3+"])
@pytest.mark.parametrize("merge_policy", ["summation", "concatenate"])
@pytest.mark.parametrize("normalization", ["bcn"])
@pytest.mark.parametrize("activation", ["leaky-relu"])
@pytest.mark.parametrize("weight_standardize", [True, False])
@pytest.mark.parametrize("attention", [None, "se"])
def test_forward_backward(
        encoder_name: str,
        model_input_size: int,
        dec_channels: List[int],
        conv_block_types: List[str],
        n_blocks: List[int],
        n_layers: int,
        short_skip: str,
        long_skip: str,
        merge_policy: str,
        normalization: str,
        activation: str,
        weight_standardize: bool,
        attention: str
    ) -> None:
    """
    Test the forward and backward method of the decoder.
    """

    features, enc_channels = _get_enc_feature_samples(encoder_name, model_input_size)

    decoder = Decoder(
        enc_channels=enc_channels,
        model_input_size=model_input_size,
        dec_channels=dec_channels,
        short_skip=short_skip,
        conv_block_types=conv_block_types,
        n_blocks=n_blocks,
        n_layers=n_layers,
        long_skip=long_skip,
        long_skip_merge_policy=merge_policy,
        normalization=normalization,
        activation=activation,
        weight_standardize=weight_standardize,
        attention=attention
    )

    out = decoder(*features)
    out.mean().backward()
