import torch
import pytest

from typing import List, Dict

from src.dl.models._base._multitask_model import MultiTaskSegModel


def _get_sample(input_size: int) -> torch.Tensor:
    """
    Get a dummy tensors emulating an input image
    """
    return torch.ones([1, 3, input_size, input_size])


@pytest.mark.parametrize("enc_name", ["tf_efficientnetv2_s", "resnet50"])
@pytest.mark.parametrize("enc_depth", [5])
@pytest.mark.parametrize("model_input_size", [64])
@pytest.mark.parametrize("dec_branches", [{"inst": 2}, {"inst": 2, "type": 4}, {"inst": 2, "type": 4, "aux":2}])
@pytest.mark.parametrize("dec_conv_types", [["basic", "mbconv", "basic", "dws", "dws"]])
@pytest.mark.parametrize("dec_channels", [[64, 64, 64, 64, 64]])
@pytest.mark.parametrize("dec_n_blocks", [[1, 1, 1, 1, 1]])
@pytest.mark.parametrize("dec_n_layers", [1, 2])
@pytest.mark.parametrize("dec_short_skip", ["residual", None])
@pytest.mark.parametrize("long_skip", ["unet", "unet3+"])
@pytest.mark.parametrize("merge_policy", ["summation", "concatenate"])
@pytest.mark.parametrize("normalization", ["bcn"])
@pytest.mark.parametrize("activation", ["leaky-relu"])
@pytest.mark.parametrize("weight_standardize", [True, False])
@pytest.mark.parametrize("attention", [None, "se"])
def test_forward(
        enc_name: str,
        enc_depth: int,
        dec_branches: Dict[str, int],
        dec_conv_types: List[str],
        dec_channels: List[int],
        dec_n_blocks: List[int],
        dec_n_layers: int,
        model_input_size: int,
        dec_short_skip: str,
        long_skip: str,
        merge_policy: str,
        normalization: str,
        activation: str,
        weight_standardize: bool,
        attention: str
    ) -> None:
    """
    Test the forward method of a multi-task seg model.
    """

    img = _get_sample(model_input_size)

    model = MultiTaskSegModel(
        enc_name=enc_name,
        enc_depth=enc_depth,
        model_input_size=model_input_size,
        dec_branches=dec_branches,
        dec_channels=dec_channels,
        dec_short_skip=dec_short_skip,
        dec_conv_types=dec_conv_types,
        dec_n_blocks=dec_n_blocks,
        dec_n_layers=dec_n_layers,
        dec_long_skip=long_skip,
        dec_long_skip_merge_policy=merge_policy,
        dec_normalization=normalization,
        dec_activation=activation,
        dec_weight_standardize=weight_standardize,
        dec_attention=attention
    )

    with torch.no_grad():
        out = model(img)

    for k, map in out.items():
        key = [dk for dk in dec_branches.keys() if dk in k][0]
        assert map.shape[1] == dec_branches[key]
        assert tuple(map.shape[2:]) == (model_input_size, model_input_size)
        
        
@pytest.mark.parametrize("enc_name", ["tf_efficientnetv2_s", "resnet50"])
@pytest.mark.parametrize("enc_depth", [5])
@pytest.mark.parametrize("model_input_size", [64])
@pytest.mark.parametrize("dec_branches", [{"inst": 2}, {"inst": 2, "type": 4}, {"inst": 2, "type": 4, "aux":2}])
@pytest.mark.parametrize("dec_conv_types", [["basic", "mbconv", "basic", "dws", "dws"]])
@pytest.mark.parametrize("dec_channels", [[64, 64, 64, 64, 64]])
@pytest.mark.parametrize("dec_n_blocks", [[1, 1, 1, 1, 1]])
@pytest.mark.parametrize("dec_n_layers", [1, 2])
@pytest.mark.parametrize("dec_short_skip", ["residual", None])
@pytest.mark.parametrize("dec_long_skip", ["unet", "unet3+"])
@pytest.mark.parametrize("dec_merge_policy", ["summation", "concatenate"])
@pytest.mark.parametrize("dec_normalization", ["bcn"])
@pytest.mark.parametrize("dec_activation", ["leaky-relu"])
@pytest.mark.parametrize("dec_weight_standardize", [True, False])
@pytest.mark.parametrize("dec_attention", [None, "se"])
def test_forward_backward(
        enc_name: str,
        enc_depth: int,
        dec_branches: Dict[str, int],
        dec_conv_types: List[str],
        dec_channels: List[int],
        dec_n_blocks: List[int],
        dec_n_layers: int,
        model_input_size: int,
        dec_short_skip: str,
        dec_long_skip: str,
        dec_merge_policy: str,
        dec_normalization: str,
        dec_activation: str,
        dec_weight_standardize: bool,
        dec_attention: str
    ) -> None:
    """
    Test the forward and backward method of a multi-task seg model.
    """

    img = _get_sample(model_input_size)

    model = MultiTaskSegModel(
        enc_name=enc_name,
        enc_depth=enc_depth,
        model_input_size=model_input_size,
        dec_branches=dec_branches,
        dec_channels=dec_channels,
        dec_short_skip=dec_short_skip,
        dec_conv_types=dec_conv_types,
        dec_n_blocks=dec_n_blocks,
        dec_n_layers=dec_n_layers,
        dec_long_skip=dec_long_skip,
        dec_long_skip_merge_policy=dec_merge_policy,
        dec_normalization=dec_normalization,
        dec_activation=dec_activation,
        dec_weight_standardize=dec_weight_standardize,
        dec_attention=dec_attention
    )

    out = model(img)
    
    # retain the graph here
    for map in out.values():
        map.mean().backward(retain_graph=True)
    
