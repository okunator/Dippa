"""
Copyright 2019 Ross Wightman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Got an error with this encoder when using the library so had to modify
# the source code a bit

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.coat import CoaT
from typing import Dict, Any, Tuple, List


URLS = {
    'coat_tiny': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_tiny-473c2a20.pth',
    'coat_mini': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_mini-2c6baf49.pth',
    'coat_lite_tiny': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_lite_tiny-461b07a7.pth',
    'coat_lite_mini': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_lite_mini-d7842000.pth',
    'coat_lite_small': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-coat-weights/coat_lite_small-fea1d5a1.pth'
}


class TimmCoatEncoder(nn.Module):
    def __init__(
            self,
            variant: str="coat_mini",
            pretrained: bool=True,
            **kwargs
        ) -> None:
        """
        DD

        Args:
        ---------
            variant (str, default="coat_mini"):
                One of the coat variants provided by the timm lib.
                One of: "coat_mini", "coat_tiny", "coat_lite_tiny",
                "coat_lite_mini", "coat_lite_small"
            pretrained (bool, default=True):
                If True, pretrained weights from timm are used.

        """
        super(TimmCoatEncoder, self).__init__()
        # get the url to the weights and set default config
        url = URLS[variant]
        default_cfg = self._cfg_coat(url)

        # set the constant model hyperparams
        model_cfg = dict(
            patch_size=4,
            num_heads=8,
            return_interm_layers=True,
            out_features=['x1_nocls', 'x2_nocls', 'x3_nocls', 'x4_nocls']
        )

        # get the other dynamic parameters
        kwargs = {**model_cfg, **self._get_model_params(variant)}

        # TODO, needs a dummy conv block here.

        # build the CoaT
        self.model = build_model_with_cfg(
            model_cls=CoaT,
            variant=variant,
            pretrained=pretrained,
            default_cfg=default_cfg,
            pretrained_filter_fn=self._checkpoint_filter_fn,
            **kwargs
        )
    
    @staticmethod
    def _checkpoint_filter_fn(state_dict, model):
        """
        Filter out the modules of the model to avoid errors..
        """
        out_dict = {}
        for k, v in state_dict.items():
            # original model had unused norm layers, removing them
            # requires filtering pretrained checkpoints
            if k.startswith("norm"):
                continue
            if k.startswith("aggregate"):
                continue
            if k.startswith("head"):
                continue
            out_dict[k] = v
        return out_dict

    @staticmethod
    def _cfg_coat(
            url: str='',
            input_size: Tuple[int, int]=(3, 256, 256),
            **kwargs
        ) -> Dict[str, Any]:
        """
        Get default model config
        """
        return {
            'url': url,
            'num_classes': 1000,
            'input_size': input_size,
            'pool_size': None,
            'crop_pct': .9,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': IMAGENET_DEFAULT_MEAN,
            'std': IMAGENET_DEFAULT_STD,
            'first_conv': 'patch_embed1.proj',
            'classifier': 'head',
            **kwargs
        }

    def _get_model_params(self, variant: str) -> Dict[str, Any]:
        """
        Get parameters that change between the variants
        """
        config = { 
            "coat_mini": {
                "embed_dims": [152, 216, 216, 216],
                "serial_depths": [2, 2, 2, 2],
                "parallel_depth": 6,
                "mlp_ratios": [4, 4, 4, 4],
            },
            "coat_tiny": {
                "embed_dims": [152, 152, 152, 152],
                "serial_depths": [2, 2, 2, 2],
                "parallel_depth": 6,
                "mlp_ratios": [4, 4, 4, 4],
            },
            "coat_lite_mini": {
                "embed_dims": [64, 128, 320, 512],
                "serial_depths": [2, 2, 2, 2],
                "parallel_depth": 0,
                "mlp_ratios": [8, 8, 4, 4],
            },
            "coat_lite_tiny": {
                "embed_dims": [64, 128, 256, 320],
                "serial_depths": [2, 2, 2, 2],
                "parallel_depth": 0,
                "mlp_ratios": [8, 8, 4, 4],
            },
            "coat_lite_small": {
                "embed_dims": [64, 128, 320, 512],
                "serial_depths": [3, 4, 6, 3],
                "parallel_depth": 0,
                "mlp_ratios": [8, 8, 4, 4],
            },
        }

        return config[variant]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.model(x)
        features = [x,] + list(features.values())
        return features