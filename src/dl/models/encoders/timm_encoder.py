"""
The MIT License

Copyright (c) 2019, Pavel Yakubovskiy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
# Ported from: 
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/encoders/timm_universal.py


import timm
import torch.nn as nn


class TimmUniversalEncoder(nn.Module):
    def __init__(
            self,
            name: str,
            pretrained: bool=True,
            in_channels: int=3,
            depth: int=5,
            **kwargs
        ) -> None:
        """
        Import any encoder from timm package. Excluding transformers..
        """
        super().__init__()

        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [in_channels, ] + self.model.feature_info.channels()
        self._depth = depth

    def forward(self, x):
        features = self.model(x)
        features = [x,] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels