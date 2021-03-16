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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.linknet.decoder import LinknetDecoder

from ..base_model import MultiTaskSegModel


class LinknetSmpMulti(MultiTaskSegModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 type_branch: bool = True,
                 aux_branch: bool = True,
                 aux_out_channels: int = 1,
                 **kwargs) -> None:
        """
        Linknet_ is a fully convolution neural network for fast image semantic segmentation
        Note:
            This implementation by default has 4 skip connections (original - 3).

        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
        ------------
            encoder_name (str, default="resnet34"):
                name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            encoder_depth (int, default=5): 
                number of stages used in decoder, larger depth - more features are generated.
                e.g. for depth=3 encoder will generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
                spatial resolution (H/(2^depth), W/(2^depth)]
            encoder_weights (str, optional, default="imagenet"):
                one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            decoder_use_batchnorm (bool, default=True)
                if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
                One of [True, False, 'inplace']
            in_channels (int, default=3): 
                number of input channels for model, default is 3.
            classes (int, default=2): 
                a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation (str, callable, optional, default=None): 
                activation function used in ``.predict(x)`` method for inference.
                One of [``sigmoid``, ``softmax2d``, callable, None]
            upsampling (int, default=4): 
                optional, final upsampling factor
                (default is 4 to preserve input -> output spatial shape identity)
            type_branch (bool, default=True)
                if True, type cls decoder branch is added to the network
            aux_branch (bool, default=True)
                if True, auxiliary decoder branch is added to the network
            aux_out_channels (int, default=1):
                number of output channels from the auxiliary branch

        Returns:
        -----------
            ``torch.nn.Module``: **Linknet**
        .. _Linknet:
            https://arxiv.org/pdf/1707.03718.pdf
        """
        super().__init__()
        self.type_branch = type_branch
        self.aux_branch = aux_branch

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # inst decoder
        self.inst_decoder = LinknetDecoder(
            encoder_channels=self.encoder.out_channels,
            n_blocks=encoder_depth,
            prefinal_channels=32,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=32, out_channels=2, activation=activation, kernel_size=1
        )

        # type decoder
        self.type_decoder = None
        self.type_seg_head = None
        if self.type_branch:
            self.type_decoder = LinknetDecoder(
                encoder_channels=self.encoder.out_channels,
                n_blocks=encoder_depth,
                prefinal_channels=32,
                use_batchnorm=decoder_use_batchnorm,
            )

            self.type_seg_head = SegmentationHead(
                in_channels=32, out_channels=classes, activation=activation, kernel_size=1
            )

        # aux decoder
        self.aux_decoder = None
        self.aux_seg_head = None
        if self.aux_branch:
            self.aux_decoder = LinknetDecoder(
                encoder_channels=self.encoder.out_channels,
                n_blocks=encoder_depth,
                prefinal_channels=32,
                use_batchnorm=decoder_use_batchnorm,
            )

            self.aux_seg_head = SegmentationHead(
                in_channels=32, out_channels=aux_out_channels, activation=activation, kernel_size=1
            )

        self.name = "link-multi-{}".format(encoder_name)
        self.initialize()