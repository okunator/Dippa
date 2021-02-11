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
from segmentation_models_pytorch.pan.decoder import PANDecoder
from src.dl.models.base_model import MultiTaskSegModel

# adapted from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/pan/model.py

class PanSmpMulti(MultiTaskSegModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: str = "imagenet",
                 encoder_dilation: bool = True,
                 decoder_channels: int = 32,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 4,
                 type_branch: bool = True,
                 aux_branch: bool = True,
                 aux_out_channels: int = 1,
                 **kwargs) -> None:

        """ 
        Implementation of _PAN (Pyramid Attention Network).
        Currently works with shape of input tensor >= [B x C x 128 x 128] for pytorch <= 1.1.0
        and with shape of input tensor >= [B x C x 256 x 256] for pytorch == 1.3.1

        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds a semantic segmentation branch for classifying cell types that
        outputs type maps (B, C, H, W). Adds also an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
            encoder_name (str, default="resnet34"):
                name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            encoder_weights (str, optional, default="imagenet"):
                one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            encoder_dilation (bool, default=True): 
                Flag to use dilation in encoder last layer.
                Doesn't work with [``*ception*``, ``vgg*``, ``densenet*``] backbones, default is True.
            decoder_channels (int, default=32):
                Number of ``Conv2D`` layer filters in decoder blocks
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
            ``torch.nn.Module``: **PAN**
        .. _PAN:
            https://arxiv.org/abs/1805.10180
        """

        super().__init__()
        self.type_branch = type_branch
        self.aux_branch = aux_branch

        # encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        if encoder_dilation:
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )

        #  inst decoder
        self.inst_decoder = PANDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=2,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling
        )

        # type decoder
        self.type_decoder = None
        self.type_seg_head = None
        if self.type_branch:
            self.type_decoder = PANDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
            )

            self.type_seg_head = SegmentationHead(
                in_channels=decoder_channels,
                out_channels=classes,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling
            )

        # aux decoder
        self.aux_decoder = None
        self.aux_seg_head = None
        if self.aux_branch:
            self.aux_decoder = PANDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
            )
            self.aux_seg_head = SegmentationHead(
                in_channels=decoder_channels,
                out_channels=aux_out_channels,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling
            )

        self.name = "pan-multi-{}".format(encoder_name)
        self.initialize()
