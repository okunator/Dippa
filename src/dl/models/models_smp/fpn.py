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

from typing import Optional
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.fpn.decoder import FPNDecoder

from ..base_model import MultiTaskSegModel


class FpnSmpMulti(MultiTaskSegModel):
    def __init__(
            self,
            encoder_name: str="resnet34",
            encoder_depth: int=5,
            encoder_weights: Optional[str]="imagenet",
            decoder_pyramid_channels: int=256,
            decoder_segmentation_channels: int=128,
            decoder_merge_policy: str="cat",
            decoder_dropout: float=0.2,
            in_channels: int=3,
            classes: int=2,
            activation: Optional[str]=None,
            upsampling: int=4,
            type_branch: bool=True,
            aux_branch: bool=True,
            aux_out_channels: int=1,
            **kwargs
        ) -> None:

        """
        FPN_ is a fully convolution neural network for image semantic segmentation

        This class uses 
            https://github.com/qubvel/segmentation_models.pytorch/ 
        implementation of the model and adds a semantic segmentation 
        branch for classifying cell types that outputs type maps. 
        Adds also an optional auxiliary branch for regressing auxiliary
        outputs

        Args:
        ------------
            encoder_name (str, default="resnet34"):
                name of classification model (without last dense layers)
                used as feature extractor to build segmentation model.
            encoder_depth (int, default=5): 
                number of stages used in decoder, larger depth - more 
                features are generated. e.g. for depth=3 encoder will 
                generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in 
                general the deepest feature will have spatial resolution
                (H/(2^depth), W/(2^depth)]
            encoder_weights (str, optional, default="imagenet"):
                one of ``None`` (random initialization), ``imagenet`` 
                (pre-training on ImageNet).
            decoder_pyramid_channels (int, default=256): 
                a number of conv filters in Feature Pyramid of FPN_.
            decoder_segmentation_channels (int, default=128): 
                a number of conv filters in segmentation head of FPN_.
            decoder_merge_policy (str, default="cat): 
                determines how to merge outputs inside FPN. 
                One of: ``add``, ``cat``
            decoder_dropout (float, default=0.2): 
                spatial dropout rate in range (0, 1).
            in_channels (int, default=3): 
                number of input channels for model, default is 3.
            classes (int, default=2): 
                a number of classes for output. 
                output shape - ``(batch, classes, h, w)``.
            activation (str, callable, optional, default=None): 
                activation function used in ``.predict(x)`` method for 
                inference. One of: ``sigmoid``, ``softmax2d``, callable,
                None
            upsampling (int, default=4): 
                optional, final upsampling factor default is 4 to 
                preserve input -> output spatial shape identity
            type_branch (bool, default=True)
                if True, type cls branch is added to the decoder
            aux_branch (bool, default=True)
                if True, auxiliary branch is added to the decoder
            aux_out_channels (int, default=1):
                number of output channels from the auxiliary branch
            
        Returns:
        -----------
            ``torch.nn.Module``: **FPN**
            .. _FPN:
            http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
        """
        super().__init__()
        self.type_branch = type_branch
        self.aux_branch = aux_branch

        # encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # inst decoder
        self.inst_decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=self.inst_decoder.out_channels,
            out_channels=2,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        # type decoder
        self.type_decoder = None
        self.type_seg_head = None
        if self.type_branch:
            self.type_decoder = FPNDecoder(
                encoder_channels=self.encoder.out_channels,
                encoder_depth=encoder_depth,
                pyramid_channels=decoder_pyramid_channels,
                segmentation_channels=decoder_segmentation_channels,
                dropout=decoder_dropout,
                merge_policy=decoder_merge_policy,
            )

            self.type_seg_head = SegmentationHead(
                in_channels=self.inst_decoder.out_channels,
                out_channels=classes,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )

        # aux decoder
        self.aux_decoder = None
        self.aux_seg_head = None
        if self.aux_branch:
            self.aux_decoder = FPNDecoder(
                encoder_channels=self.encoder.out_channels,
                encoder_depth=encoder_depth,
                pyramid_channels=decoder_pyramid_channels,
                segmentation_channels=decoder_segmentation_channels,
                dropout=decoder_dropout,
                merge_policy=decoder_merge_policy,
            )

            self.aux_seg_head = SegmentationHead(
                in_channels=self.aux_decoder.out_channels,
                out_channels=aux_out_channels,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )

        self.name = "fpn_multi-{}".format(encoder_name)
        self.initialize()
