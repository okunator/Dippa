import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.pspnet.decoder import PSPDecoder
from src.dl.models.base_model import InstSegModel, InstSegModelWithClsBranch

# Adapted from https: // github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/pspnet/model.py

class PSPNetSmp(InstSegModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_depth: int = 3,
                 psp_out_channels: int = 512,
                 psp_use_batchnorm: bool = True,
                 psp_dropout: float = 0.2,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 8,
                 aux_branch_name: str = None,
                 **kwargs) -> None:

        """
        PSPNet_ is a fully convolution neural network for image semantic segmentation
        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
            encoder_name: name of classification model used as feature
                    extractor to build segmentation model.
            encoder_depth: number of stages used in decoder, larger depth - more features are generated.
                e.g. for depth=3 encoder will generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
                spatial resolution (H/(2^depth), W/(2^depth)]
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            psp_out_channels: number of filters in PSP block.
            psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
                One of [True, False, 'inplace']
            psp_dropout: spatial dropout rate between 0 and 1.
            in_channels: number of input channels for model, default is 3.
            classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation: activation function used in ``.predict(x)`` method for inference.
                One of [``sigmoid``, ``softmax``, callable, None]
            upsampling: optional, final upsampling factor
                (default is 8 for depth=3 to preserve input -> output spatial shape identity)
            
        Returns:
            ``torch.nn.Module``: **PSPNet**
        .. _PSPNet:
            https://arxiv.org/pdf/1612.01105.pdf
        """
        super().__init__()
        self.aux_branch_name = aux_branch_name

        # encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # inst decoder
        self.inst_decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=2,
            kernel_size=1,
            activation=activation,
            upsampling=upsampling,
        )


        # aux decoder
        if aux_branch_name is not None:
            self.aux_decoder = PSPDecoder(
                encoder_channels=self.encoder.out_channels,
                use_batchnorm=psp_use_batchnorm,
                out_channels=psp_out_channels,
                dropout=psp_dropout,
            )
            self.aux_seg_head = SegmentationHead(
                in_channels=psp_out_channels,
                out_channels=2,
                kernel_size=1,
                activation=activation,
                upsampling=upsampling,
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None

        self.name = "psp-{}".format(encoder_name)
        self.initialize()


class PSPNetSmpWithClsBranch(InstSegModelWithClsBranch):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 encoder_depth: int = 3,
                 psp_out_channels: int = 512,
                 psp_use_batchnorm: bool = True,
                 psp_dropout: float = 0.2,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 8,
                 aux_branch_name: str = None,
                 **kwargs) -> None:

        """
        PSPNet_ is a fully convolution neural network for image semantic segmentation
        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds a semantic segmentation branch for classifying cell types that
        outputs type maps (B, C, H, W). Adds also an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
            encoder_name: name of classification model used as feature
                    extractor to build segmentation model.
            encoder_depth: number of stages used in decoder, larger depth - more features are generated.
                e.g. for depth=3 encoder will generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
                spatial resolution (H/(2^depth), W/(2^depth)]
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            psp_out_channels: number of filters in PSP block.
            psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
                One of [True, False, 'inplace']
            psp_dropout: spatial dropout rate between 0 and 1.
            in_channels: number of input channels for model, default is 3.
            classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation: activation function used in ``.predict(x)`` method for inference.
                One of [``sigmoid``, ``softmax``, callable, None]
            upsampling: optional, final upsampling factor
                (default is 8 for depth=3 to preserve input -> output spatial shape identity)

        Returns:
            ``torch.nn.Module``: **PSPNet**
        .. _PSPNet:
            https://arxiv.org/pdf/1612.01105.pdf
        """
        super().__init__()
        self.aux_branch_name = aux_branch_name

        # encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # inst decoder
        self.inst_decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=2,
            kernel_size=1,
            activation=activation,
            upsampling=upsampling,
        )

        # type decoder
        self.inst_decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=1,
            activation=activation,
            upsampling=upsampling,
        )


        # aux decoder
        if aux_branch_name is not None:
            self.aux_decoder = PSPDecoder(
                encoder_channels=self.encoder.out_channels,
                use_batchnorm=psp_use_batchnorm,
                out_channels=psp_out_channels,
                dropout=psp_dropout,
            )
            self.aux_seg_head = SegmentationHead(
                in_channels=psp_out_channels,
                out_channels=2,
                kernel_size=1,
                activation=activation,
                upsampling=upsampling,
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None

        self.name = "psp-cls-{}".format(encoder_name)
        self.initialize()
