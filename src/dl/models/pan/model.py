
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.pan.decoder import PANDecoder
from src.dl.models.base_model import InstSegModel, InstSegModelWithClsBranch

# adapted from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/pan/model.py

class PanSmp(InstSegModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: str = "imagenet",
                 encoder_dilation: bool = True,
                 decoder_channels: int = 32,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 4,
                 aux_branch_name: str = None,
                 **kwargs) -> None:

        """ 
        Implementation of _PAN (Pyramid Attention Network).
        Currently works with shape of input tensor >= [B x C x 128 x 128] for pytorch <= 1.1.0
        and with shape of input tensor >= [B x C x 256 x 256] for pytorch == 1.3.1

        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds a semantic segmentation branch for classifying cell types that
        outputs type maps (B, C, H, W). Adds also an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
            encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            encoder_dilation: Flag to use dilation in encoder last layer.
                Doesn't work with [``*ception*``, ``vgg*``, ``densenet*``] backbones, default is True.
            decoder_channels: Number of ``Conv2D`` layer filters in decoder blocks
            in_channels: number of input channels for model, default is 3.
            classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation: activation function to apply after final convolution;
                One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
            upsampling: optional, final upsampling factor
                (default is 4 to preserve input -> output spatial shape identity)
            aux_branch_name (str): one of ("hover", "micro", None)
            
        Returns:
            ``torch.nn.Module``: **PAN**
        .. _PAN:
            https://arxiv.org/abs/1805.10180
        """

        super().__init__()

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

        # aux decoder
        if aux_branch_name is not None:
            self.aux_decoder = PANDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
            )
            self.aux_seg_head = SegmentationHead(
                in_channels=decoder_channels,
                out_channels=2,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None

        self.name = "pan-{}".format(encoder_name)
        self.initialize()


class PanSmpWithClsBranch(InstSegModelWithClsBranch):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: str = "imagenet",
                 encoder_dilation: bool = True,
                 decoder_channels: int = 32,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[Union[str, callable]] = None,
                 upsampling: int = 4,
                 aux_branch_name: str = None,
                 **kwargs) -> None:

        """ 
        Implementation of _PAN (Pyramid Attention Network).
        Currently works with shape of input tensor >= [B x C x 128 x 128] for pytorch <= 1.1.0
        and with shape of input tensor >= [B x C x 256 x 256] for pytorch == 1.3.1

        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds a semantic segmentation branch for classifying cell types that
        outputs type maps (B, C, H, W). Adds also an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
            encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            encoder_dilation: Flag to use dilation in encoder last layer.
                Doesn't work with [``*ception*``, ``vgg*``, ``densenet*``] backbones, default is True.
            decoder_channels: Number of ``Conv2D`` layer filters in decoder blocks
            in_channels: number of input channels for model, default is 3.
            classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation: activation function to apply after final convolution;
                One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
            upsampling: optional, final upsampling factor
                (default is 4 to preserve input -> output spatial shape identity)
            aux_branch_name (str): one of ("hover", "micro", None)
            
        Returns:

            ``torch.nn.Module``: **PAN**
        .. _PAN:
            https://arxiv.org/abs/1805.10180
        """

        super().__init__()

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
        if aux_branch_name is not None:
            self.aux_decoder = PANDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
            )
            self.aux_seg_head = SegmentationHead(
                in_channels=decoder_channels,
                out_channels=2,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None

        self.name = "pan-cls-{}".format(encoder_name)
        self.initialize()