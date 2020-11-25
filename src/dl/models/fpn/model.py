import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from src.dl.models.base_model import InstSegModel, InstSegModelWithClsBranch


# adapted from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/fpn/model.py

class FpnSmp(InstSegModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_pyramid_channels: int = 256,
                 decoder_segmentation_channels: int = 128,
                 decoder_merge_policy: str = "cat",
                 decoder_dropout: float = 0.2,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[str] = None,
                 upsampling: int = 4,
                 aux_branch_name: str = None,
                 **kwargs) -> None:

        """
        FPN_ is a fully convolution neural network for image semantic segmentation
        
        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds an optional aux branch for regressing (H, W) outputs
        
        Args:
            encoder_name: name of classification model (without last dense layers) used as feature
                    extractor to build segmentation model.
            encoder_depth: number of stages used in decoder, larger depth - more features are generated.
                e.g. for depth=3 encoder will generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
                spatial resolution (H/(2^depth), W/(2^depth)]
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
            decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
            decoder_merge_policy: determines how to merge outputs inside FPN.
                One of [``add``, ``cat``]
            decoder_dropout: spatial dropout rate in range (0, 1).
            in_channels: number of input channels for model, default is 3.
            classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation (str, callable): activation function used in ``.predict(x)`` method for inference.
                One of [``sigmoid``, ``softmax2d``, callable, None]
            upsampling: optional, final upsampling factor
                (default is 4 to preserve input -> output spatial shape identity)
            
        Returns:
            ``torch.nn.Module``: **FPN**
        .. _FPN:
            http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

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

        # aux decoder
        if aux_branch_name is not None:
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
                out_channels=2,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None

 
        self.name = "fpn-{}".format(encoder_name)
        self.initialize()


class FpnSmpWithClsBranch(InstSegModelWithClsBranch):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_pyramid_channels: int = 256,
                 decoder_segmentation_channels: int = 128,
                 decoder_merge_policy: str = "cat",
                 decoder_dropout: float = 0.2,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[str] = None,
                 upsampling: int = 4,
                 aux_branch_name: str = None,
                 **kwargs) -> None:

        """
        FPN_ is a fully convolution neural network for image semantic segmentation

        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds a semantic segmentation branch for classifying cell types that
        outputs type maps (B, C, H, W). Adds also an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
            encoder_name: name of classification model (without last dense layers) used as feature
                    extractor to build segmentation model.
            encoder_depth: number of stages used in decoder, larger depth - more features are generated.
                e.g. for depth=3 encoder will generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
                spatial resolution (H/(2^depth), W/(2^depth)]
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
            decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
            decoder_merge_policy: determines how to merge outputs inside FPN.
                One of [``add``, ``cat``]
            decoder_dropout: spatial dropout rate in range (0, 1).
            in_channels: number of input channels for model, default is 3.
            classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation (str, callable): activation function used in ``.predict(x)`` method for inference.
                One of [``sigmoid``, ``softmax2d``, callable, None]
            upsampling: optional, final upsampling factor
                (default is 4 to preserve input -> output spatial shape identity)
            aux_branch_name (str): one of ("hover", "micro", None)
            
        Returns:
            ``torch.nn.Module``: **FPN**
            .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
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
        if aux_branch_name is not None:
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
                out_channels=2,
                activation=activation,
                kernel_size=1,
                upsampling=upsampling,
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None


        self.name = "fpn_cls-{}".format(encoder_name)
        self.initialize()
