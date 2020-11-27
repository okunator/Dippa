import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from src.dl.models.base_model import InstSegModel, InstSegModelWithClsBranch


# adapted from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/unet/model.py

class UnetSmp(InstSegModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[Union[str, callable]] = None,
                 aux_branch_name: str = None, 
                 **kwargs) -> None:

        """
        Unet_ is a fully convolution neural network for image semantic segmentation

        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
            encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
                e.g. for depth=3 encoder will generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
                spatial resolution (H/(2^depth), W/(2^depth)]
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
            decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
                One of [True, False, 'inplace']
            decoder_attention_type: attention module used in decoder of the model
                One of [``None``, ``scse``]
            in_channels: number of input channels for model, default is 3.
            classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation: activation function to apply after final convolution;
                One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
            aux_branch_name (str): one of ("hover", "micro", None)

        Returns:
            ``torch.nn.Module``: **Unet**
        .. _Unet:
            https://arxiv.org/pdf/1505.04597
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
        self.inst_decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=2,
            activation=activation,
            kernel_size=1,
        )

        # aux decoder
        if aux_branch_name is not None:
            self.aux_decoder = UnetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )
            self.aux_seg_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=2,
                activation=activation,
                kernel_size=1,
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None

 
        self.name = "u-net-{}".format(encoder_name)
        self.initialize()


class UnetSmpWithClsBranch(InstSegModelWithClsBranch):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 in_channels: int = 3,
                 classes: int = 2,
                 activation: Optional[Union[str, callable]] = None,
                 aux_branch_name: str = None,
                 **kwargs) -> None:

        """
        Unet is a fully convolution neural network for image semantic segmentation
        
        This class uses https://github.com/qubvel/segmentation_models.pytorch/ implementation
        of the model and adds a semantic segmentation branch for classifying cell types that
        outputs type maps (B, C, H, W). Adds also an optional aux branch for regressing (B, 2, H, W) outputs

        Args:
            encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
            encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
                e.g. for depth=3 encoder will generate list of features with following spatial shapes
                [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
                spatial resolution (H/(2^depth), W/(2^depth)]
            encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
            decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
            decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
                One of [True, False, 'inplace']
            decoder_attention_type: attention module used in decoder of the model
                One of [``None``, ``scse``]
            in_channels: number of input channels for model, default is 3.
            classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
            activation: activation function to apply after final convolution;
                One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
            aux_branch_name (str): one of ("hover", "micro", None)

        Returns:
            ``torch.nn.Module``: **Unet**
        .. _Unet:
            https://arxiv.org/pdf/1505.04597
        """

        super().__init__()
        self.aux_branch_name = aux_branch_name

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # inst decoder
        self.inst_decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=2,
            activation=activation,
            kernel_size=1,
        )

        # type decoder
        self.type_decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.type_seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=1,
        )

        # aux decoder
        if aux_branch_name is not None:
            self.aux_decoder = UnetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )
            self.aux_seg_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=2,
                activation=activation,
                kernel_size=1,
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None

        self.name = "unet_cls-{}".format(encoder_name)
        self.initialize()
