import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from src.dl.models.base_model import InstSegModel, InstSegModelWithClsBranch
from src.dl.models.unet3plus.decoder import Unet3pDecoder


class Unet3pInst(InstSegModel):
    def __init__(self,
                 depth: int = 5,
                 cat_channels: int = 64,
                 encoder_name: str = "resnext50_32x4d",
                 encoder_weights: str = "imagenet",
                 aux_branch_name: str = None,
                 activation: Optional[Union[str, callable]] = None,
                 **kwargs) -> None:
        """
        UNET3+
        https://arxiv.org/pdf/2004.08790.pdf
        
        Class for Unet3+ for segmentation with optional aux branch for regressing outputs
        Adjusted to use pre-trained encoders from the pytorch_segmentation_models library

        Args:
            depth (int): depth of the encoder
            cat_channels (int): number of output channels after every convolutin in decoder
            encoder_name (str): name of the encoder
            encoder_weights (str): encoder weights
            aux_branch (str): one of (None, "hover", "micro")
        """
        assert aux_branch_name in ("hover", "micro", None)
        super().__init__()
        self.aux_branch_name = aux_branch_name
        self.encoder = get_encoder(encoder_name, depth=depth, weights=encoder_weights)

        # inst decoder
        self.inst_decoder = Unet3pDecoder(
            self.encoder.out_channels,
            n_blocks=depth,
            cat_channels=cat_channels
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=cat_channels*(depth+1),
            out_channels=2,
            activation=None,
            kernel_size=1,
        )

        # aux decoder
        if aux_branch_name is not None:
            self.aux_decoder = Unet3pDecoder(
                self.encoder.out_channels,
                n_blocks=depth,
                cat_channels=cat_channels
            )

            self.aux_seg_head = SegmentationHead(
                in_channels=cat_channels*(depth+1),
                out_channels=2,
                activation=None,
                kernel_size=1,
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None

        # init weights
        self.initialize()
        self.name = "unet3+-{}".format(encoder_name)


class Unet3pWithClsBranch(InstSegModelWithClsBranch):
    def __init__(self,
                 classes: int = 2,
                 depth: int = 5,
                 cat_channels: int = 64,
                 encoder_name: str = "resnext50_32x4d",
                 encoder_weights: str = "imagenet",
                 aux_branch_name: str = None,
                 activation: Optional[Union[str, callable]] = None,
                 **kwargs) -> None:
        """
        UNET3+
        https://arxiv.org/pdf/2004.08790.pdf
        
        Class for Unet3+ for segmentation and type classification
        with optional aux branch for regressing outputs
        Adjusted to use pre-trained encoders from the pytorch_segmentation_models library

        Args:
            classes (int): number of classes
            depth (int): depth of the encoder
            cat_channels (int): number of output channels after every convolutin in decoder
            encoder_name (str): name of the encoder
            encoder_weights (str): encoder weights
            aux_branch (str): one of (None, "hover", "micro")
        """
        assert aux_branch_name in ("hover", "micro", None)
        super().__init__()
        self.aux_branch_name = aux_branch_name
        self.encoder = get_encoder(encoder_name, depth=depth, weights=encoder_weights)

        # inst decoder
        self.inst_decoder = Unet3pDecoder(
            self.encoder.out_channels,
            n_blocks=depth,
            cat_channels=cat_channels
        )

        self.inst_seg_head = SegmentationHead(
            in_channels=cat_channels*(depth+1),
            out_channels=2,
            activation=activation,
            kernel_size=1,
        )

        # type decoder
        self.type_decoder = Unet3pDecoder(
            self.encoder.out_channels,
            n_blocks=depth,
            cat_channels=cat_channels
        )

        self.type_seg_head = SegmentationHead(
            in_channels=cat_channels*(depth+1),
            out_channels=classes,
            activation=activation,
            kernel_size=1,
        )

        # aux decoder
        if aux_branch_name is not None:
            self.aux_decoder = Unet3pDecoder(
                self.encoder.out_channels,
                n_blocks=depth,
                cat_channels=cat_channels
            )

            self.aux_seg_head = SegmentationHead(
                in_channels=cat_channels*(depth+1),
                out_channels=2,
                activation=activation,
                kernel_size=1,
            )
        else:
            self.aux_decoder = None
            self.aux_seg_head = None
        
        # init weights
        self.initialize()
        self.name = "unet3+-{}".format(encoder_name)
