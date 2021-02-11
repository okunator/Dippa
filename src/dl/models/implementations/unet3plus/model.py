import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from src.dl.models.base_model import MultiTaskSegModel
from src.dl.models.unet3plus.decoder import Unet3pDecoder


class Unet3pMulti(MultiTaskSegModel):
    def __init__(self,
                 depth: int = 5,
                 cat_channels: int = 64,
                 classes: int = 2,
                 encoder_name: str = "resnext50_32x4d",
                 encoder_weights: str = "imagenet",
                 activation: Optional[Union[str, callable]] = None,
                 type_branch: bool = True,
                 aux_branch: bool = True,
                 aux_out_channels: int = 1,
                 **kwargs) -> None:
        """
        UNET3+
        https://arxiv.org/pdf/2004.08790.pdf

        This implementation deviates from the original one such that the decoder path
        is dense and not as memory expensive as in the original 
        
        Class for Unet3+ for segmentation with optional aux branch for regressing outputs
        Adjusted to use pre-trained encoders from the pytorch_segmentation_models library

        Args:
            depth (int, default=5): 
                depth of the encoder
            cat_channels (int, default=64): 
                number of output channels after every convolutin in decoder
            classes (int, default=2): 
                a number of classes for output (output shape - ``(batch, classes, h, w)``).
            encoder_name (str, default=resnext50_32x4d): 
                name of the encoder
            encoder_weights (str, default="imagenet"): 
                encoder weights
            type_branch (bool, default=True)
                if True, type cls decoder branch is added to the network
            aux_branch (bool, default=True)
                if True, auxiliary decoder branch is added to the network
            aux_out_channels (int, default=1):
                number of output channels from the auxiliary branch

        Returns:
            ``torch.nn.Module``: **Unet3pMulti**
        """
        super().__init__()
        self.type_branch = type_branch
        self.aux_branch = aux_branch

        # encoder
        self.encoder = get_encoder(
            encoder_name, 
            depth=depth, 
            weights=encoder_weights
        )

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

        # type decoder
        self.type_decoder = None
        self.type_seg_head = None
        if self.type_branch:
            self.type_decoder = Unet3pDecoder(
                self.encoder.out_channels,
                n_blocks=depth,
                cat_channels=cat_channels
            )

            self.type_seg_head = SegmentationHead(
                in_channels=cat_channels*(depth+1),
                out_channels=classes,
                activation=None,
                kernel_size=1,
            )

        # aux decoder
        self.aux_decoder = None
        self.aux_seg_head = None
        if self.aux_branch:
            self.aux_decoder = Unet3pDecoder(
                self.encoder.out_channels,
                n_blocks=depth,
                cat_channels=cat_channels
            )

            self.aux_seg_head = SegmentationHead(
                in_channels=cat_channels*(depth+1),
                out_channels=aux_out_channels,
                activation=None,
                kernel_size=1,
            )

        # init weights
        self.initialize()
        self.name = "unet3+-multi-{}".format(encoder_name)