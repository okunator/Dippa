import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from omegaconf import DictConfig
from typing import Optional, List

from .decoders import BasicDecoder
from .base_model import MultiTaskSegModel
from .heads import SegHead


class Model(MultiTaskSegModel):
    def __init__(self,
                 encoder_name: str,
                 in_channels: int,
                 pretrain: bool,
                 encoder_depth: int,
                 freeze_encoder: bool,
                 activation: str,
                 normalization: str,
                 weight_standardize: bool,
                 weight_init: str,
                 decoder_n_blocks: int,
                 decoder_short_skips: str,
                 long_skips: str,
                 long_skip_merge_policy: str,
                 upsampling: str,
                 type_branch: bool,
                 n_classes: int,
                 aux_branch: str,
                 decoder_channels: List[int]=None) -> None:
        """
        Class which builds the model from the architectural desing choices 
        Encoders are from the segmentation_models_pytorch library.
        
        Args:
        -----------
            encoder_name (str):
                Name of the encoder. Available encoders from:
                https://github.com/qubvel/segmentation_models.pytorch
            in_channels (int):
                Number of input channels in the encoder.
            pretrain (bool):
                imagenet pretrained encoder weights
            encoder_depth (int):
                Number of encoder blocks. 
            freeze_encoder (bool, default=False):
                freeze the encoder for training
            activation (str):
                Activation method. One of ("mish", "swish", "relu")
            normalization (str):
                Normalization method. One of ("bn", "bcn" "nope") where "nope" = None
            weight_standardize (bool):
                Apply weight standardization in conv layers
            weight_init (str):
                weight initialization method One of ("he", "eoc", "fixup")
            decoder_n_blocks (int):
                Number of conv blocks in each layer of the decoder.
            decoder_short_skips (str):
                The short skip connection style of the decoder. One of 
                ("residual", "dense", "nope") where "nope" = None.
            long_skips (str):
                The long skip connection style. One of (unet, unet++, unet3+).
            long_skip_merge_policy (str):
                How to merge the features in long skips. One of ("sum", "cat")
            upsampling (str):
                The upsampling method. One of ("interp", "max_unpool", transconv", "fixed_unpool")
            type_branch (bool):
                Flag whether to include a type semantic segmentation branch to the network.
            n_classes (int):
                Number of classes in the dataset. Type decoder branch is set to output this 
                number of classes. If type_branch = False, this is ignored.
            aux_branch (str):
                The auxiliary branch type. One of ("hover", "dist", "contour", None). If None, no
                auxiliary branch is included in the network.
            decoder_channels (List[int], default=None):
                list of integers for the number of channels in each decoder block.
                Length of the list has to be equal to n_blocks.
        """
        super(Model, self).__init__()

        # encoder args
        self.in_channels = in_channels
        self.encoder_name = encoder_name
        self.pretrained = pretrain
        self.depth = encoder_depth
        self.weights = "imagenet" if self.pretrained else None
        self.freeze = freeze_encoder

        # module args
        self.activation = activation
        self.normalization = normalization
        self.weight_standardize = weight_standardize
        self.weight_init = weight_init

        # Decoder args
        self.n_blocks = decoder_n_blocks
        self.short_skips = decoder_short_skips
        self.long_skips = long_skips
        self.merge_policy = long_skip_merge_policy
        self.upsampling = upsampling

        # multi-task args
        self.type_branch = type_branch
        self.aux_branch = aux_branch
        self.decoder_channels = [256, 128, 64, 32, 16] if decoder_channels is not None else decoder_channels
        assert len(self.decoder_channels) == self.depth, (
            f"decoder n_blocks: {self.depth} != len(decoder_channels): {len(self.decoder_channels)}" 
        )

        # set encoder
        self.encoder = smp.encoders.get_encoder(
            self.encoder_name,
            in_channels=self.in_channels,
            depth=self.depth,
            weights=self.weights
        )
        
        self.inst_decoder = BasicDecoder(
            encoder_channels=list(self.encoder.out_channels),
            decoder_channels=self.decoder_channels,
            same_padding=True,
            batch_norm=self.normalization,
            activation=self.activation,
            weight_standardize=self.weight_standardize,
            n_blocks=self.n_blocks,
            up_sampling=self.upsampling,
            short_skip=self.short_skips,
            long_skip=self.long_skips,
            long_skip_merge_policy=self.merge_policy
        )

        self.inst_seg_head = SegHead(
            in_channels=self.decoder_channels[-1],
            out_channels=2,
            kernel_size=1
        )

        self.type_decoder = None
        self.type_seg_head = None
        if self.type_branch:
            self.type_decoder = BasicDecoder(
                encoder_channels=list(self.encoder.out_channels),
                decoder_channels=self.decoder_channels,
                same_padding=True,
                batch_norm=self.normalization,
                activation=self.activation,
                weight_standardize=self.weight_standardize,
                n_blocks=self.n_blocks,
                up_sampling=self.upsampling,
                short_skip=self.short_skips,
                long_skip=self.long_skips,
                long_skip_merge_policy=self.merge_policy
            )

            self.type_seg_head = SegHead(
                in_channels=decoder_channels[-1],
                out_channels=n_classes,
                kernel_size=1
            )

        self.aux_decoder = None
        self.aux_seg_head = None
        if self.aux_branch is not None:
            aux_out_channels = 2 if self.aux_branch == "hover" else 1
            self.aux_decoder = BasicDecoder(
                encoder_channels=list(self.encoder.out_channels),
                decoder_channels=self.decoder_channels,
                same_padding=True,
                batch_norm=self.normalization,
                activation=self.activation,
                weight_standardize=self.weight_standardize,
                n_blocks=self.n_blocks,
                up_sampling=self.upsampling,
                short_skip=self.short_skips,
                long_skip=self.long_skips,
                long_skip_merge_policy=self.merge_policy
            )

            self.aux_seg_head = SegHead(
                in_channels=self.decoder_channels[-1],
                out_channels=aux_out_channels,
                kernel_size=1
            )


        self.name = "custom-multi-task-model-{}".format(self.encoder_name)
        
        # init decoder weights
        self.initialize()

        # freeze encoder if specified
        if self.freeze:
            self.freeze_encoder()



