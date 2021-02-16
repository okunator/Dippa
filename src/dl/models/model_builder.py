import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from omegaconf import DictConfig
from typing import Optional

from .decoders import BasicDecoder
from .base_model import MultiTaskSegModel
from .heads import SegHead


class Model(MultiTaskSegModel):
    def __init__(self,
                 model_args: DictConfig,
                 n_classes: int,
                 aux_out_channels: int = None,
                 **kwargs) -> None:
        """
        Class which builds the model from the architectural desing choices 
        which are specified in experiment.yml
        
        Args:
            model_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the model architecture that is being used.
            n_classes (int):
                Number of classes in the dataset. Type decoder branch
                is set to output this number of classes
            aux_out_channels (int, default=None):
                Number of output channels from the auxiliary branch
        """
        super(Model, self).__init__()

        # module args
        self.activation: str = model_args.architecture_design.module_args.activation
        self.normalization: str = model_args.architecture_design.module_args.normalization
        self.weight_standardize: bool = model_args.architecture_design.module_args.weight_standardize
        self.weight_init: str = model_args.architecture_design.module_args.weight_init

        # encoder args
        self.in_channels: int = model_args.architecture_design.encoder_args.in_channels
        self.encoder_name: str = model_args.architecture_design.encoder_args.encoder
        self.pretrained: bool = model_args.architecture_design.encoder_args.pretrain
        self.depth: int = model_args.architecture_design.encoder_args.encoder_depth
        self.weights = "imagenet" if self.pretrained else None
        self.freeze = model_args.architecture_design.encoder_args.encoder_depth

        # Decoder args
        self.n_blocks: int = model_args.architecture_design.decoder_args.n_blocks
        self.short_skips: str = model_args.architecture_design.decoder_args.short_skips
        self.long_skips: str = model_args.architecture_design.decoder_args.long_skips
        self.merge_policy: str = model_args.architecture_design.decoder_args.merge_policy
        self.upsampling: str = model_args.architecture_design.decoder_args.upsampling

        # multi-task args
        self.type_branch: bool = model_args.decoder_branches.type
        self.aux_branch: bool = model_args.decoder_branches.aux
        self.aux_type: bool = model_args.decoder_branches.aux_type

        # set encoder
        self.encoder = smp.encoders.get_encoder(
            self.encoder_name,
            in_channels=self.in_channels,
            depth=self.depth,
            weights=self.weights
        )
        
        # set decoders
        kwargs = kwargs.copy()
        decoder_channels = kwargs["decoder_channels"] if "decoder_channels" in kwargs.keys() else [256, 128, 64, 32, 16]
        same_padding = kwargs["same_padding"] if "same_padding" in kwargs.keys() else False
        self.inst_decoder = BasicDecoder(
            encoder_channels=list(self.encoder.out_channels),
            decoder_channels=decoder_channels,
            same_padding=same_padding,
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
            in_channels=decoder_channels[-1],
            out_channels=2,
            kernel_size=1
        )

        self.type_decoder = None
        self.type_seg_head = None
        if self.type_branch:
            self.type_decoder = BasicDecoder(
                encoder_channels=list(self.encoder.out_channels),
                decoder_channels=decoder_channels,
                same_padding=same_padding,
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
        if self.aux_branch:
            self.aux_decoder = BasicDecoder(
                encoder_channels=list(self.encoder.out_channels),
                decoder_channels=decoder_channels,
                same_padding=same_padding,
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
                in_channels=decoder_channels[-1],
                out_channels=aux_out_channels,
                kernel_size=1
            )


        self.name = "custom-multi-task-model-{}".format(self.encoder_name)
        self.initialize()

    @classmethod
    def set_model(cls,
                  model_args: DictConfig,
                  n_classes: int,
                  aux_out_channels: int,
                  **kwargs) -> nn.Module:
        """
        Initializes smp or other pytorch model specifications

        Args:
            model_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the model architecture that is being used.
            n_classes (int):
                Number of classses in the training dataset
            aux_out_channels (int, default=1):
                Number of output channels from the auxiliary branch

        Returns:
            nn.Module initialized pytorch model specification
        """
        model = cls(model_args, n_classes, aux_out_channels)
        return model




