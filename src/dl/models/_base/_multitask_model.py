from typing import List, Dict
from collections import OrderedDict
from omegaconf import DictConfig

from ._base_model import BaseMultiTaskSegModel
from ._heads import SegHead
from ..decoders import Decoder
from ..encoders import TimmUniversalEncoder


class MultiTaskSegModel(BaseMultiTaskSegModel):
    def __init__(
            self,
            dec_branches: Dict[str, int],
            model_input_size: int=256,
            enc_name: str="resnet50",
            enc_pretrain: bool=True,
            enc_depth: int=5,
            enc_freeze: bool=False,
            dec_n_blocks: List[int]=None,
            dec_conv_types: List[str]=None,
            dec_channels: List[int]=None,
            dec_n_layers: int=1,
            dec_short_skip: str=None,
            dec_long_skip: str="unet",
            dec_long_skip_merge_policy: str="concatenate",
            dec_activation: str="relu",
            dec_preactivate: bool=False,
            dec_upsampling: str="fixed-unpool",
            dec_normalization: str="bn",
            dec_weight_standardize: bool=False,
            dec_attention: str=None,
            **kwargs
        ) -> None: 
        """
        Abstraction of a hard parameter sharing multi-task segmentation 
        model with an arbitrary set of decoder branches and an encoder
        backbone.
        
                       ------ DECODER_1 --- SEG_HEAD_1
                       |         .
        ENCODER -------|         .
                       |         .
                       ------ DECODER_N --- SEG_HEAD_N
        
        Encoders are from the timm library:
        https://github.com/rwightman/pytorch-image-models
        
        Args:
        -----------
            dec_branches (Dict[str]):
                The types of decoders that are used for this multi-task
                seg model mapped to the number of output classes from
                that branch. The keys need to be unique in the dict.
                Allowed keys: "inst", "type", "aux", "sem"
            model_input_size (int, default=256):
                The input image size of the model. Assumes that input 
                images are square patches i.e. H == W.
            enc_name (str, default="resnet50"):
                Name of the encoder. Available encoders from:
                https://github.com/qubvel/segmentation_models.pytorch
            enc_pretrain (bool, default=True):
                imagenet pretrained encoder weights.
            enc_depth (int, default=5):
                Number of encoder blocks.
            enc_freeze (bool, default=False):
                freeze the encoder for training
            dec_n_blocks (List[int], default=None):
                Number of conv blocks inside multi-conv-blocks for each 
                stage of the decoder
            dec_conv_types (List[str], default=None):
                A list of the convolution block types for each stage of
                the decoder. e.g. ["dws", "mbconv", "dws", "basic"]
                Valid conv block types are values: "mbconv", "basic",
                "bottleneck", "fusedmbconv", "dws"
            dec_channels (List[int], default=None):
                list of integers for the number of channels in each 
                decoder block. Length of the list has to be equal to 
                `n_blocks`.
            dec_n_layers (int, default=1):
                Number of multi-conv blocks inside each level of the
                decoder
            dec_short_skip (str, default=None):
                The short skip connection style of the decoder. One of: 
                "residual", "dense", None
            dec_long_skip (str, default="unet"):
                The long skip connection style. One of: unet, unet++, 
                unet3+.
            dec_long_skip_merge_policy (str, default="sum"):
                How to merge the features in long skips. One of: "sum", 
                "cat"
            dec_normalization (str, default="bn"): 
                Normalization method to be used.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            dec_activation (str, default="relu"):
                Activation method. One of: "mish", "swish", "relu",
                "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh",
                "sigmoid", "silu", "prelu", "leaky-relu", "elu",
                "hardshrink", "tanhshrink", "hardsigmoid"
            dec_preactivate (bool, default=False):
                If True, normalization and activation are applied before
                convolution
            dec_upsampling (str, default="fixed-unpool"):
                The upsampling method. One of: "interp", "transconv", 
                "fixed-unpool"
            dec_weight_standardize (bool, default=False):
                Apply weight standardization in conv layers
            dec_attention (str, default=None):
                Attention method used in the conv blocks. 
                One of: "se", None
        """
        super().__init__()
        self.model_input_size = model_input_size
        self.enc_name = enc_name
        self.enc_pretrain = enc_pretrain
        self.enc_depth = enc_depth
        self.enc_freeze = enc_freeze
        self.dec_branches = dec_branches
        self.dec_n_blocks = dec_n_blocks
        self.dec_n_layers = dec_n_layers
        self.dec_conv_types = dec_conv_types
        self.dec_channels = dec_channels
        self.dec_short_skip = dec_short_skip
        self.dec_long_skip = dec_long_skip
        self.dec_long_skip_merge = dec_long_skip_merge_policy
        self.dec_normalization = dec_normalization
        self.dec_activation = dec_activation
        self.dec_preactivate = dec_preactivate
        self.dec_upsampling = dec_upsampling
        self.dec_weight_standardize = dec_weight_standardize
        self.dec_attention = dec_attention
        
        dec_branches = OrderedDict(dec_branches)
        allowed_branches = ["inst", "type", "sem", "aux"]
        given_branches = list(dec_branches.keys())
        if not all(k in allowed_branches for k in dec_branches.keys()):
            raise ValueError(f"""
                Invalid branch type given. Got: {given_branches}.
                Allowed: {allowed_branches}"""
            )
            
        if len(dec_branches) != len(set(dec_branches.keys())):
            raise ValueError(f"""
                Branch names need to be unique. Got {given_branches}."""
            )

        # set timm encoder
        self.encoder = TimmUniversalEncoder(
            enc_name,
            depth=enc_depth,
            pretrained=enc_pretrain,
        )

        # set decoders and heads
        for dec_type, n_classes in dec_branches.items():
            decoder = Decoder(
                enc_channels=list(self.encoder.out_channels),
                dec_channels=dec_channels,
                model_input_size=model_input_size,
                conv_block_types=dec_conv_types,
                n_blocks=dec_n_blocks,
                n_layers=dec_n_layers,
                normalization=dec_normalization,
                activation=dec_activation,
                weight_standardize=dec_weight_standardize,
                upsampling=dec_upsampling,
                short_skip=dec_short_skip,
                long_skip=dec_long_skip,
                long_skip_merge_policy=dec_long_skip_merge_policy,
                preactivate=dec_preactivate,
                attention=dec_attention
            )
            self.add_module(f"{dec_type}_decoder", decoder)
            
            seg_head = SegHead(
                in_channels=decoder.out_channels,
                out_channels=n_classes,
                kernel_size=1
            )
            self.add_module(f"{dec_type}_seg_head", seg_head)

        self.name = f"multi-task-model-{enc_name}"
        
        # init decoder weights
        self.initialize()
            
        # freeze encoder if specified
        if enc_freeze:
            self.freeze_encoder()

    @classmethod
    def from_conf(cls, conf: DictConfig, **kwargs):
        """
        Build the model from configuration .yml file

        Args:
        ----------
            conf (DictConfig):
                A config .yml file specifying all the different parts of
                the model.
        """
        
        return cls(
            model_input_size=conf.model.input_size,
            enc_name=conf.model.encoder.name,
            enc_pretrain=conf.model.encoder.pretrain,
            enc_depth=conf.model.encoder.depth,
            enc_freeze=conf.model.encoder.freeze,
            dec_branches=conf.model.decoder.branches,
            dec_n_blocks=conf.model.decoder.n_blocks,
            dec_conv_types=conf.model.decoder.conv_types,
            dec_channels=conf.model.decoder.out_channels,
            dec_n_layers=conf.model.decoder.n_layers,
            dec_preactivate=conf.model.decoder.preactivate,
            dec_upsampling=conf.model.decoder.upsampling,
            dec_short_skip=conf.model.decoder.short_skip,
            dec_normalization=conf.model.decoder.normalization,
            dec_activation=conf.model.decoder.activation,
            dec_long_skip=conf.model.decoder.long_skip,
            dec_long_skip_merge_policy=conf.model.decoder.long_skip_merge,
            dec_weight_standardize=conf.model.decoder.weight_standardize,
            dec_attention=conf.model.decoder.attention,
        )






