import torch
import torch.nn as nn
from typing import List, Tuple

from .decoder_stage import DecoderStage


class Decoder(nn.ModuleDict):
    def __init__(
            self,
            enc_channels: List[int],
            model_input_size: int,
            dec_channels: List[int]=None,
            conv_block_types: List[str]=None,
            n_blocks: List[int]=None,
            n_layers: int=1,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            upsampling: str="fixed-unpool",
            short_skip: str="basic",
            long_skip: str="unet",
            long_skip_merge_policy: str="summation",
            preactivate: bool=False,
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Basic Decoder block. Adapted from the implementation of 
        the unet model in segmentation_models_pytorch.

        Args:
        ------------
            enc_channels (List[int]):
                Number of channels in each encoder layer output
            model_input_size (int):
                The input image size of the model. Assumes that input 
                images are square patches i.e. H == W.
            dec_channels (List[int], default=None):
                Number of channels at each decoder layer output. 
                If None, the decoder channels will be set to the
                corresponding `encoder_channels`
            n_blocks (List[int], default=None):
                Number of conv blocks inside multi-conv-blocks for each 
                stage of the decoder
            n_layers (int, default=1):
                The number of multi-conv blocks inside one decoder stage
            same_padding (bool, default=True):
                if True, performs same-covolution
            normalization (str): 
                Normalization method to be used.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            conv_block_types (List[str], default=None):
                A list of the convolution block types for each stage of
                the decoder. e.g. ["dws", "mbconv", "dws", "basic"]
                Valid conv block types are values: "mbconv", "basic",
                "bottleneck", "fusedmbconv", "dws"
            activation (str):
                Activation method. One of: "mish", "swish", "relu",
                "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh",
                "sigmoid", "silu", "prelu", "leaky-relu", "elu",
                "hardshrink", "tanhshrink", "hardsigmoid"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            upsampling (str, default="fixed-unpool"):
                upsampling method to be used. One of "linear", "bicubic"
                "transconv", "fixed-unpool", "bilinear", "trilinear"
            short_skip (str, default=None):
                Use short skip connections inside the decoder blocks.
                One of ("resdidual", "dense", None)
            long_skip (str, default="unet"):
                long skip connection style to be used.
                One of ("unet", "unet++", "unet3+", None)
            long_skip_merge_policy (str, default: "cat):
                whether long skip is summed or concatenated
                One of ("summation", "concatenate") 
            n_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one dense 
                multiconv block.
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
            attention (str, default=None):
                Attention method used in the conv blocks. 
                One of: "se", None
        """
        super().__init__()
        
        # flip channels nums to start from the deepest channel
        # and remove the input channel num (RGB channels).
        enc_channels = enc_channels[1:][::-1]
        
        if dec_channels is not None:
            if len(enc_channels) != len(dec_channels):
                raise ValueError(f"""
                    The number of encoder channels need to match the number of
                    decoder channels. Got {len(dec_channels)} decoder channels
                    and {len(enc_channels)} encoder channels"""
                )
            dec_channels = [enc_channels[0]] + list(dec_channels)
        else:
            dec_channels = enc_channels + [enc_channels[-1]//2]
            
        if conv_block_types is not None:
            if len(conv_block_types) != len(enc_channels):
                raise ValueError(f"""
                    Then number of conv block types needs to match the number
                    of the number of decoder stages. Got {len(conv_block_types)}
                    although there are {len(enc_channels)} decoder stages"""
                )
        else:
            conv_block_types = ["basic"]*len(enc_channels)
            
        if n_blocks is not None:
            if len(n_blocks) != len(enc_channels):
                raise ValueError(f"""
                    Then number of n_blocks needs to match the number
                    of the number of decoder stages. Got {len(n_blocks)}
                    although there are {len(enc_channels)} decoder stages"""
                )
        else:
            n_blocks = [2]*len(enc_channels)
            
        skip_channels = enc_channels[1:]

        # scaling factor assumed to be 2 for the spatial dims in the
        depth = len(dec_channels)
        out_dims = [model_input_size // 2**i for i in range(depth)][::-1]

        # set up kwargs
        kwargs = kwargs.copy()
        kwargs.setdefault("same_padding", same_padding)
        kwargs.setdefault("normalization", normalization)
        kwargs.setdefault("activation", activation)
        kwargs.setdefault("weight_standardize", weight_standardize)
        kwargs.setdefault("n_layers", n_layers)
        kwargs.setdefault("preactivate", preactivate)
        kwargs.setdefault("upsampling", upsampling)
        kwargs.setdefault("long_skip", long_skip)
        kwargs.setdefault("long_skip_merge_policy", long_skip_merge_policy)
        kwargs.setdefault("dec_out_dims", out_dims)
        kwargs.setdefault("short_skip", short_skip)
        kwargs.setdefault("attention", attention)

        # Build decoder
        for i in range(depth - 1):
            kwargs["conv_block_type"] = conv_block_types[i]
            kwargs["n_blocks"] = n_blocks[i]
            
            decoder_block = DecoderStage(
                i, dec_channels, skip_channels, **kwargs
            )
            self.add_module(f"decoder_block{i + 1}", decoder_block)
            
        self.out_channels = decoder_block.out_channels

    def forward(self, *features: Tuple[torch.Tensor]) -> torch.Tensor:
        extra_skips = [] # unet++, unet3+
        features = features[1:][::-1]
        head = features[0]
        skips = features[1:]
        
        x = head
        for i, decoder_stage in enumerate(self.values()):
            x, extra = decoder_stage(x, ix=i, skips=skips, extra_skips=extra_skips)
            extra_skips = extra

        return x
