import torch
import torch.nn as nn
from typing import List, Tuple

from . import BasicDecoderBlock, ResidualDecoderBlock, DenseDecoderBlock


class Decoder(nn.ModuleDict):
    def __init__(self,
                 encoder_channels: List[int],
                 decoder_channels: List[int],
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 up_sampling: str="fixed_unpool",
                 short_skip: str="basic",
                 long_skip: str="unet",
                 long_skip_merge_policy: str ="summation",
                 n_layers: int=1,
                 n_blocks: int=2,
                 preactivate: bool=False,
                 model_input_size: int=256,
                 **kwargs) -> None:
        """
        Basic Decoder block. Adapted from the implementation of 
        the unet model in segmentation_models_pytorch.

        Args:
        ------------
            encoder_channels (List[int]):
                Number of channels in each encoder layer output
            decoder_channels (List[int]):
                Number of channels in each decoder layer output
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Perform normalization. Methods:
                Batch norm, batch channel norm, group norm, etc.
                One of ("bn", "bcn", None)
            activation (str, default="relu"):
                Activation method. One of ("relu", "swish". "mish")
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            n_blocks (int, default=2):
                Number of basic convolution blocks in this Decoder block
            up_sampling (str, default="fixed_unpool"):
                up sampling method to be used.
                One of ("interp", "max_unpool", "transconv", "fixed_unpool")
            short_skip (str, default=None):
                Use short skip connections inside the decoder blocks.
                One of ("resdidual", "dense", None)
            long_skip (str, default="unet"):
                long skip connection style to be used.
                One of ("unet", "unet++", "unet3+", None)
            long_skip_merge_policy (str, default: "cat):
                whether long skip is summed or concatenated
                One of ("summation", "concatenate") 
            n_layers (int, default=1):
                The number of multiconv blocks inside one decoder block
                A multiconv block is a block containing atleast two 
                (bn->relu->conv)-operations
            n_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one dense 
                multiconv block.
            preactivate (bool, default=False)
                If True, normalization and activation are applied before convolution
            model_input_size (int,default=256):
                The input image size of the model. Assumes that input images are square
                patches i.e. H == W.
        """
        super(Decoder, self).__init__()
        assert len(encoder_channels[1:]) == len(decoder_channels), "Encoder and decoder need to have same number of layers (symmetry)"
        assert short_skip in ("residual", "dense", None)
        self.decoder_type = short_skip

        # flip channels nums to start from the deepest channel
        # and remove the input channel num.
        encoder_channels = encoder_channels[1:][::-1]

        # in_channels for the first layer of decoder
        head_channels = encoder_channels[0]

        # in_channels for all decoder layers
        in_channels = [head_channels] + list(decoder_channels)

        # skip channels for every decoder layer
        # no skip connection at the last decoder layer
        skip_channels = encoder_channels[1:] + [0]

        # Height/width of the encoder/decoder output feature maps
        # assumes that the downscaling factor is 2 for every pooling op
        # in the encoder. Used for skip connection arithmetics
        depth = len(skip_channels)
        out_dims = [model_input_size // 2**i for i in range(depth)][::-1]

        # set up kwargs
        kwargs = kwargs.copy()
        kwargs.setdefault("same_padding", same_padding)
        kwargs.setdefault("batch_norm", batch_norm)
        kwargs.setdefault("activation", activation)
        kwargs.setdefault("weight_standardize", weight_standardize)
        kwargs.setdefault("n_layers", n_layers)
        kwargs.setdefault("n_blocks", n_blocks)
        kwargs.setdefault("preactivate", preactivate)
        kwargs.setdefault("up_sampling", up_sampling)
        kwargs.setdefault("long_skip", long_skip)
        kwargs.setdefault("long_skip_merge_policy", long_skip_merge_policy)
        kwargs.setdefault("out_dims", out_dims)

        # Set decoder type
        if short_skip == "dense":
            DecoderBlock = DenseDecoderBlock
        elif short_skip == "residual":
            DecoderBlock = ResidualDecoderBlock
        else:
            DecoderBlock = BasicDecoderBlock

        # Build decoder
        for i, (in_ch, _) in enumerate(zip(in_channels, skip_channels)):
            kwargs["skip_index"] = i
            decoder_block = DecoderBlock(in_ch, decoder_channels, skip_channels, **kwargs)
            self.add_module(f"decoder_block{i + 1}", decoder_block)

    def forward(self, *features: Tuple[torch.Tensor]):
        features = features[1:][::-1]
        head = features[0]
        skips = features[1:]
        
        x = head
        for i, (key, block) in enumerate(self.items()):
            kwargs = {}
            kwargs["idx"] = i
            x = block(x, skips, **kwargs)
        return x
