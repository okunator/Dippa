import torch
import torch.nn as nn
from typing import List

from .long_skips import Unet3pSkipBlock, UnetSkipBlock
from ..modules import FixedUnpool


class BaseDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 skip_channels: List[int],
                 up_sampling: str,
                 long_skip: str,
                 long_skip_merge_policy: str,
                 skip_index: int=None,
                 out_dims: List[int]=None,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 n_blocks: int=2,
                 preactivate: bool=False) -> None:
        """
        Base class for all decoder blocks. Inits the upsampling and long skip
        connection that is the same for each decoder block. 

        Args:
        ---------
            in_channels (int):
                Number of input channels
            out_channels (List[int]):
                List of the number of output channels at each of the decoder blocks
            skip_channels (List[int]):
                List of the number of channels in the each of the encoder skip tensors.
                Ignored if long_skip is None.
            up_sampling (str):
                up sampling method to be used.
                One of ("interp", "max_unpool", "transconv", "fixed_unpool")
            long_skip (str):
                long skip connection style to be used.
                One of ("unet", "unet++", "unet3+", None)
            long_skip_merge_policy (str):
                whether long skip is summed or concatenated
                One of ("summation", "concatenate")
            skip_index (int, default=Nome):
                the index of the skip_channels list. Used if long_skip="unet"
            out_dims (List[int]):
                List of the heights/widths of each encoder/decoder feature map
                e.g. [256, 128, 64, 32, 16]. Assumption is that feature maps are
                square. This is used for skip blocks (unet3+, unet++)
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
            up_sampling (str, default="fixed_unpool"):
                up sampling method to be used.
                One of ("interp", "max_unpool", "transconv", "fixed_unpool")
            n_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one residual
                multiconv block        
            preactivate (bool, default=False)
                If True, normalization and activation are applied before convolution
        """

        assert up_sampling in ("interp", "max_unpool", "transconv", "fixed_unpool")
        assert long_skip in ("unet", "unet++", "unet3+", None)
        assert long_skip_merge_policy in ("concatenate", "summation")
        super(BaseDecoderBlock, self).__init__()
        self.in_channels = in_channels

        # set upsampling method
        if up_sampling == "fixed_unpool":
            self.upsample = FixedUnpool(scale_factor=2)
        elif up_sampling == "interp":
            pass
        elif up_sampling == "transconv":
            pass
        elif up_sampling == "max_unpool":
            pass
        
        # Set skip long skip connection if not None
        self.skip = None
        if long_skip is not None:

            if long_skip == "unet":
                # adjust input channel dim if "concatenate"
                if long_skip_merge_policy == "concatenate":
                    self.in_channels += skip_channels[skip_index]

                self.skip = UnetSkipBlock(
                    merge_policy=long_skip_merge_policy, 
                    skip_channels=skip_channels[skip_index], 
                    in_channels=self.in_channels
                )
            elif long_skip == "unet3+":
                self.skip = Unet3pSkipBlock(
                    in_channels=in_channels,
                    out_channels=out_channels[skip_index],
                    skip_channels=skip_channels[skip_index:],
                    out_dims=out_dims[skip_index:],
                    same_padding=same_padding,
                    batch_norm=batch_norm,
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate,
                    n_conv_blocks=n_blocks
            )
            elif long_skip == "unet++":
                pass
