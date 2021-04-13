import torch
import torch.nn as nn
from typing import List

from .long_skips import (
    Unet3pSkipBlock, 
    UnetSkipBlock, 
    UnetppCatSkipBlock,
    UnetppSumSkipBlock
)
from ..modules import FixedUnpool


class BaseDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channel_list: List[int],
                 skip_channel_list: List[int],
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
                 preactivate: bool=False,
                 reduce_params: bool=True) -> None:
        """
        Base class for all decoder blocks. Inits the upsampling and long skip
        connection that is the same for each decoder block. 

        Args:
        ---------
            in_channels (int):
                the number of channels coming in from the previous head/decoder branch
            out_channel_list (List[int]):
                List of the number of output channels in the decoder output tensors 
            skip_channel_list (List[int]):
                List of the number of channels in each of the encoder skip tensors.
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
                the index of the skip_channel_list list. Used if long_skip="unet"
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
            reduce_params (bool, default=False):
                If True, divides the channels from out_channel_list evenly to all
                skip blocks similarly to unet3+ or uses the number of out_channels
                in the skip blocks rather than the number of skip_channels
        """

        assert up_sampling in ("interp", "max_unpool", "transconv", "fixed_unpool")
        assert long_skip in ("unet", "unet++", "unet3+", None)
        assert long_skip_merge_policy in ("concatenate", "summation")
        super(BaseDecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.long_skip = long_skip
        self.out_channels = out_channel_list[skip_index]

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
        # Little kludgy for now...
        self.skip = None
        if long_skip == "unet":
            # adjust input channel dim if "concatenate"
            if long_skip_merge_policy == "concatenate":
                self.in_channels += skip# num channels for the final conv block_channel_list[skip_index]

            self.skip = UnetSkipBlock(
                in_channels=self.in_channels,
                skip_channels=skip_channel_list[skip_index], 
                merge_policy=long_skip_merge_policy, 
            )
            # num channels for the final conv block
            self.conv_in_channels = self.in_channels
        
        elif long_skip == "unet3+":
            self.skip = Unet3pSkipBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                skip_channel_list=skip_channel_list[skip_index:],
                out_dims=out_dims[skip_index:],
                same_padding=same_padding,
                batch_norm=batch_norm,
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate,
                n_conv_blocks=n_blocks
            )

            # num channels for the final conv block
            self.conv_in_channels = self.in_channels
        
        elif long_skip == "unet++":
            UnetppBlock = UnetppSumSkipBlock if long_skip_merge_policy == "summation" else UnetppCatSkipBlock

            self.skip = UnetppBlock(
                in_channels=self.in_channels,
                out_channel_list=out_channel_list,
                skip_channel_list=skip_channel_list,
                skip_index=skip_index,
                merge_policy=long_skip_merge_policy,
                same_padding=same_padding,
                batch_norm=batch_norm,
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate,
                n_conv_blocks=n_blocks,
                reduce_params=reduce_params
            )

            # set out and in channels for the next block..  This became kludgy.. TODO: unkludge
            if long_skip_merge_policy == "summation":
                self.conv_in_channels = skip_channel_list[skip_index] if skip_index == 0 else self.in_channels

            elif long_skip_merge_policy == "concatenate":
                self.in_channels += skip_channel_list[skip_index]*(skip_index+1)
                self.conv_in_channels = self.in_channels
            
            if reduce_params:
                self.conv_in_channels = self.out_channels if skip_index < len(skip_channel_list[1:]) else self.in_channels
            
