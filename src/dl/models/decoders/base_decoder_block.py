import torch.nn as nn
from typing import List

from .long_skips import (
    UnetSkipBlock, 
    UnetppCatSkipBlock,
    UnetppSumSkipBlock,
    Unet3pCatSkipBlock,
    Unet3pSumSkipBlock,
    UnetppCatSkipBlockLight,
    UnetppSumSkipBlockLight,
    Unet3pCatSkipBlockLight,
    Unet3pSumSkipBlockLight
)
from ..modules import FixedUnpool


class BaseDecoderBlock(nn.Module):
    def __init__(
            self,
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
            reduce_params: bool=True
        ) -> None:
        """
        Base class for all decoder blocks. Inits the upsampling and long
        skip connection that is the same for each decoder block. 

        Args:
        ---------
            in_channels (int):
                the number of channels coming in from the previous head
                or decoder branch
            out_channel_list (List[int]):
                List of the number of output channels in the decoder 
                output tensors. First index contains the number of head
                channels
            skip_channel_list (List[int]):
                List of the number of channels in each of the encoder 
                skip tensors. Ignored if long_skip is None.
            up_sampling (str):
                up sampling method to be used. One of: "interp", 
                "max_unpool", "transconv", "fixed_unpool". TODO
            long_skip (str):
                long skip connection style to be used. One of: "unet", 
                "unet++", "unet3+", None
            long_skip_merge_policy (str):
                whether long skip is summed or concatenated. One of: 
                "summation", "concatenate"
            skip_index (int, default=Nome):
                the index of the skip_channel_list. Used if 
                `long_skip`=="unet"
            out_dims (List[int]):
                List of the heights/widths of each encoder/decoder 
                feature map e.g. [256, 128, 64, 32, 16]. Assumption is 
                that feature maps are square. This is used for skip 
                blocks (unet3+, unet++)
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Perform normalization. Methods:
                Batch norm, batch channel norm, group norm, etc.
                One of ("bn", "bcn", None)
            activation (str, default="relu"):
                Activation method. One of: "relu", "swish". "mish"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            up_sampling (str, default="fixed_unpool"):
                up sampling method to be used.
                One of: "interp", "max_unpool", "transconv", 
                "fixed_unpool"
            n_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one 
                residual multiconv block        
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
            reduce_params (bool, default=False):
                If True, divides the channels from out_channel_list 
                evenly to all skip blocks similarly to unet3+ or uses 
                the number of out_channels in the skip blocks rather 
                than the number of skip_channels
        """

        allowed_ups = ("interp", "max_unpool", "transconv", "fixed_unpool")
        assert up_sampling in allowed_ups
        assert long_skip in ("unet", "unet++", "unet3+", None)
        assert long_skip_merge_policy in ("concatenate", "summation")
        
        super(BaseDecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.long_skip = long_skip
        self.out_channels = out_channel_list[skip_index + 1]

        # set upsampling method
        if up_sampling == "fixed_unpool":
            self.upsample = FixedUnpool(scale_factor=2)
        elif up_sampling == "interp":
            raise NotImplementedError
        elif up_sampling == "transconv":
            raise NotImplementedError
        elif up_sampling == "max_unpool":
            raise NotImplementedError
        
        # Set skip long skip connection if not None
        # Little kludgy for now...
        self.skip = None
        if long_skip == "unet":
            # adjust input channel dim if "concatenate"
            if long_skip_merge_policy == "concatenate":
                self.in_channels += skip_channel_list[skip_index]

            self.skip = UnetSkipBlock(
                in_channels=self.in_channels,
                skip_channels=skip_channel_list[skip_index], 
                merge_policy=long_skip_merge_policy, 
            )
            # num channels for the final conv block
            self.conv_in_channels = self.in_channels
        
        elif long_skip == "unet3+":
            if reduce_params:
                Unet3pBlock = Unet3pCatSkipBlockLight
                if long_skip_merge_policy == "summation":
                    Unet3pBlock = Unet3pSumSkipBlockLight
                    
            else:
                Unet3pBlock = Unet3pCatSkipBlock
                if long_skip_merge_policy == "summation":
                    Unet3pBlock = Unet3pSumSkipBlock
                    

            self.skip = Unet3pBlock(
                in_channels=self.in_channels,
                out_channel_list=out_channel_list,
                skip_channel_list=skip_channel_list,
                skip_index=skip_index,
                out_dims=out_dims,
                same_padding=same_padding,
                batch_norm=batch_norm,
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate,
                n_conv_blocks=n_blocks
            )

            # num channels for the final conv block
            self.conv_in_channels = self.out_channels
        
        elif long_skip == "unet++":
            if reduce_params:
                UnetppBlock = UnetppCatSkipBlockLight
                if long_skip_merge_policy == "summation":
                    UnetppBlock = UnetppSumSkipBlockLight
            else:
                UnetppBlock = UnetppCatSkipBlock
                if long_skip_merge_policy == "summation":
                    UnetppBlock = UnetppSumSkipBlock
                

            self.skip = UnetppBlock(
                in_channels=self.in_channels,
                out_channel_list=out_channel_list[1:],
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

            # set out and in channels for the next block..  
            # This became kludgy.. TODO: unkludge
            if long_skip_merge_policy == "summation":

                self.conv_in_channels = self.in_channels
                if skip_index == 0:
                    self.conv_in_channels = skip_channel_list[skip_index]

            elif long_skip_merge_policy == "concatenate":
                self.in_channels += skip_channel_list[skip_index]*(skip_index+1)
                self.conv_in_channels = self.in_channels
            
            if reduce_params:
                self.conv_in_channels = self.in_channels
                if skip_index < len(skip_channel_list[1:]):
                    self.conv_in_channels = self.out_channels
            
