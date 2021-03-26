import torch
import torch.nn as nn

import src.dl.models.layers as layers


class BaseDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 up_sampling: str,
                 long_skip: str,
                 long_skip_merge_policy: str,
                 preactivate: bool) -> None:
        """
        Base class for all decoder blocks. Inits the upsampling and long skip
        connection that is the same for each decoder block.

        Args:
        ---------
            in_channels (int):
                Number of input channels
            skip_channels (int):
                Number of channels in the encoder skip tensor.
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
            preactivate (bool)
                If True, normalization and activation are applied before convolution
        """

        assert up_sampling in ("interp", "max_unpool", "transconv", "fixed_unpool")
        assert long_skip in ("unet", "unet++", "unet3+", None)
        assert long_skip_merge_policy in ("concatenate", "summation")
        super(BaseDecoderBlock, self).__init__()

        self.up_sampling = up_sampling
        self.long_skip = long_skip
        self.merge_pol = long_skip_merge_policy
        self.preactivate = preactivate
        self.in_channels = in_channels
        self.skip_channels = skip_channels

        # set upsampling method
        if up_sampling == "fixed_unpool":
            self.upsample = layers.FixedUnpool(scale_factor=2)
        elif up_sampling == "interp":
            pass
        elif up_sampling == "transconv":
            pass
        elif up_sampling == "max_unpool":
            pass
        
        # Set skip long skip connection if not None
        self.skip = None
        if self.long_skip is not None:

            # adjust input channel dim if "concatenate"
            if self.merge_pol == "concatenate":
                self.in_channels += self.skip_channels

            if self.long_skip == "unet":
                self.skip = layers.UnetSkipBlock(
                    merge_policy=self.merge_pol, 
                    skip_channels=self.skip_channels, 
                    in_channels=self.in_channels
                )
            elif self.long_skip == "unet++":
                pass
            elif self.long_skip == "unet3+":
                pass
