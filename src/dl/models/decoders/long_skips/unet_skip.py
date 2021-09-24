import torch
import torch.nn as nn
from typing import Tuple


class UnetSkipBlock(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int=None,
            skip_channels: int=None,
            merge_policy: str="summation"
        ) -> None:
        """
        Simple U-net like skip connection block

        Args:
        ----------
            in_channels (int, default=None):
                The number of channels in the tensor generated in the
                previous decoder block that gets upsampled and merged
                with the encoder generated tensor. If merge policy is 
                "sum". The skip feature channel dim  needs to be pooled
                with 1x1 conv to match input size.
            skip_channels (int, default=None)
                The number of channels in the tensor generated in the
                encoder. If merge policy is "sum". The skip feature 
                channel dim  needs to be pooled with 1x1 conv to match 
                input size.
            merge_policy (str, default="summation"):
                Sum or concatenate the features together.
                One of ("summation", "concatenate")
        """
        super(UnetSkipBlock, self).__init__()
        assert merge_policy in ("concatenate", "summation")
        self.merge_policy = merge_policy

        # channel pooling for skip features if "summation"
        if self.merge_policy == "summation" and skip_channels > 0:
            self.add_module(
                "ch_pool", nn.Conv2d(
                    skip_channels, in_channels, 
                    kernel_size=1, padding=0, bias=False
                )
            )

    def forward(
            self,
            x: torch.Tensor,
            skips: Tuple[torch.Tensor],
            idx: int,
            **kwargs
        ) -> torch.Tensor:
        """
        Args:
        ------------
            x (torch.Tensor):
                input from the previous decoder layer
            skips (Tuple[torch.Tensor]):
                all the features from the encoder
            idx (int):
                index for the for the feature from the encoder
        """
        if idx < len(skips):
            if self.merge_policy == "concatenate":
                skip = skips[idx]
                x = torch.cat([x, skip], dim=1)
            elif self.merge_policy == "summation":
                skip = skips[idx]
                # if num skip channels != num input channels
                if skip.shape[1] != x.shape[1]:
                    skip = self.ch_pool(skips[idx])
                x += skip

        return x