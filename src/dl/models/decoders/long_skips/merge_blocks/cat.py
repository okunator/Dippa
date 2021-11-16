import torch
import torch.nn as nn

from typing import List


class CatBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            skip_channels: int,
            **kwargs
        ) -> None:
        """
        Concatenate merge

        Args:
        ---------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            skip_channels (List[int]):
                Number of skip channels
        """
        super(CatBlock, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.skip_channels = skip_channels

    def forward(
            self,
            x: torch.Tensor,
            skips: List[torch.Tensor],
        ) -> torch.Tensor:
        """
        Args:
        ------------
            x (torch.Tensor):
                input tensor
            skips (List[torch.Tensor]):
                all the skip features in a list

        Returns:
        ------------
            torch.Tensor: The summed out tensor. Shape (B, in_channels, H, W)
        """

        return torch.cat([x, *skips], dim=1)

    def __repr__(self):
        return f"CatBlock(in_channels={self.in_channels}, skip_channels={self.skip_channels} out_channels={self.out_channels})"