import torch
import torch.nn as nn

from typing import List

from ....modules.conv.utils import conv_block_func


class SumBlock(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            skip_channels: int,
            **kwargs
        ) -> None:
        """
        Sum merge. Handles clashing channel numbers with 1x1 conv block.

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            skip_channels (List[int]):
                Number of skip channels
                
        """
        super(SumBlock, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.pool = [chl != self.in_channels for chl in self.skip_channels]

        # add channel pooling modules
        for i, needs_pooling in enumerate(self.pool):
            if needs_pooling:
                ch_pool = conv_block_func(
                    in_channels=skip_channels[i], out_channels=in_channels,
                    bias=False, kernel_size=1, padding=0, n_blocks=1,
                    name="basic", skip_type="basic"
                )
                self.add_module(f"ch_pool{i + 1}", ch_pool)

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
        if self.values():
            for i, ch_pool in enumerate(self.values()):
                if self.pool[i]:
                    skips[i] = ch_pool(skips[i])

        x = torch.stack([x, *skips], dim=0).sum(dim=0)

        return x
    
    def extra_repr(self) -> str:
        s = ('in_channels={in_channels}, skip_channels={skip_channels}'
                ', out_channels={out_channels})')
        return s.format(**self.__dict__)
