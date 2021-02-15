import torch
import torch.nn as nn


class SegHead(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1) -> None:
        """
        Reduce the number of channels to out_channels

        Args:
            in_channels (int):
                Number of channels in the input
            out_channels (int):
                Number of channels in the output
            kernel_size (int, default=1):

        """
        super(SegHead, self).__init__()

        if kernel_size != 1:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        return x