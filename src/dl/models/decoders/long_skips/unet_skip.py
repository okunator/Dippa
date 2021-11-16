import torch
import torch.nn as nn
from typing import Tuple, List

from .merge_blocks.utils import merge_func


class UnetSkip(nn.ModuleDict):
    def __init__(
            self,
            stage_ix: int,
            in_channels: int=None,
            skip_channels: List[int]=None,
            merge_policy: str="summation",
            **kwargs
        ) -> None:
        """
        Simple U-net like skip connection block.

        U-Net: Convolutional Networks for Biomedical Image Segmentation
            - https://arxiv.org/abs/1505.04597#

        Supports different summation and concatenation merging policies.

        Args:
        ----------
            stage_ix (int):
                Index number signalling the current decoder stage
            in_channels (int, default=None):
                The number of channels in the tensor generated in the
                previous decoder block that gets upsampled and merged
                with the encoder generated tensor.
            skip_channels (List[int]):
                List of the number of channels in the encoder
                stages. Order is bottom up. This list does not include
                the final bottleneck stage out channels since it is 
                included in `dec_channels`. e.g. [1024, 512, 256, 64] 
            merge_policy (str, default="summation"):
                Sum or concatenate the features together.
                One of ("summation", "concatenate")
        """
        super(UnetSkip, self).__init__()
        self.merge_policy = merge_policy
        self.in_channels = in_channels
        self.stage_ix = stage_ix

        self.skip_out_chl = 0
        self.merge = nn.Identity()
        if stage_ix < len(skip_channels):
            self.skip_out_chl = skip_channels[stage_ix]

            self.merge = merge_func(
                self.merge_policy,
                out_channels=self.out_channels,
                in_channels=self.in_channels,
                skip_channels=[self.skip_out_chl]
            )
        
    @property
    def out_channels(self) -> int:
        out_channels = self.in_channels
        if self.merge_policy == "concatenate":
            out_channels += self.skip_out_chl

        return out_channels

    def forward(
            self,
            x: torch.Tensor,
            skips: Tuple[torch.Tensor],
            ix: int,
            **kwargs
        ) -> Tuple[torch.Tensor, None]:
        """
        Args:
        ------------
            x (torch.Tensor):
                input from the previous decoder layer
            skips (Tuple[torch.Tensor]):
                all the features from the encoder
            ix (int):
                index for the for the feature from the encoder

        Returns:
        ------------
            Tuple[torch.Tensor]: The skip connection tensor and None.
                None is returned for convenience to avoid clashes with
                the other parts of the repo
        """
        if ix < len(skips):
            skip = skips[ix]
            x = self.merge(x, [skip])

        return x, None