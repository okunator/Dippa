import torch
import torch.nn as nn
from typing import Tuple, List


class UnetSkipBlock(nn.ModuleDict):
    def __init__(
            self,
            dec_stage_ix: int,
            in_channels: int=None,
            enc_out_channels: List[int]=None,
            merge_policy: str="summation"
        ) -> None:
        """
        Simple U-net like skip connection block

        Args:
        ----------
            dec_stage_ix (int):
                Index number signalling the current decoder stage
            in_channels (int, default=None):
                The number of channels in the tensor generated in the
                previous decoder block that gets upsampled and merged
                with the encoder generated tensor. If merge policy is 
                "sum". The skip feature channel dim needs to be pooled
                with 1x1 conv to match input size.
            enc_out_channels (List[int]):
                List of the number of channels in each of the encoder
                stages. Order is bottom up. This list does not include
                the final bottleneck stage out channels since it is 
                included in `dec_out_channels`. Also, the last element 
                of the list is zero to avoid a skip at the final decoder
                stage. e.g. [1024, 512, 256, 64, 0] 
            merge_policy (str, default="summation"):
                Sum or concatenate the features together.
                One of ("summation", "concatenate")
        """
        assert merge_policy in ("concatenate", "summation")
        
        super(UnetSkipBlock, self).__init__()
        self.merge_policy = merge_policy
        self._in_channels = in_channels
        skip_channels = enc_out_channels[dec_stage_ix]

        # adjust input channel dim if "concatenate"
        if merge_policy == "concatenate":
            self._in_channels += skip_channels

        # channel pooling for skip features if "summation"
        if self.merge_policy == "summation" and skip_channels > 0:
            self.add_module(
                "ch_pool", nn.Conv2d(
                    skip_channels, in_channels, 
                    kernel_size=1, padding=0, bias=False
                )
            )

    @property
    def out_channels(self) -> int:
        return self._in_channels

    def _merge(
            self,
            features: List[torch.Tensor],
        ) -> torch.Tensor:
        """
        Merges all the feature maps of this module together

        Args:
        ---------
            features (List[torch.Tensor]):

        Returns:
        ---------
            torch.Tensor: The merged tensor
        """
        assert self.merge_policy in ("concatenate", "summation")

        if self.merge_policy == "concatenate":
            x = torch.cat(features, dim=1)
        else:
            # do channel pooling if num of skip channels does not match 
            # with the num of in channels  
            if features[1].shape[1] != features[0].shape[1]:
                features[1] = self.ch_pool(features[1])

            x = torch.stack(features, dim=0).sum(dim=0)

        return x

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
            x = self._merge([x, skip])

        return x, None