import torch
import torch.nn as nn
from typing import List, Tuple

from ...modules import FixedUnpool
from .. import MultiBlockBasic


class MergeBlock(nn.ModuleDict):
    def __init__(self, 
                 prev_channels: int=None,
                 current_channels: int=None,
                 merge_policy: str="summation") -> None:
        """
        Merge block for all the skip connections in unet++ 

        Args:
        ----------
            prev_channels (int, default=None)
                The number of channels in the tensor originating from 
                the previous (deeper) layer of the encoder. If merge 
                policy is "sum". The skip feature channel dim needs to
                be pooled with 1x1 conv to match input size.
            current_channels (int, default=None):
                The number of channels in the tensor originating from
                the current encoder level. If merge policy is "sum". 
                The skip feature channel dim needs to be pooled with 
                1x1 conv to match input size.
            merge_policy (str, default="summation"):
                Sum or concatenate the features together.
                One of ("summation", "concatenate")
        """
        super(MergeBlock, self).__init__()
        assert merge_policy in ("concatenate", "summation")
        self.merge_policy = merge_policy

        # channel pooling for skip features if "summation"
        if self.merge_policy == "summation":
            if prev_channels > current_channels:
                self.add_module("ch_pool", nn.Conv2d(prev_channels, current_channels, kernel_size=1, padding=0, bias=False))
            elif prev_channels < current_channels:
                self.add_module("ch_pool", nn.Conv2d(current_channels, prev_channels, kernel_size=1, padding=0, bias=False))

    def forward(self, prev_feat: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
        ------------
            x (torch.Tensor):
                input from the previous decoder layer
            skips (List[torch.Tensor]):
                all the features from the encoder
            idx (int):
                index for the for the feature from the encoder
        """
        if self.merge_policy == "concatenate":
            skips.append(prev_feat)
            prev_feat = torch.cat(skips, dim=1)
        elif self.merge_policy == "summation":
            print("prev feat shape in merge block: ", prev_feat.shape)
            for i, skip in enumerate(skips):
                print(f"{i}th skip shape in merge block: ", skip.shape)
                if skip.shape[1] > prev_feat.shape[1]:
                    print("skip gets pooled")
                    skip = self.ch_pool(skip)
                    print("shape after pooling: ", skip.shape)
                elif skip.shape[1] < prev_feat.shape[1]:
                    print("prev feat gets pooled")
                    prev_feat = self.ch_pool(prev_feat)
                    print("shape after pooling: ", prev_feat.shape)
                print("prev feat += skip \n")
                prev_feat += skip

        return prev_feat


class UnetppSkipBlock(nn.Module):
    def __init__(self,
                 decoder_channels: List[int],
                 skip_channels: List[int],
                 skip_index: int,
                 merge_policy: str="summation",
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 preactivate: bool=False,
                 n_conv_blocks: int=1,
                 **kwargs) -> None:
        """
        Unet++ skip block for one level in the decoder
        https://arxiv.org/abs/1807.10165
        
        Args:
        ---------
            decoder_channels (List[int]):
                List of the number of channels in each decoder layer output
            skip_channels (List[int]):
                List of the number of channels in each of the encoder skip tensors.
            skip_index (int):
                index of teh current skip channel in skip_channels list.
            merge_policy (str, default="summation"):
                Sum or concatenate the features together.
                One of ("summation", "concatenate")
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
            preactivate (bool, default=False)
                If True, normalization and activation are applied before convolution
            n_conv_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one residual
                multiconv block      
            
        """
        super(UnetppSkipBlock, self).__init__()
        # ignore last channels where skips are not applied
        decoder_channels = decoder_channels[:-1]
        skip_channels = skip_channels[:-1]

        if skip_index < len(skip_channels):
            skip_chl = skip_channels[skip_index]

            # sub block name index
            sub_block_idx0 = len(decoder_channels) - (skip_index + 2)

            self.ups = nn.ModuleDict()
            self.skips = nn.ModuleDict()
            self.conv_blocks = nn.ModuleDict()
            for i in range(skip_index):
                # up block for the deeper feature maps
                self.ups[f"up{i}"] = FixedUnpool()

                # merge blocks for the feature maps in the sub network
                self.skips[f"sub_skip{i}"] = MergeBlock(
                    prev_channels=skip_channels[skip_index - 1], 
                    current_channels=skip_chl,
                    merge_policy=merge_policy 
                )
                
                # conv blocks for the feature maps in the sub network
                in_channels = skip_chl
                self.conv_blocks[f"x_{sub_block_idx0}_{i + 1}"] = MultiBlockBasic(
                    in_channels=skip_chl,
                    out_channels=skip_chl,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm, 
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )

            prev_chl = decoder_channels[skip_index]
            self.final_merge = MergeBlock(prev_channels=prev_chl, current_channels=skip_chl, merge_policy=merge_policy)

    def forward(self, 
                x: torch.Tensor, 
                encoder_skips: Tuple[torch.Tensor], 
                prev_sub_skips: Tuple[torch.Tensor]=None, 
                idx: int=None) -> List[torch.Tensor]:
        """
        Args:
        ----------
            x (torch.Tensor):
                Input tensor. Shape (B, C, H, W).
            encoder_skips (Tuple[torch.Tensor]):
                Tuple of tensors generated from consecutive encoder blocks.
                Shapes (B, C, H, W).
            prev_sub_skips (Tuple[torch.Tensor], default=None):
                Tuple of tensors generated in the previous layers sub networks.
                In the paper, these are the middle blocks in the architecture schema
            idx (int, default=None):
                runnning index used to get the right skip tensor(s) from the skips
                Tuple for the skip connection.

        Returns:
        ----------
            A List of tensors where the first index contains the first sub network tensor
            and the last index contains the decoder output tensor. If there is no sub skips
            (first decoder block) then the tuple only contains the decoder output. Shapes:
            (B, C, H, W)
        """
        out = [x]
        if idx < len(encoder_skips):
            print("encoder skip shape: ", encoder_skips[idx].shape)
            skips = [encoder_skips[idx]]
            for i, (up, skip, conv) in enumerate(zip(self.ups.values(), self.skips.values(), self.conv_blocks.values())):
                print(i)
                if i == 0:
                    print("prev feat is the prev encoder skip: ", encoder_skips[idx - 1].shape)
                    prev_feat = up(encoder_skips[idx - 1])
                else:
                    print(f"prev feat is the sub skip {i} from the prev sub network")
                    prev_feat = up(prev_sub_skips[i - 1])

                print("prev feat shape after up: ", prev_feat.shape)
                sub_block = skip(prev_feat, skips[::-1])
                print("sub block shape after merge: ", sub_block.shape)
                sub_block = conv(sub_block)
                print("sub block shape after conv: ", sub_block.shape)
                skips.append(sub_block)
                print("\n")

            print("\n the final merge:")
            x = self.final_merge(x, skips)
            print("out shape after final merge: ", x.shape)
            out = [x, *skips[1:]]

        return out
                