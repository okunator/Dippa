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
        # if self.merge_policy == "summation":
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
        pooled_skips = []
        print("prev feat shape in the beginning of merge block: ", prev_feat.shape)
        for i, skip in enumerate(skips):
            print(f"{i}th skip shape in merge block: ", skip.shape)
            if skip.shape[1] < prev_feat.shape[1]:
                print("prev feat gets pooled")
                prev_feat = self.ch_pool(prev_feat)

            if self.merge_policy == "summation":
                if skip.shape[1] > prev_feat.shape[1]:
                    print("skip gets pooled")
                    skip = self.ch_pool(skip)
                    print("skip shape: ", skip.shape)

            print("prev_feat shape: ", prev_feat.shape)
    
            if self.merge_policy == "summation":
                prev_feat += skip
                print("prev feat += skip \n")
            else:
                pooled_skips.append(skip)

        if self.merge_policy == "concatenate":
            pooled_skips.append(prev_feat)
            print("pooled skip shapes: ", [f.shape for f in pooled_skips])
            prev_feat = torch.cat(pooled_skips, dim=1)
            print("prev feat cat skip. Shape: ", prev_feat.shape, "\n")

        return prev_feat


class UnetppSkipBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channel_list: List[int],
                 skip_channel_list: List[int],
                 skip_index: int,
                 merge_policy: str="summation",
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 preactivate: bool=False,
                 n_conv_blocks: int=1,
                 reduce_params: bool=True,
                 **kwargs) -> None:
        """
        Unet++ skip block for one level in the decoder
        https://arxiv.org/abs/1807.10165

        Supports concatenation, summation merge policies
        Parameter reduction similar to unet3+ also available
        (This became a huge spaghetti kludge because of these but whatevs)
        
        Args:
        ---------
            in_channels (int):
                The number of channels coming in from the previous head/decoder branch
            out_channel_list (List[int]):
                List of the number of output channels in the decoder output tensors 
            skip_channel_list (List[int]):
                List of the number of channels in each of the encoder skip tensors.
            skip_index (int):
                index of the current skip channel in skip_channels_list.
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
            reduce_params (bool, default=True):
                If True, divides the channels from out_channel_list evenly to all
                skip blocks similarly to unet3+
            
        """
        super(UnetppSkipBlock, self).__init__()
        self.reduce_params = reduce_params

        # ignore last channels where skips are not applied
        skip_channel_list = skip_channel_list[:-1]

        if skip_index < len(skip_channel_list):
            
            # sub block name index
            sub_block_idx0 = len(skip_channel_list) - (skip_index + 1)
  
            self.ups = nn.ModuleDict()
            self.skips = nn.ModuleDict()
            self.conv_blocks = nn.ModuleDict()

            # init encoder feat map channel_pool if reduce_param = True
            reminder = 0
            current_skip_chl = skip_channel_list[skip_index]
            if self.reduce_params:

                current_skip_chl = out_channel_list[skip_index]
                if merge_policy == "concatenate":
                    current_skip_chl, reminder = divmod(out_channel_list[skip_index], (skip_index + 2))

                # channel pool for the encoder skip
                self.pre_chl_pool =  nn.Conv2d(
                    skip_channel_list[skip_index], 
                    current_skip_chl + reminder, 
                    kernel_size=1, 
                    padding=0, 
                    bias=False
                )

            # Create the sub blocks
            conv_in_channels = current_skip_chl
            for i in range(skip_index):
                
                prev_enc_channel = skip_channel_list[skip_index - 1]
                if reduce_params and i != 0:
                    prev_enc_channel = out_channel_list[skip_index - 1] // (skip_index + 1)

                # up block for the deeper feature maps
                self.ups[f"up{i}"] = FixedUnpool()

                # merge blocks for the feature maps in the sub network
                self.skips[f"sub_skip{i + 1}"] = MergeBlock(
                    prev_channels=prev_enc_channel,
                    current_channels=current_skip_chl,
                    merge_policy=merge_policy 
                )
                
                # conv blocks for the feature maps in the sub network
                if merge_policy == "concatenate":
                    conv_in_channels += (current_skip_chl + reminder*int(i == 0))
                
                self.conv_blocks[f"x_{sub_block_idx0}_{i + 1}"] = MultiBlockBasic(
                    in_channels=conv_in_channels,
                    out_channels=current_skip_chl,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm, 
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )

            # Merge all the feature maps before the final conv in the decoder
            self.final_merge = MergeBlock(
                prev_channels=in_channels, 
                current_channels=current_skip_chl, # + reminder, 
                merge_policy=merge_policy
            )

    def forward(self, 
                x: torch.Tensor, 
                idx: int,
                skips: Tuple[torch.Tensor], 
                extra_skips: Tuple[torch.Tensor]=None, 
                **kwargs) -> List[torch.Tensor]:
        """
        Args:
        ----------
            x (torch.Tensor):
                Input tensor. Shape (B, C, H, W).
            idx (int, default=None):
                runnning index used to get the right skip tensor(s) from the skips
                Tuple for the skip connection.
            skips (Tuple[torch.Tensor]):
                Tuple of tensors generated from consecutive encoder blocks.
                Shapes (B, C, H, W).
            extra_skips (Tuple[torch.Tensor], default=None):
                Tuple of tensors generated in the previous layers sub networks.
                In the paper, these are the middle blocks in the architecture schema

        Returns:
        ----------
            A Tuple of tensors out tensors. First return tensor is the decoder branch output
            the second is a list of subnetwork tensors that are needed in the next layer.
        """
        sub_network_tensors = None
        if idx < len(skips):
            print("current skip shape: ", skips[idx].shape)
            current_skip = skips[idx]
            prev_skip = skips[idx - 1]
            if self.reduce_params:
                current_skip = self.pre_chl_pool(current_skip)
                print("current_skip shape after chl pool: ", current_skip.shape)

            all_skips = [current_skip]
            for i, (up, skip, conv) in enumerate(zip(self.ups.values(), self.skips.values(), self.conv_blocks.values())):
                if i == 0:
                    print("PREV ENCODER skip: ", prev_skip.shape)
                    prev_feat = up(prev_skip)
                else:
                    print(f"PREV SUB skip {i} from the prev sub network")
                    prev_feat = up(extra_skips[i - 1])

                print("prev feat shape after up: ", prev_feat.shape)
                sub_block = skip(prev_feat, all_skips[::-1])
                print("sub block shape after merge: ", sub_block.shape)
                sub_block = conv(sub_block)
                print("sub block shape after conv: ", sub_block.shape)
                all_skips.append(sub_block)
                print("\n")

            print("the final merge:")
            x = self.final_merge(x, all_skips)
            print("out shape after final merge: ", x.shape)
            sub_network_tensors = all_skips[1:]

        return x, sub_network_tensors
                