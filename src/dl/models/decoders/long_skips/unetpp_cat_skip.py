import torch
import torch.nn as nn
from typing import List, Tuple

from .. import MultiBlockBasic
from ...modules import FixedUnpool


class CatBlock(nn.ModuleDict):
    def __init__(self) -> None:
        """
        Cat merge block for all the skip connections in unet++ 
        """
        super(CatBlock, self).__init__()
       

    def forward(
            self,
            prev_feat: torch.Tensor,
            skips: List[torch.Tensor]
        ) -> torch.Tensor:
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
        pooled_skips = [skip for skip in skips]  
        pooled_skips.append(prev_feat)
        prev_feat = torch.cat(pooled_skips, dim=1)

        return prev_feat


class UnetppCatSkipBlock(nn.Module):
    def __init__(
            self,
            skip_channel_list: List[int],
            skip_index: int,
            batch_norm: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            preactivate: bool=False,
            n_conv_blocks: int=1,
            **kwargs) -> None:
        """
        Unet++ skip block for one level in the decoder
        https://arxiv.org/abs/1807.10165

        Supports concatenation merge policy
         
        Args:
        ---------
            skip_channel_list (List[int]):
                List of the number of channels in each of the encoder
                skip tensors.
            skip_index (int):
                index of the current skip channel in skip_channels_list.
            batch_norm (str, default="bn"): 
                Normalization method. One of: "bn", "bcn", None
            activation (str, default="relu"):
                Activation method. One of: "relu", "swish". "mish"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
            n_conv_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one
                multiconv block      
        """
        super(UnetppCatSkipBlock, self).__init__()

        # ignore last channels where skips are not applied
        skip_channel_list = skip_channel_list[:-1]

        if skip_index < len(skip_channel_list):
            
            # sub block name index
            sub_block_idx0 = len(skip_channel_list) - (skip_index + 1)
  
            self.ups = nn.ModuleDict()
            self.skips = nn.ModuleDict()
            self.conv_blocks = nn.ModuleDict()

            # init encoder feat map channel_pool if reduce_param = True
            current_skip_chl = skip_channel_list[skip_index]

            conv_in_chl = current_skip_chl
            for i in range(skip_index):
                # set the enc channel num
                prev_enc_chl = current_skip_chl
                if i == 0:
                    prev_enc_chl = skip_channel_list[skip_index - 1]
                
                # up block for the deeper feature maps
                self.ups[f"up{i}"] = FixedUnpool()

                # merge blocks for the feature maps in the sub network
                self.skips[f"sub_skip{i + 1}"] = CatBlock()
                
                # conv blocks for the feature maps in the sub network
                conv_in_chl += prev_enc_chl
                
                self.conv_blocks[f"x_{sub_block_idx0}_{i+1}"] = MultiBlockBasic(
                    in_channels=conv_in_chl,
                    out_channels=current_skip_chl,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm,
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )

            # Merge all the feature maps before the final conv in the decoder
            self.final_merge = CatBlock()

    def forward(
            self, 
            x: torch.Tensor,
            idx: int,
            skips: Tuple[torch.Tensor],
            extra_skips: Tuple[torch.Tensor]=None,
            **kwargs
        ) -> List[torch.Tensor]:
        """
        Args:
        ----------
            x (torch.Tensor):
                Input tensor. Shape (B, C, H, W).
            idx (int, default=None):
                runnning index used to get the right skip tensor(s) from
                the skips tuple for the skip connection.
            skips (Tuple[torch.Tensor]):
                Tuple of tensors generated from consecutive encoder
                blocks. Shapes (B, C, H, W).
            extra_skips (Tuple[torch.Tensor], default=None):
                Tuple of tensors generated in the previous layers sub
                networks. In the paper, these are the middle blocks in 
                the architecture schema

        Returns:
        ----------
            A Tuple of tensors out tensors: First return tensor is the 
            decoder branch output the second is a list of subnetwork
            tensors that are needed in the next layer.
        """
        sub_network_tensors = None
        if idx < len(skips):
            current_skip = skips[idx]

            all_skips = [current_skip]
            for i, (up, skip, conv) in enumerate(
                zip(
                    self.ups.values(), 
                    self.skips.values(), 
                    self.conv_blocks.values()
                )
            ):
                prev_feat = up(extra_skips[i])
                sub_block = skip(prev_feat, all_skips[::-1])
                sub_block = conv(sub_block)
                all_skips.append(sub_block)

            x = self.final_merge(x, all_skips)
            sub_network_tensors = all_skips

        return x, sub_network_tensors


class UnetppCatSkipBlockLight(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channel_list: List[int],
            skip_channel_list: List[int],
            skip_index: int,
            batch_norm: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            preactivate: bool=False,
            n_conv_blocks: int=1,
            **kwargs
        ) -> None:
        """
        Unet++ skip block for one level in the decoder
        https://arxiv.org/abs/1807.10165

        Supports concatenation merge policy This is a light version 
        which has lower memory footprint. This is done similarly to 
        the unet3+ by basically just outputting a small number of 
        feature maps at every conv block which at the end sum up 
        to the number of output channels of the decoder block 
        (after concatenation)
        
        Args:
        ---------
            in_channels (int):
                The number of channels coming in from the previous head
                decoder branch
            out_channel_list (List[int]):
                List of the number of output channels in the decoder
                output tensors 
            skip_channel_list (List[int]):
                List of the number of channels in each of the encoder
                skip tensors.
            skip_index (int):
                index of the current skip channel in skip_channels_list.
            batch_norm (str, default="bn"): 
                Normalization method. One of: "bn", "bcn", None
            activation (str, default="relu"):
                Activation method. One of: "relu", "swish". "mish"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
            n_conv_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one
                residual multiconv block      
        """
        super(UnetppCatSkipBlockLight, self).__init__()

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
            cat_channels, reminder = divmod(
                out_channel_list[skip_index], (skip_index + 2)
            )

            # pre conv for the encoder skip
            self.pre_conv = MultiBlockBasic(
                in_channels=current_skip_chl,
                out_channels=cat_channels,
                n_blocks=n_conv_blocks,
                batch_norm=batch_norm,
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate
            )

            # post conv for the in decoder feat map
            self.post_conv = MultiBlockBasic(
                in_channels=in_channels,
                out_channels=cat_channels + reminder,
                n_blocks=n_conv_blocks,
                batch_norm=batch_norm,
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate
            )

            conv_in_chl = out_channel_list[skip_index - 1] // (skip_index + 1)
            for i in range(skip_index):
                prev_enc_chl = cat_channels
                
                # up block for the deeper feature maps
                self.ups[f"up{i}"] = FixedUnpool()

                # merge blocks for the feature maps in the sub network
                self.skips[f"sub_skip{i + 1}"] = CatBlock()
                
                # conv blocks for the feature maps in the sub network
                conv_in_chl += prev_enc_chl
                
                self.conv_blocks[f"x_{sub_block_idx0}_{i + 1}"] = MultiBlockBasic(
                    in_channels=conv_in_chl,
                    out_channels=cat_channels,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm, 
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )

            # Merge all the feature maps before the final conv in the decoder
            self.final_merge = CatBlock()

    def forward(
            self,
            x: torch.Tensor,
            idx: int,
            skips: Tuple[torch.Tensor],
            extra_skips: Tuple[torch.Tensor]=None,
            **kwargs
        ) -> List[torch.Tensor]:
        """
        Args:
        ----------
            x (torch.Tensor):
                Input tensor. Shape (B, C, H, W).
            idx (int, default=None):
                runnning index used to get the right skip tensor(s) from
                the skips tuple for the skip connection.
            skips (Tuple[torch.Tensor]):
                Tuple of tensors generated from consecutive encoder 
                blocks. Shapes (B, C, H, W).
            extra_skips (Tuple[torch.Tensor], default=None):
                Tuple of tensors generated in the previous layers sub 
                networks. In the paper, these are the middle blocks in
                the architecture schema

        Returns:
        ----------
            A Tuple of tensors out tensors: First return tensor is the
            decoder branch output the second is a list of subnetwork
            tensors that are needed in the next layer.
        """
        sub_network_tensors = None
        if idx < len(skips):

            current_skip = skips[idx]
            current_skip = self.pre_conv(current_skip)
            all_skips = [current_skip]
            for i, (up, skip, conv) in enumerate(
                zip(
                    self.ups.values(), 
                    self.skips.values(), 
                    self.conv_blocks.values()
                )
            ):
                prev_feat = up(extra_skips[i])
                sub_block = skip(prev_feat, all_skips[::-1])
                sub_block = conv(sub_block)
                all_skips.append(sub_block)

            x = self.post_conv(x)
            x = self.final_merge(x, all_skips)
            sub_network_tensors = all_skips

        return x, sub_network_tensors
                