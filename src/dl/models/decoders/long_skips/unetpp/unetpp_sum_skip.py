import torch
import torch.nn as nn
from typing import List, Tuple

from ....modules import FixedUnpool
from ... import MultiBlockBasic


class SumBlock(nn.ModuleDict):
    def __init__(self, 
                 prev_channels: int=None,
                 current_channels: int=None) -> None:
        """
        Sum merge block for all the skip connections in unet++ 

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
        """
        super(SumBlock, self).__init__()

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
        for i, skip in enumerate(skips):
            if skip.shape[1] < prev_feat.shape[1]:
                prev_feat = self.ch_pool(prev_feat)
            elif skip.shape[1] > prev_feat.shape[1]:
                skip = self.ch_pool(skip)

            prev_feat += skip

        return prev_feat



class UnetppSumSkipBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channel_list: List[int],
                 skip_channel_list: List[int],
                 skip_index: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 preactivate: bool=False,
                 n_conv_blocks: int=1,
                 reduce_params: bool=False,
                 **kwargs) -> None:
        """
        Unet++ skip block for one level in the decoder
        https://arxiv.org/abs/1807.10165

        Supports only summation merge policy
        Set reduce_params = True if the original version (i.e. reduce_params = False)
        takes too much memory.

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
                If True, uses the number of out_channels
                in the skip blocks rather than the number of skip_channels.
                Results is much smaller memory footprint if the decoder out channels
                are smaller than the encoder out channels. (Which is usually the case)
            
        """
        super(UnetppSumSkipBlock, self).__init__()

        # ignore last channels where skips are not applied
        skip_channel_list = skip_channel_list[:-1]

        if skip_index < len(skip_channel_list):
            
            # sub block name index
            sub_block_idx0 = len(skip_channel_list) - (skip_index + 1)
  
            self.ups = nn.ModuleDict()
            self.skips = nn.ModuleDict()
            self.conv_blocks = nn.ModuleDict()

            # Create sub blocks
            current_skip_chl = skip_channel_list[skip_index]
            conv_in_chl = current_skip_chl
            for i in range(skip_index):

                # num channels in the prev encoder skip
                prev_chl = skip_channel_list[skip_index - 1]
                
                # up block for the deeper feature maps
                self.ups[f"up{i}"] = FixedUnpool()

                # merge blocks for the feature maps in the sub network
                self.skips[f"sub_skip{i + 1}"] = SumBlock(
                    prev_channels=prev_chl,
                    current_channels=current_skip_chl,
                )
                                
                self.conv_blocks[f"x_{sub_block_idx0}_{i + 1}"] = MultiBlockBasic(
                    in_channels=conv_in_chl,
                    out_channels=current_skip_chl,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm, 
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )

            # Merge all the feature maps before the final conv in the decoder
            self.final_merge = SumBlock(
                prev_channels=in_channels, 
                current_channels=current_skip_chl
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

            current_skip = skips[idx]
            all_skips = [current_skip]
            for i, (up, skip, conv) in enumerate(zip(self.ups.values(), self.skips.values(), self.conv_blocks.values())):
                prev_feat = up(extra_skips[i])
                sub_block = skip(prev_feat, all_skips[::-1])
                sub_block = conv(sub_block)
                all_skips.append(sub_block)

            x = self.final_merge(x, all_skips)
            sub_network_tensors = all_skips

        return x, sub_network_tensors


class UnetppSumSkipBlockLight(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channel_list: List[int],
                 skip_channel_list: List[int],
                 skip_index: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 preactivate: bool=False,
                 n_conv_blocks: int=1,
                #  reduce_params: bool=False,
                 **kwargs) -> None:
        """
        Unet++ skip block for one level in the decoder
        https://arxiv.org/abs/1807.10165

        Supports only summation merge policy
        Set reduce_params = True if the original version (i.e. reduce_params = False)
        takes too much memory.

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
                If True, uses the number of out_channels
                in the skip blocks rather than the number of skip_channels.
                Results is much smaller memory footprint if the decoder out channels
                are smaller than the encoder out channels. (Which is usually the case)
            
        """
        super(UnetppSumSkipBlockLight, self).__init__()

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
            
            # pre conv for the encoder skip
            self.pre_conv = MultiBlockBasic(
                in_channels=current_skip_chl,
                out_channels=out_channel_list[skip_index],
                n_blocks=n_conv_blocks,
                batch_norm=batch_norm, 
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate
            )

            # Create the sub blocks
            conv_in_chl = current_skip_chl
            for i in range(skip_index):

                current_skip_chl = out_channel_list[skip_index]
                conv_in_chl = out_channel_list[skip_index]
                prev_chl = out_channel_list[skip_index - 1]
                
                # up block for the deeper feature maps
                self.ups[f"up{i}"] = FixedUnpool()

                # merge blocks for the feature maps in the sub network
                self.skips[f"sub_skip{i + 1}"] = SumBlock(
                    prev_channels=prev_chl,
                    current_channels=current_skip_chl,
                )
                                
                self.conv_blocks[f"x_{sub_block_idx0}_{i + 1}"] = MultiBlockBasic(
                    in_channels=conv_in_chl,
                    out_channels=current_skip_chl,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm, 
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )

            # Merge all the feature maps before the final conv in the decoder
            self.final_merge = SumBlock(
                prev_channels=in_channels, 
                current_channels=out_channel_list[skip_index]
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
            current_skip = skips[idx]

            # channel pool encoder fmaps if reduce_params
            current_skip = self.pre_conv(current_skip)

            all_skips = [current_skip]
            for i, (up, skip, conv) in enumerate(zip(self.ups.values(), self.skips.values(), self.conv_blocks.values())):
                prev_feat = up(extra_skips[i])
                sub_block = skip(prev_feat, all_skips[::-1])
                sub_block = conv(sub_block)
                all_skips.append(sub_block)

            x = self.final_merge(x, all_skips)
            sub_network_tensors = all_skips

        return x, sub_network_tensors