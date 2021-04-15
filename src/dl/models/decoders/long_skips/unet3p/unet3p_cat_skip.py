import torch
import torch.nn as nn
from typing import Tuple, List

from ....modules import FixedUnpool
from ... import MultiBlockBasic


class Unet3pCatSkipBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channel_list: List[int],
                 skip_channel_list: List[int],
                 skip_index: int,
                 out_dims: List[int],
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 preactivate: bool=False,
                 n_conv_blocks: int=1,
                 **kwargs) -> None:
        """
        U-net3+ like skip connection block.
        https://arxiv.org/abs/2004.08790

        Args:
        ----------
            in_channels (int):
                Number of channels in the upsampled decoder feature map
            out_channel_list (List[int]):
                List of the number of output channels in decoder blocks.
                First index contains the number of head channels
            skip_channel_list (List[int]):
                List of the number of channels in each of the encoder skip tensors.
            skip_index (int):
                index of the current skip channel in skip_channels_list.
            out_dims (List[int]):
                List of the heights/widths of each encoder/decoder feature map
                e.g. [256, 128, 64, 32, 16]. Assumption is that feature maps are
                square shaped.
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
        super(Unet3pCatSkipBlock, self).__init__()
        current_out_chl = out_channel_list[skip_index + 1]
        dense_channels = out_channel_list[:skip_index]
        prev_dims = out_dims[:skip_index]
        
        if prev_dims:
            prev_dims.insert(0, prev_dims[0]) # insert head dims to pre dims
        
        skip_channels = skip_channel_list[skip_index:-1] # no skip at the final block
        out_dims = out_dims[skip_index:] # no skip at the final block

        # divide the number of out channels for conv blocks evenly and save the 
        # remainder so that the final number of out channels can be set to out_channels
        cat_channels, reminder = divmod(current_out_chl, len(skip_channel_list))
       
        # at the final deocder block out_channels need to be same as the input arg 
        num_out_features = cat_channels + reminder if skip_channels else current_out_chl
        
        # TODO option for short skips
        self.decoder_feat_conv = MultiBlockBasic(
            in_channels=in_channels, 
            out_channels=num_out_features,
            n_blocks=n_conv_blocks,
            batch_norm=batch_norm, 
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=preactivate
        )

        # if there are skip channels, init the all the skip convs
        # init the dense decoder to decoder skips
        if skip_channels:
            target_size = out_dims[0]
            self.up_scales = nn.ModuleDict()
            self.dense_convs = nn.ModuleDict()
            for i, (in_chl, prev_dim) in enumerate(zip(dense_channels, prev_dims), 1):
                up_scale = self.scale(prev_dim, target_size)
                self.up_scales[f"up_scale{i}"] = up_scale
                self.dense_convs[f"dec_skip_conv{i}"] = MultiBlockBasic(
                    in_channels=in_chl, 
                    out_channels=cat_channels,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm, 
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )

            self.down_scales = nn.ModuleDict()
            self.skip_convs = nn.ModuleDict()
            for i, (in_chl, out_dim) in enumerate(zip(skip_channels, out_dims)):
                down_scale = self.scale(out_dim, target_size)
                self.down_scales[f"down_scale{i + 1}"] = down_scale
                self.skip_convs[f"skip_conv{i + 1}"] = MultiBlockBasic(
                    in_channels=in_chl, 
                    out_channels=cat_channels,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm, 
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )
    
    @staticmethod
    def scale(in_size: int, target_size: int) -> nn.Module:
        """
        Get the scaling operation for the feature map
        Can be downsampling or identity

        Args:
        ---------
            in_size (int):
                H/W of the input tensor
            target_size (int):
                H/W of the output tesnor

        Returns:
        ---------
            nn.MaxPool2d, FixedUnpool or nn.identity if scaling not needed
        """
        scale_factor = in_size / target_size
        
        if scale_factor > 1:
            scale_op = nn.MaxPool2d(kernel_size=int(scale_factor), ceil_mode=True)
        elif scale_factor < 1:
            scale_op = FixedUnpool(scale_factor=int(1 / scale_factor)) 
        else:
            scale_op = nn.Identity()

        return scale_op

    def forward(self, 
                x: torch.Tensor, 
                idx: int, 
                skips: Tuple[torch.Tensor], 
                extra_skips: List[torch.Tensor]=None, 
                **kwargs) -> torch.Tensor:
        """
        Args:
        ------------
            x (torch.Tensor):
                input from the previous decoder layer
            idx (int):
                index for the the feature from the encoder
            skips (Tuple[torch.Tensor]):
                all the features from the encoder
            extra_skips (List[torch.Tensor], default=None):
                extra skip connections. Here, the dense deocder block 
                to upper decoder block connections

        Returns:
        ------------
            Tuple of tensors. First return is the decoder branch output.
            The second return value are the outputs from the previous decoder branches
        """
        decoder_out_features = [x] if idx == 0 else extra_skips
        x = self.decoder_feat_conv(x)

        if idx < len(skips):
            skip_features = []
            decoder_features = []
            skips = skips[idx:]
            
            if skips:
                # loop over the encoder features
                for i, (scale, conv_block) in enumerate(zip(self.down_scales.values(), self.skip_convs.values())):
                    skip_feat = scale(skips[i])
                    skip_feat = conv_block(skip_feat)
                    skip_features.append(skip_feat)

            if extra_skips:
                # loop over the decoder features
                for i, (scale, conv_block) in enumerate(zip(self.up_scales.values(), self.dense_convs.values())):
                    dense_feat = scale(extra_skips[i])
                    dense_feat = conv_block(dense_feat)
                    decoder_features.append(dense_feat)

            all_features = skip_features + decoder_features
            all_features.append(x)
            x = torch.cat(all_features, dim=1)
            decoder_out_features.append(x)

        return x, decoder_out_features




class Unet3pCatSkipBlockLight(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channel_list: List[int],
                 skip_channel_list: List[int],
                 skip_index: int,
                 out_dims: List[int],
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 preactivate: bool=False,
                 n_conv_blocks: int=1,
                 **kwargs) -> None:
        """
        U-net3+ like skip connection block.
        https://arxiv.org/abs/2004.08790

        This has no dense decoder to decoder skip connections
        --> Lighter memory footprint. 

        Args:
        ----------
            in_channels (int):
                Number of channels in the upsampled decoder feature map
            out_channel_list (List[int]):
                List of the number of output channels in decoder blocks.
                First index contains the number of head channels
            skip_channel_list (List[int]):
                List of the number of channels in each of the encoder skip tensors.
            skip_index (int):
                index of the current skip channel in skip_channels_list.
            out_dims (List[int]):
                List of the heights/widths of each encoder/decoder feature map
                e.g. [256, 128, 64, 32, 16]. Assumption is that feature maps are
                square shaped.
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
        super(Unet3pCatSkipBlockLight, self).__init__()
        current_out_chl = out_channel_list[skip_index + 1]        
        skip_channels = skip_channel_list[skip_index:-1] # no skip at the final block
        out_dims = out_dims[skip_index:] # no skip at the final block


        # divide the number of out channels for conv blocks evenly and save the 
        # remainder so that the final number of out channels can be set to out_channels
        cat_channels, reminder = divmod(current_out_chl, len(skip_channels) + 1)#len(skip_channel_list))
       
        # at the final deocder block out_channels need to be same as the input arg 
        num_out_features = cat_channels + reminder if skip_channels else current_out_chl
        
        # TODO option for short skips
        self.decoder_feat_conv = MultiBlockBasic(
            in_channels=in_channels, 
            out_channels=num_out_features,
            n_blocks=n_conv_blocks,
            batch_norm=batch_norm, 
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=preactivate
        )
        
        if skip_channels:
            target_size = out_dims[0]
            self.down_scales = nn.ModuleDict()
            self.skip_convs = nn.ModuleDict()
            for i, (in_chl, out_dim) in enumerate(zip(skip_channels, out_dims)):
                down_scale = self.scale(out_dim, target_size)
                self.down_scales[f"down_scale{i + 1}"] = down_scale
                self.skip_convs[f"skip_conv{i + 1}"] = MultiBlockBasic(
                    in_channels=in_chl, 
                    out_channels=cat_channels,
                    n_blocks=n_conv_blocks,
                    batch_norm=batch_norm, 
                    activation=activation,
                    weight_standardize=weight_standardize,
                    preactivate=preactivate
                )
    
    @staticmethod
    def scale(in_size: int, target_size: int) -> nn.Module:
        """
        Get the scaling operation for the feature map
        Can be downsampling or identity

        Args:
        ---------
            in_size (int):
                H/W of the input tensor
            target_size (int):
                H/W of the output tesnor

        Returns:
        ---------
            nn.MaxPool2d, FixedUnpool or nn.identity if scaling not needed
        """
        scale_factor = in_size / target_size
        
        if scale_factor > 1:
            scale_op = nn.MaxPool2d(kernel_size=int(scale_factor), ceil_mode=True)
        else:
            scale_op = nn.Identity()

        return scale_op

    def forward(self, 
                x: torch.Tensor, 
                idx: int, 
                skips: Tuple[torch.Tensor], 
                **kwargs) -> torch.Tensor:
        """
        Args:
        ------------
            x (torch.Tensor):
                input from the previous decoder layer
            idx (int):
                index for the the feature from the encoder
            skips (Tuple[torch.Tensor]):
                all the features from the encoder

        Returns:
        ------------
            Tuple of tensors. First return is the decoder branch output.
            The second return value is None. Code doesnt align w/o it..
        """
        x = self.decoder_feat_conv(x)

        if idx < len(skips):
            skip_features = []
            skips = skips[idx:]            
            if skips:
                # loop over the encoder features
                for i, (scale, conv_block) in enumerate(zip(self.down_scales.values(), self.skip_convs.values())):
                    skip_feat = scale(skips[i])
                    skip_feat = conv_block(skip_feat)
                    skip_features.append(skip_feat)

            skip_features.append(x)
            x = torch.cat(skip_features, dim=1)

        return x, None
