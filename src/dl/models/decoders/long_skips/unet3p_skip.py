import torch
import torch.nn as nn
from typing import Tuple, List, Union

from .merge_blocks.utils import merge_func
from ...modules import FixedUnpool
from ...modules.conv.utils import conv_block_func


class Unet3pSkip(nn.Module):
    def __init__(
        self,
        stage_ix: int,
        in_channels: int,
        dec_channels: List[int],
        skip_channels: List[int],
        dec_out_dims: List[int],
        short_skip: str=None,
        conv_block_type: str="basic",
        activation: str="relu",
        normalization: str="bn",
        weight_standardize: bool=False,
        preactivate: bool=False,
        n_conv_blocks: int=1,
        merge_policy: str="concatenate",
        lite_version: bool=False,
        **kwargs
    ) -> None:
        """
        U-net3+ like skip connection block.

        UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation
            - https://arxiv.org/abs/2004.08790

        Sets up a conv block for the upsampled feature map from the 
        previous decoder stage and dynamically sets up the conv blocks 
        for the outputs of encoder stages and previous decoder stages. 
        The number of these conv blocks depend on the decoder stage ix.

        Supports different merging policies of the feature maps and also
        a lite version can be used where the decoder-to-decoder skips 
        are skipped.

        Args:
        ----------
            stage_ix (int):
                Index number signalling the current decoder stage
            in_channels (int):
                Number of channels in the upsampled decoder feature map
            dec_channels (List[int]):
                List of the number of output channels in all of the 
                decoder stages. Order is bottom up. Includes the number 
                of out channels at the last encoder head stage 
                as the first element in the list. 
                e.g. [2048, 256, 128, 64, 32, 16]
            skip_channels (List[int]):
                List of the number of channels in each of the encoder
                stages. Order is bottom up. This list does not include
                the final bottleneck stage out channels since it is 
                included in `dec_channels`. Also, the last element 
                of the list is zero to avoid a skip at the final decoder
                stage. e.g. [1024, 512, 256, 64, 0] 
            dec_out_dims (List[int]):
                List of the spatial dimensions (H, W) of each encoder 
                stage outptut. e.g. [8, 16, 32, 64, 128, 256]. Order is
                bottom up. Assumption is that feature maps are square 
                shaped i.e. H == W. The encoder head dim is the first ix
            short_skip (str, default=None):
                The type of short skip connection applied in the conv
                blocks of this module. One of ("residual", "dense", None)
            activation (str):
                Activation method. One of: "mish", "swish", "relu",
                "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh",
                "sigmoid", "silu", "prelu", "leaky-relu", "elu",
                "hardshrink", "tanhshrink", "hardsigmoid"
            normalization (str): 
                Normalization method to be used.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
            n_conv_blocks (int, default=2):
                Number of conv-blocks used in the skip connection
            merge_policy (str):
                One of: "summation", "concatenate" 
            lite_version (bool, default=False):
                If True, the dense decoder-to-decoder skips are not
                utilized at all. Reduces the model params quite a lot.      
        """
        if in_channels <= 4:
            raise ValueError(
                f"Input channels need to be larger than 4. Got {in_channels}"
            )

        super().__init__()
        self.normalization = normalization
        self.act = activation
        self.weight_standardize = weight_standardize
        self.preactivate = preactivate
        self.n_conv_blocks = n_conv_blocks
        self.lite_version = lite_version
        self.merge_policy = merge_policy
        self.stage_ix = stage_ix
        self.short_skip = short_skip if short_skip is not None else "basic"
        self.conv_block_type = conv_block_type
        self.in_channels = in_channels
        self.n_skips = len(skip_channels) + 1
        self.conv_out_channels = []

        enc_head_dim = dec_out_dims[0]
        enc_head_channels = dec_channels[0]

        out_dim_list = dec_out_dims[1:]
        out_chl_list = dec_channels[1:]

        self.current_out_dim = out_dim_list[stage_ix]
        self.current_out_chl = out_chl_list[stage_ix]

        # current skip dims from the encoder side
        self.enc_skip_dims = out_dim_list[stage_ix:]

        # current skip channels from the encoder side
        self.enc_skip_chls = skip_channels[stage_ix:]

        if stage_ix != 0:
            out_dim_list = [enc_head_dim*2] + out_dim_list
            out_chl_list = [enc_head_channels] + out_chl_list

        # current skip dims from the decoder side
        self.dec_skip_dims = out_dim_list[:stage_ix]

        # current skip channels from the decoder side
        self.dec_skip_chls = out_chl_list[:stage_ix]

        # Compute the output channels for the skip conv blocks
        cat_channels, num_out_feats = self._get_conv_chls()

        # The number of out channels for each of the skip conv blocks
        # The number depends on the merging policy
        self.skip_out_chl = num_out_feats
        if merge_policy == "concatenate":
            self.skip_out_chl = cat_channels

        # convblock for the raw upsampled incoming feat map
        self.decoder_feat_conv = conv_block_func(
            name=self.conv_block_type,
            skip_type=self.short_skip,
            in_channels=in_channels, 
            out_channels=num_out_feats,
            n_blocks=n_conv_blocks,
            normalization=normalization, 
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=preactivate
        )
        self.conv_out_channels.append(self.decoder_feat_conv.out_channels)

        # if there are skip channels (I.e. this is not the last decoder 
        # stage), init all the encoder-to-decoder skip convs
        if self.enc_skip_chls:
            self.down_scales, self.enc2dec_convs = self._get_ops(
                from_to="enc2dec",
                target_size=self.current_out_dim,
                out_dim_list=self.enc_skip_dims,
                in_chl_list=self.enc_skip_chls,
                out_chls=self.skip_out_chl,
            )

            # get the decoder2decoder skips if not lite_version
            if not self.lite_version and stage_ix != 0:
                self.up_scales, self.dec2dec_convs = self._get_ops(
                    from_to="dec2dec",
                    target_size=self.current_out_dim,
                    out_dim_list=self.dec_skip_dims,
                    in_chl_list=self.dec_skip_chls,
                    out_chls=self.skip_out_chl
                )

        self.merge = merge_func(
            merge_policy,
            out_channels=self.out_channels,
            in_channels=self.conv_out_channels[0],
            skip_channels=self.conv_out_channels[1:]
        )

    @property
    def out_channels(self) -> int:
        if self.merge_policy == "summation":
            out_chl = self.conv_out_channels[0]
        else:
            out_chl = sum(self.conv_out_channels)
        
        return out_chl

    @staticmethod
    def scale(in_size: int, target_size: int) -> nn.Module:
        """
        Get the scaling operation for the feature map
        Can be downsampling, identity or unpooling depending on the
        in and target sizes.

        Args:
        ---------
            in_size (int):
                H/W of the input tensor
            target_size (int):
                H/W of the output tesnor

        Returns:
        ---------
            nn.MaxPool2d: FixedUnpool or nn.identity
        """
        scale_factor = in_size / target_size

        if scale_factor > 1:
            scale_op = nn.MaxPool2d(
                kernel_size=int(scale_factor), ceil_mode=True
            )
        elif scale_factor < 1:
            scale_op = FixedUnpool(scale_factor=int(1 / scale_factor)) 
        else:
            scale_op = nn.Identity()

        return scale_op

    def _get_conv_chls(self) -> Tuple[int, int]:
        """
        Convolution channel arithmetic to get the right number of chls
        for the convolutions in this module. If merge policy is to cat
        all the channels, the number of out channels is divided evenly
        such that it matches the decoder output channels. If the merge
        policy is to sum, then no need to divide.

        Returns:
        ---------
            Tuple[int, int]: The number of channels for the skip conv 
                             blocks and the number of channels for the
                             conv block used for the upsampled feat map.

        """
        cat_channels = None
        num_out_feats = self.current_out_chl
        
        if self.merge_policy == "concatenate":
            # divide the number of out channels for each of the skip 
            # conv blocks to be even.
            divider = self.n_skips
            if self.lite_version:
                divider = len(self.enc_skip_chls) + 1

            cat_channels, reminder = divmod(
                self.current_out_chl, divider
            )

            # The reminder will be added to the decoder-to-decoder conv 
            # block such that the out chls of this module match the 
            # current ``current_dec_chl``. At the final decoder
            # stage ther is no skip connection -> no need to divide the 
            # out channels evenly
            if self.enc_skip_chls:
                num_out_feats = cat_channels + reminder

        return cat_channels, num_out_feats

    def _get_ops(
            self,
            from_to: str,
            target_size: int,
            out_dim_list: List[int],
            in_chl_list: List[int],
            out_chls: int,
        ) -> Tuple[nn.ModuleDict, nn.ModuleDict]:
        """
        Get the convolution and scaling operations needed for the skip
        connection at the current decoder stage
    
        Args:
        ---------
            from_to (str):
                Specifies the part of the network where the feature map
                starts and where it propagates
            target_size (int):
                The spatial dims (H, W) of the output feature map
            out_dim_list (List[int]):
                List of the output spatial dimensions of the 
            in_chl_list (List[int]):
                List of the number of channels from the encoder and
                previous decoder stages
            out_chls (int):
                The number of output channels for the conv blocks

        Returns:
        ---------
            Tuple: ModuleDicts defining the scale and conv operations
                   for the skip connection.
        """
        assert from_to in ("enc2dec", "dec2dec")
        scales = nn.ModuleDict()
        convs = nn.ModuleDict()

        for i, (in_, dim_) in enumerate(zip(in_chl_list, out_dim_list)):
            scale_op = self.scale(dim_, target_size)
            scales[f"{from_to}_scale{i + 1}"] = scale_op
            conv_op = conv_block_func(
                name=self.conv_block_type,
                skip_type=self.short_skip,
                in_channels=in_, 
                out_channels=out_chls,
                n_blocks=self.n_conv_blocks,
                normalization=self.normalization, 
                activation=self.act,
                weight_standardize=self.weight_standardize,
                preactivate=self.preactivate
            )
            convs[f"{from_to}_conv{i + 1}"] = conv_op
            self.conv_out_channels.append(conv_op.out_channels)

        return scales, convs

    def forward(
            self,
            x: torch.Tensor,
            ix: int,
            skips: Tuple[torch.Tensor],
            extra_skips: List[torch.Tensor]=None,
            **kwargs
        ) -> Tuple[torch.Tensor, Union[None, List[torch.Tensor]]]:
        """
        Args:
        ------------
            x (torch.Tensor):
                Upsampled input tensor from the previous decoder layer
            ix (int):
                Index for the the feature map from the encoder
            skips (Tuple[torch.Tensor]):
                all the feature maps from the encoder except the encoder
                head which is not used for any skip connection
            extra_skips (List[torch.Tensor], default=None):
                extra skip connections. Here, the deocder-to-decoder
                skip connections

        Returns:
        ------------
            Tuple: First return is the decoder branch output. The second
                   return value are the outputs from the prev decoder 
                   branches that are used for skips. If `lite_version` 
                   is being used the second return value is None
        """
        # Init the previous decoder feat maps
        decoder_out_features = None
        if not self.lite_version:
            decoder_out_features = [x] if ix == 0 else extra_skips

        # Convolve the input feat map
        x = self.decoder_feat_conv(x)
        
        # Loop over the scale and conv blocks
        if ix < len(skips):
            skip_features = []
            decoder_features = []
            skips = skips[ix:]
            
            if skips:
                # loop over the encoder features (enc2dec skips)
                for i, (scale, conv_block) in enumerate(
                    zip(self.down_scales.values(), self.enc2dec_convs.values())
                ):
                    skip_feat = scale(skips[i])
                    skip_feat = conv_block(skip_feat)
                    skip_features.append(skip_feat)

            if extra_skips and not self.lite_version:
                # loop over the prev decoder features (dec2dec skips)
                for i, (scale, conv_block) in enumerate(
                    zip(self.up_scales.values(), self.dec2dec_convs.values())
                ):
                    dense_feat = scale(extra_skips[i])
                    dense_feat = conv_block(dense_feat)
                    decoder_features.append(dense_feat)
                    

            skip_feats = skip_features + decoder_features
            x = self.merge(x, skip_feats)
            
            # Add to prev decoder feature list
            if not self.lite_version:
                decoder_out_features.append(x)

        return x, decoder_out_features