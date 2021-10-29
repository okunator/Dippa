import torch
from torch._C import ModuleDict
import torch.nn as nn
from typing import Tuple, List

from torch.nn.modules import normalization

from src.dl.models.modules import FixedUnpool
from src.dl.models.decoders import (
    MultiBlockBasic, MultiBlockDense, MultiBlockResidual
)


class Unet3pSkip(nn.Module):
    def __init__(
        self,
        dec_stage_ix: int,
        in_channels: int,
        dec_out_channels: List[int],
        enc_out_channels: List[int],
        dec_out_dims: List[int],
        short_skip: str=None,
        batch_norm: str="bn",
        activation: str="relu",
        weight_standardize: bool=False,
        preactivate: bool=False,
        n_conv_blocks: int=1,
        merge_policy: str="concatenate",
        lite_version: bool=True,
        **kwargs
    ) -> None:
        """
        U-net3+ like skip connection block.
        https://arxiv.org/abs/2004.08790

        Sets up a conv block for the upsampled feature map from the 
        previous decoder stage and dynamically sets up the conv blocks 
        for the outputs of encoder stages and previous decoder stages. 
        The number of these conv blocks depend on the decoder stage ix.

        Supports different merging policies of the feature maps and also
        a lite version can be used where the decoder-to-decoder skips 
        are skipped.

        Args:
        ----------
            dec_stage_ix (int):
                Index number signalling the current decoder stage
            in_channels (int):
                Number of channels in the upsampled decoder feature map
            dec_out_channels (List[int]):
                List of the number of output channels in all of the 
                decoder stages. Order is bottom up. Includes the number 
                of out channels at the last encoder (bottleneck) stage 
                as the first element in the list. 
                e.g. [2048, 256, 128, 64, 32, 16]
            enc_out_channels (List[int]):
                List of the number of channels in each of the encoder
                stages. Order is bottom up. This list does not include
                the final bottleneck stage out channels since it is 
                included in `dec_out_channels`. Also, the last element 
                of the list is zero to avoid a skip at the final decoder
                stage. e.g. [1024, 512, 256, 64, 0] 
            dec_out_dims (List[int]):
                List of the spatial dimensions (H, W) of each encoder 
                stage outptut. e.g. [16, 32, 64, 128, 256]. Order is
                bottom up. Assumption is that feature maps are square 
                shaped i.e. H == W
            short_skip (str, default=None):
                The type of short skip connection applied in the conv
                blocks of this module. One of ("residual", "dense", None)
            batch_norm (str, default="bn"): 
                Normalization method. One of "bn", "bcn", None
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
            merge_policy (str):
                One of: "summation", "concatenate" 
            lite_version (bool, default=False):
                If True, the dense decoder-to-decoder skips are not
                utilized at all. Reduces the model params quite a lot.      
        """
        if lite_version and batch_norm == "bcn" and \
            merge_policy == "concatenate":
            raise ValueError(
                "lite version does not support bcn norm when merge policy",
                "is set to 'concatenate'"
            )
        
        super(Unet3pSkip, self).__init__()
        self.batch_norm = batch_norm
        self.act = activation
        self.weight_standardize = weight_standardize
        self.preactivate = preactivate
        self.n_conv_blocks = n_conv_blocks
        self.lite_version = lite_version
        self.merge_policy = merge_policy
        self.dec_stage_ix = dec_stage_ix

        self.MultiConv = MultiBlockBasic
        if short_skip == "residual":
            self.MultiConv = MultiBlockResidual
        elif short_skip == "dense":
            self.MultiConv = MultiBlockDense

        # spatial dims (H, W) at prev decoder stages below this stage
        prev_dec_out_dims = dec_out_dims[:dec_stage_ix]
        if prev_dec_out_dims:
            # insert the encoder bottleneck dims to align
            prev_dec_out_dims.insert(0, prev_dec_out_dims[0])

        # num of channels at prev decoder stages below this stage
        prev_dec_out_chls = dec_out_channels[:dec_stage_ix] 

        # skip channels from the encoder 
        skip_channels = enc_out_channels[dec_stage_ix:-1] 
        dec_out_dims = dec_out_dims[dec_stage_ix:]

        # current out dim & chls
        self._out_channels = dec_out_channels[dec_stage_ix + 1]
        current_dec_out_dim = dec_out_dims[0]

        # Compute the output channels for the skip conv blocks
        cat_channels, num_out_feats = self._get_conv_chls(
            self._out_channels,
            enc_out_channels,
            skip_channels
        )

        # The number of skip chls depends on the merging policy
        skip_out_chl = num_out_feats
        if merge_policy == "concatenate":
            skip_out_chl = cat_channels


        # convblock for the raw upsampled incoming feat map
        self.decoder_feat_conv = self.MultiConv(
            in_channels=in_channels, 
            out_channels=num_out_feats,
            n_blocks=n_conv_blocks,
            batch_norm=batch_norm, 
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=preactivate
        )

        # if there are skip channels (I.e. this is not the last decoder 
        # stage), init all the encoder-to-decoder skip convs
        if skip_channels:
            target_size = current_dec_out_dim
            self.down_scales, self.enc2dec_convs = self._get_ops(
                from_to="enc2dec",
                target_size=target_size,
                out_dims=dec_out_dims,
                in_chls=skip_channels,
                out_chls=skip_out_chl
            )

            # get the decoder2decoder skips if not lite_version
            if not self.lite_version:
                self.up_scales, self.dec2dec_convs = self._get_ops(
                    from_to="dec2dec",
                    target_size=target_size,
                    out_dims=prev_dec_out_dims,
                    in_chls=prev_dec_out_chls,
                    out_chls=skip_out_chl
                )

    @property
    def out_channels(self) -> int:
        return self._out_channels

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

    def _get_conv_chls(
            self,
            current_dec_out_chl: int,
            enc_out_channels: List[int],
            skip_channels: List[int]
        ) -> Tuple[int, int]:
        """
        Convolution channel arithmetic to get the right number of chls
        for the convolutions in this module. If merge policy is to cat
        all the channels, the number of out channels is divided evenly
        such that it matches the decoder output channels. If the merge
        policy is to sum, then no need to divide.

        Args:
        ---------
            current_dec_out_chl (int):
                The number of output channels from the current decoder
                stage.
            enc_out_channels (List[int]):
                List of the number of channels for each encoder output
            skip_channels (List[int]):
                List of the number of channels for each skip convolution
                applied at the current decoder stage. 

        Returns:
        ---------
            Tuple[int, int]: The number of channels for the skip conv 
                             blocks and the number of channels for the
                             conv block used for the upsampled feat map.

        """
        cat_channels = None
        num_out_feats = current_dec_out_chl
        
        if self.merge_policy == "concatenate":
            # divide the number of out channels for each of the skip 
            # conv blocks to be even.
            divider = len(enc_out_channels)
            if self.lite_version:
                divider = len(skip_channels) + 1

            cat_channels, reminder = divmod(
                current_dec_out_chl, divider
            )

            # The reminder will be added to the decoder-to-decoder conv 
            # block such that the out chls of this module match the 
            # current ``current_dec_out_chl``. At the final decoder
            # stage ther is no skip connection -> no need to divide the 
            # out channels evenly
            if skip_channels:
                num_out_feats = cat_channels + reminder

        return cat_channels, num_out_feats

    def _get_ops(
            self,
            from_to: str,
            target_size: int,
            out_dims: List[int],
            in_chls: List[int],
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
            out_dims (List[int]):
                List of the output spatial dimensions of the 
            in_chls (List[int]):
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

        for i, (in_, dim_) in enumerate(zip(in_chls, out_dims)):
            scale_op = self.scale(dim_, target_size)
            scales[f"{from_to}_scale{i + 1}"] = scale_op
            convs[f"{from_to}_conv{i + 1}"] = self.MultiConv(
                in_channels=in_, 
                out_channels=out_chls,
                n_blocks=self.n_conv_blocks,
                batch_norm=self.batch_norm, 
                activation=self.act,
                weight_standardize=self.weight_standardize,
                preactivate=self.preactivate
            )

        return scales, convs

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
            x = torch.stack(features, dim=0).sum(dim=0)

        return x

    def forward(
            self,
            x: torch.Tensor,
            ix: int,
            skips: Tuple[torch.Tensor],
            extra_skips: List[torch.Tensor]=None,
            **kwargs
        ) -> torch.Tensor:
        """
        Args:
        ------------
            x (torch.Tensor):
                Upsampled input tensor from the previous decoder layer
            ix (int):
                Index for the the feature map from the encoder
            skips (Tuple[torch.Tensor]):
                all the feature maps from the encoder
            extra_skips (List[torch.Tensor], default=None):
                extra skip connections. Here, the deocder-to-decoder
                skip connections

        Returns:
        ------------
            Tuple of tensors: First return is the decoder branch output.
            The second return value are the outputs from the previous 
            decoder branches
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

            all_features = skip_features + decoder_features
            all_features.append(x)
            x = self._merge(all_features)
            
            # Add to prev decoder feature list
            if not self.lite_version:
                decoder_out_features.append(x)

        return x, decoder_out_features