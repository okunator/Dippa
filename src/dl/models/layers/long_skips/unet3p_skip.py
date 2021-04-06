import torch
import torch.nn as nn
from typing import Tuple, List
from src.dl.models.decoders import MultiBlockBasic


class Unet3pSkipBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 skip_channels: List[int],
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

        This includes only the skips from the encoder to decoder.
        Not the decoder to decoder skips 

        Args:
        ----------
            in_channels (int):
                Number of channels in the upsampled decoder feature map
            out_channels (int):
                Number of output channels in the decoder block
            skip_channels (List[int]):
                List of the number of channels in the each of the encoder skip tensors.
            out_dims (List[int]):
                List of the heights/widths of each encoder/decoder feature map
                e.g. [256, 128, 64, 32, 16]. Again, assumption is that feature maps are
                square shapes like in the target_size argument.
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
        super(Unet3pSkipBlock, self).__init__()

        out_dims = out_dims[:-1]
        skip_channels = skip_channels[:-1]

        if skip_channels:
            cat_channels, reminder = divmod(out_channels, (len(skip_channels) + 1))
            target_size = out_dims[0]

            self.down_scales = nn.ModuleDict()
            self.skip_convs = nn.ModuleDict()
            self.decoder_feat_conv = MultiBlockBasic(
                in_channels=in_channels, 
                out_channels=cat_channels+reminder,
                n_blocks=n_conv_blocks,
                batch_norm=batch_norm, 
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate
            )

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
        Can be up or downsampling or identity

        Args:
        ---------
            in_size (int):
                H/W of the input tensor
            target_size (int):
                H/W of the output tesnor

        Returns:
        ---------
            nn.MaxPool2d or nn.identity if scaling not needed
        """
        scale_factor = in_size // target_size
        
        if scale_factor > 1:
            scale_op = nn.MaxPool2d(kernel_size=scale_factor, ceil_mode=True)
        else:
            scale_op = nn.Identity()

        return scale_op

    def forward(self, x: torch.Tensor, skips: Tuple[torch.Tensor], idx: int) -> torch.Tensor:
        """
        Args:
        ------------
            x (torch.Tensor):
                input from the previous decoder layer
            skips (Tuple[torch.Tensor]):
                all the features from the encoder
            idx (int):
                index for the for the feature from the encoder
        """
        if idx < len(skips):
            skips = skips[idx:]
            decoder_feat = self.decoder_feat_conv(x)

            skip_features = []
            for i, (scale, conv_block) in enumerate(zip(self.down_scales.values(), self.skip_convs.values())):
                skip_feat = scale(skips[i])
                skip_feat = conv_block(skip_feat)
                skip_features.append(skip_feat)

            skip_features.append(decoder_feat)
            x = torch.cat(skip_features, dim=1)

        return x