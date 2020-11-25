import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from src.dl.models.modules import Mish

# Implementation follows same class structure as in https://github.com/qubvel/segmentation_models.pytorch
# so that the pre-trained encoders can be used


class Unet3pConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding: bool = True,
                 batch_norm: bool = True) -> None:
        """
        Operations done for every unet block

        Args:
            in_channels (int): number of channels in the input layer (feature map)
            out_channels (int): number of channels in the output layer (feature map)
            padding (bool): if True, perform padding to the input image such that the input
                            shape is the same as the output.
            batch_norm (bool): if True perform batch normalization after each activation
        """
        super(Unet3pConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=int(padding)))

        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))

        # block.append(nn.ReLU(inplace=True))
        block.append(Mish(inplace=False))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)



class Unet3pEncoderSkipBlock(nn.ModuleDict):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int = 64,
                 padding: bool = True,
                 batch_norm: bool = True,
                 **kwargs) -> None:
        """
        All the operations involved in one Unet3+ for every encoder skip connection
        UNET3+
        https://arxiv.org/pdf/2004.08790.pdf

        (max pooling replaced with center cropping for speed/memory)
        (bilinear interpolation in upsampling replaced with nearest neighbor upsampling)

        Args:
            in_channels (List[int]): List of the number of input channels in each feature map
            out_channels int: number of output channels. 64 used to end up with the 320 for each block like in the paper
            padding (bool): do padding
            batch_norm (bool): do batch normalization
        """
        super(Unet3pEncoderSkipBlock, self).__init__()

        for i in range(len(in_channels)):
            conv_layer = Unet3pConvBlock(in_channels[i], out_channels, padding, batch_norm)
            self.add_module('conv%d' % (i + 1), conv_layer)


    def center_crop(self, features: List[torch.Tensor], target_size: Tuple[int]) -> List[torch.Tensor]:
        """
        center crop the encoder side skip connections to target_size

        Args:
            features (List[torch.Tensor]): the encoder side features that will be cat to fused vector
            target_size (Tuple[int]): Tuple of the spatial dims (H, W) of the features
        """

        cropped_features = []
        for f in features:
            _, _, H, W = f.size()

            # dont crop smaller features than decoder block
            if H >= target_size[0]:
                diff_y = (H - target_size[0]) // 2
                diff_x = (W - target_size[1]) // 2
                cropped_features.append(
                    f[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])])

        return cropped_features

    def forward(self, encoder_skips: List[torch.Tensor], target_size: Tuple[int]) -> List[torch.Tensor]:
        """
        Args:
            encoder_skips (List[torch.Tensor]): List of the encoder skip connection tensors
        """
        encoder_skips = self.center_crop(encoder_skips, target_size)
        features = []
        for i, (name, layer) in enumerate(self.items()):
            new_features = layer(encoder_skips[i])
            features.append(new_features)
        return features


class Unet3pMergeBlock(nn.Module):
    def __init__(self, merge_policy: str = "cat") -> None:
        """
        Merges input features

        Args:
            merge_policy (str): One of ('cat', 'add')
        """
        assert merge_policy in ("cat", "add"), f"merge_policy not in {('cat', 'add')}"
        super(Unet3pMergeBlock, self).__init__()
        self.merge_policy = merge_policy

    def forward(self, *features: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.merge_policy == "cat":
            out = torch.cat(*features, dim=1)
        elif self.merge_policy == "add":
            out = torch.sum(features)

        return out


class DenseLayer(nn.Module):
    def __init__(self,                 
                 in_channels: int,
                 out_channels: int,
                 padding: bool = True,
                 batch_norm: bool = True) -> None:
        """
        Operations done for every dense upsample block

        Args:
            in_channels (int): number of channels in the input layer (feature map)
            out_channels (int): number of channels in the output layer (feature map)
            padding (bool): if True, perform padding to the input image such that the input
                            shape is the same as the output.
            batch_norm (bool): if True perform batch normalization after each activation
        """
        super(DenseLayer, self).__init__()
        self.conv1 = Unet3pConvBlock(in_channels, out_channels, padding, batch_norm)

    def out_function(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        current_block = F.interpolate(inputs[-1], scale_factor=2, mode="nearest")
        current_block = [self.conv1(current_block)]
        
        if len(inputs) > 1:
            target_size = current_block[0].shape[2]
            prev_block = inputs[0]
            prev_block = [F.interpolate(prev_block, scale_factor=target_size//prev_block.shape[2], mode="nearest")]
        else:
            prev_block = []

        features = current_block + prev_block
        concated_features = torch.cat(features, dim=1)
        
        return concated_features
        
    def forward(self, features: torch.Tensor) -> torch.Tensor: 
        if isinstance(features, torch.Tensor):
            prev_features = [features]
        else:
            prev_features = features
        new_features = self.out_function(prev_features)
        return new_features



class Unet3pDecoder(nn.ModuleDict):
    def __init__(self,
                 encoder_channels: List[int],
                 n_blocks: int = 5,
                 cat_channels: int = 64,
                 merge_policy: str = "cat",
                 **kwargs) -> None:
        """
        UNET3+ decoder part
        https://arxiv.org/pdf/2004.08790.pdf

        Args:
            encoder_channels (List[int]): List of the number of channels in each encoder block
            n_blocks (int): number of decoder blocks
            cat_channels (int): number of output channels from convolutions operations 
            merge_policy (str): One of ("cat", "add")
        """
        super(Unet3pDecoder, self).__init__()
        decoder_channels = [encoder_channels[-1]] # first decoder channel is the head with 2048 channels
        encoder_channels = encoder_channels[:-1] # (64, 256, 512, 1024)
        encoder_channels = encoder_channels[::-1]
        cat_channels = cat_channels # 32
        cat_blocks = len(encoder_channels) + len(decoder_channels)  # 5
        up_channels = cat_channels * cat_blocks  # = 32*5 = 160

        encoder_skip_blocks = []
        for i in range(len(encoder_channels)):
            encoder_skip_blocks.append(
                Unet3pEncoderSkipBlock(
                    in_channels=encoder_channels[i:], out_channels=cat_channels)
            )

        decoder_dense_blocks = []
        for i in range(n_blocks):
            decoder_dense_blocks.append(
                DenseLayer(
                    in_channels=decoder_channels[i], out_channels=cat_channels)
                )
            decoder_channels.append(up_channels)

        self.encoder_skip_blocks = nn.ModuleList(encoder_skip_blocks)
        self.decoder_dense_blocks = nn.ModuleList(decoder_dense_blocks)
        self.merge_block = Unet3pMergeBlock(merge_policy)
        self.fuse_block = Unet3pConvBlock(up_channels, up_channels)

    def forward(self, *features: Tuple[torch.Tensor]) -> torch.Tensor:
        head = features[-1]
        features = features[:-1] # drop the head channel
        encoder_skips = features[::-1] # (3, 64, 246, 512, 1024)
    
        x0 = head
        for i, (encoder_skip_block, decoder_dense_block) in enumerate(zip(self.encoder_skip_blocks, self.decoder_dense_blocks)):
            x0 = decoder_dense_block(x0)
            encoder_features = encoder_skip_block(encoder_skips[i:], target_size = x0.shape[2:])
            features = encoder_features + [x0]
            x1 = self.merge_block(features) # (B, up_channels, x0.shape[2], x0.shape[3])
            x1 = self.fuse_block(x1) # (B, up_channels, x0.shape[2], x0.shape[3])
            x0 = [x0, x1]

        return x1