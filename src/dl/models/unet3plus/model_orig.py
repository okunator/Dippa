# ported from https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/UNet_3Plus.py

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dl.models.unet3plus.layers_orig import unetConv2
from src.dl.models.init_weights import init_weights


class UNet_3PLus_Encoder(nn.Module):
    def __init__(self,
                in_channels: int = 3,
                is_deconv: bool = True, 
                is_batchnorm: bool =True,
                **kwargs) -> None:
        """
        UNET3+ encoder
        https://arxiv.org/pdf/2004.08790.pdf

        Args:
            lflf
        """
        super().__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]

        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, inputs):
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024
        
        return {
            "h1":h1,
            "h2":h2,
            "h3":h3,
            "h4":h4,
            "h5":h5,
            "hd5":hd5
        }


class UNet_3PLus_Decoder(nn.Module):
    def __init__(self,
                 classes: int = 2,
                 ** kwargs) -> None:
        """
        UNET3+ decoder
        https://arxiv.org/pdf/2004.08790.pdf

        Args:
            lfkorekf
        """
        super().__init__()
        filters = [64, 128, 256, 512, 1024]
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.outconv1 = nn.Conv2d(self.UpChannels, classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self,
                h1: torch.Tensor,
                h2: torch.Tensor,
                h3: torch.Tensor,
                h4: torch.Tensor,
                h5: torch.Tensor,
                hd5: torch.Tensor) -> torch.Tensor:
        
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # 256
        return d1



# class UNetConvBlock(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  padding: bool = True,
#                  batch_norm: bool = True) -> None:
#         """
#         Operations done for every unet block

#         Args:
#             in_channels (int): number of channels in the input layer (feature map)
#             out_channels (int): number of channels in the output layer (feature map)
#             padding (bool): if True, perform padding to the input image such that the input
#                             shape is the same as the output.
#             batch_norm (bool): if True perform batch normalization after each activation
#         """
#         super(UNetConvBlock, self).__init__()
#         block = []

#         block.append(nn.Conv2d(in_channels, out_channels,
#                                kernel_size=3, padding=int(padding)))

#         if batch_norm:
#             block.append(nn.BatchNorm2d(out_channels))

#         block.append(nn.ReLU(inplace=True))
#         self.block = nn.Sequential(*block)

#     def forward(self, x):
#         return self.block(x)


# class Unet3pDecoderBlock(nn.Module):
#     def __init__(self,
#                  in_channels: List[int],
#                  up_channels: int = 320,
#                  out_channels: int = 64,
#                  padding: bool = True,
#                  batch_norm: bool = True,
#                  **kwargs) -> None:
#         """
#         All the operations involved in one Unet3+ decoder block
#         UNET3+ decoder block
#         https://arxiv.org/pdf/2004.08790.pdf

#         (max pooling replaced with center cropping for speed/memory)
#         (bilinear interpolation in upsampling replaced with nearest neighbor upsampling)

#         Args:
#             in_channels (List[int]): List of the number of input channels in each feature map
#             up_channels (int): number of channels for each decoder block. 320 as default 
#             out_channels int: number of output channels. 64 used to end up with the 320 for each block like in the paper
#             padding (bool): do padding
#             batch_norm (bool): do batch normalization
#         """
#         super().__init__()

#         # current decoder block operation for x
#         self.conv0 = UNetConvBlock(
#             in_channels[-1],
#             out_channels,
#             padding,
#             batch_norm
#         )

#         # operation for skip1
#         self.conv1 = UNetConvBlock(
#             in_channels[0],
#             out_channels,
#             padding,
#             batch_norm
#         )

#         # operation for skip2
#         self.conv2 = UNetConvBlock(
#             in_channels[1],
#             out_channels,
#             padding,
#             batch_norm
#         )

#         # operation for skip3
#         self.conv3 = UNetConvBlock(
#             in_channels[2],
#             out_channels,
#             padding,
#             batch_norm
#         )

#         # operation for skip4
#         self.conv4 = UNetConvBlock(
#             in_channels[3],
#             out_channels,
#             padding,
#             batch_norm
#         )

#         # TODO: Add attention block
#         # after cat
#         self.conv5 = UNetConvBlock(
#             up_channels,
#             up_channels,
#             padding,
#             batch_norm
#         )

#         # TODO: Add attention block

#     def center_crop(self, features: List[torch.Tensor], target_size: Tuple[int]) -> List[torch.Tensor]:
#         """
#         center crop the encoder side skip connections to target_size

#         Args:
#             features (List[torch.Tensor]): the encoder side features that will be cat to fused vector
#             target_size (Tuple[int]): Tuple of the spatial dims (H, W) of the features
#         """

#         cropped_features = []
#         for f in features:
#             _, _, H, W = f.size()

#             # dont crop smaller features than decoder block
#             if H >= target_size[0]:
#                 diff_y = (H - target_size[0]) // 2
#                 diff_x = (W - target_size[1]) // 2
#                 cropped_features.append(
#                     f[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])])

#         return cropped_features

#     def up(self, features: List[torch.Tensor], target_size: Tuple[int]) -> List[torch.Tensor]:
#         """        features = features[1:] # first feature only 3 channels
#         head = features[-1]
#         encoder_skips = list(features[:-1])
#         decoder_skips = [head]
#         Upsample all the decoder skip connections to target_size

#         Args:
#             features (List[torch.Tensor]): the decoder side features that will be cat to fused vector
#             target_size (Tuple[int]): Tuple of the spatial dims (H, W) of the features
#         """
#         upped_features = [F.interpolate(x, scale_factor=target_size[0]//x.shape[2], mode="nearest") for x in features]
#         return upped_features

#     def forward(self,
#                 x: torch.Tensor,
#                 encoder_skips: List[torch.Tensor],
#                 decoder_skips: List[torch.Tensor]) -> torch.Tensor:
#         """
#         Forward pass of unet3p decoder block

#         Args:
#             x (torch.Tensor): Current decoder block
#             encoder_skips (List[torch.Tensor]): List of the encoder skip connection tensors
#             decoder_skips (List[torch.Tensor]): List of the decoder skip connection tensors
#         """
#         # decoder block
#         d = F.interpolate(x, scale_factor=2, mode="nearest")
#         d = self.conv0(d)

#         # fully scaled skips
#         encoder_skips = self.center_crop(encoder_skips, d.shape[2:])
#         decoder_skips = self.up(decoder_skips, d.shape[2:])
#         skips = encoder_skips + decoder_skips

#         x1 = self.conv1(skips[0])
#         x2 = self.conv2(skips[1])
#         x3 = self.conv3(skips[2])
#         x4 = self.conv4(skips[3])

#         # Fuse
#         out = torch.cat([x1, x2, x3, x4, d], dim=1)
#         # TODO: self.attention1(d)
#         out = self.conv5(out)  # 1st it: in_channels = encoder.out_channels[-1]
#         # TODO: self.attention2(d)
#         return out


# class UNet3pDecoder(nn.Module):
#     def __init__(self,
#                  encoder_channels: List[int],
#                  n_blocks: int = 5,
#                  cat_channels: int = 32,
#                  **kwargs) -> None:
#         """
#         UNET3+ decoder part
#         https://arxiv.org/pdf/2004.08790.pdf

#         Args:
#             encoder_channels (List[int]): List of the number of channels in each encoder block
#             n_blocks (int): number of decoder blocks
#             cat_channels (int): number of output channels from convolutions operations 
#         """
#         super().__init__()
#         head_channels = [encoder_channels[-1]]  # 2048
#         encoder_channels = list(encoder_channels[1:-1])
#         cat_channels = cat_channels # 32
#         cat_blocks = len(encoder_channels) + len(head_channels)  # 5
#         up_channels = cat_channels * cat_blocks  # = 32*5 = 160

#         decoder_blocks = []
#         decoder_channels = head_channels
#         for _ in range(n_blocks):
#             channels = encoder_channels + decoder_channels
#             decoder_blocks.append(
#                 Unet3pDecoderBlock(
#                     in_channels=channels, out_channels=cat_channels, up_channels=up_channels)
#             )
#             if encoder_channels:
#                 encoder_channels.pop()
#                 decoder_channels.append(up_channels)

#         self.center = nn.Identity()
#         self.blocks = nn.ModuleList(decoder_blocks)

#     def forward(self, *features: Tuple[torch.Tensor]) -> torch.Tensor:
#         features = features[1:] # first feature only 3 channels
#         head = features[-1]
#         encoder_skips = list(features[:-1])
#         decoder_skips = [head]

#         x = self.center(head)
#         for i, decoder_block in enumerate(self.blocks):
#             x = decoder_block(x, encoder_skips[i:], decoder_skips)
#             decoder_skips.append(x)

#             x = decoder_block(x, encoder_skips, decoder_skips)
#             if encoder_skips:
#                 encoder_skips.pop()
#                 decoder_skips.append(x)

#         # del encoder_skips
#         # del decoder_skips

#         return x

