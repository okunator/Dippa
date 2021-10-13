import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=1,
            dilation: int=1,
            groups: int=1,
            bias: bool=True,
            eps: float=1e-7,
            **kwargs
        ) -> None:
        """
        Weight standardized convolution. 
        From: https://github.com/joe-siyuan-qiao/WeightStandardization

        https://arxiv.org/abs/1903.10520

        Args: 
        --------
            Refer to torch Convolution2D
        """
        super(WSConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )
        self.eps = eps

    def forward(self, x: torch.Tensor):
        weight = self.weight
        
        weight_mean = weight.mean(
            dim=1, keepdim=True
        ).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        
        weight = weight - weight_mean
        std = weight.view(
            weight.size(0), -1
        ).std(dim=1).view(-1, 1, 1, 1) + self.eps

        weight = weight / std.expand_as(weight)
        
        return F.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )


class WSConv2dStaticSamePadding(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int=1,
            image_size: int=None,
            padding: int=1,
            dilation: int=1,
            groups: int=1,
            bias: bool=True,
            eps: float=1e-7,
            **kwargs
        ) -> None:
        """
        Tensorflow like convolution with "SAME" padding. padding 
        computed from the input image size in the constructor and used 
        in forward

        Adapted from:
        https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py


        Why?:
            Needed to make model builder etc compatible with the 
            efficient-net encoder that seems to perform better than 
            other encoders.

        Args:
        -----------
            Refer to: 
            https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        """
        assert image_size is not None

        super(WSConv2dStaticSamePadding, self).__init__(
            in_channels, out_channels, kernel_size, 
            stride, padding, dilation, groups, bias
        )

        self.eps = eps

        self.stride = [self.stride[0]] * 2
        if len(self.stride) == 2:
            self.stride = self.stride

        if isinstance(image_size, int):
            ih, iw = (image_size, image_size)
        else:
            ih, iw = image_size

        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max(
            (oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0
        )

        pad_w = max(
            (ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0
        )

        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.static_padding(x)
        weight = self.weight
        weight_mean = weight.mean(
            dim=1, keepdim=True
        ).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        weight = weight - weight_mean
        std = weight.view(
            weight.size(0), -1
        ).std(dim=1).view(-1, 1, 1, 1) + self.eps

        weight = weight / std.expand_as(weight)
        
        return F.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )