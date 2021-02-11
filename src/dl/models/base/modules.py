import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class WSConv2d(nn.Conv2d):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 eps: float = 1e-7) -> None:
        """
        Weight standardized convolution. 
        From: https://github.com/joe-siyuan-qiao/WeightStandardization

        https://arxiv.org/abs/1903.10520

        Args: Refer to https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        """
        super(WSConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )
        self.eps = eps

    def forward(self, x: torch.Tensor):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )


class EstBN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-7) -> None:
        """
        Estimate of the batch statistics
        From: https://github.com/joe-siyuan-qiao/Batch-Channel-Normalization
        
        Args:
            num_features (int):
                Number of input channels/features
            eps (float, default=1e-7):
                small constant for numerical stability
        """
        super(EstBN, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('estbn_moving_speed', torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.estbn_moving_speed.item()
        if self.training:
            with torch.no_grad():
                xt = x.transpose(0, 1).contiguous().view(self.num_features, -1)
                running_mean = xt.mean(dim=1)
                xt -= self.running_mean.view(-1, 1)
                running_var = torch.mean(xt*xt, dim=1)
                self.running_mean.data.mul_(1 - ms).add_(ms*running_mean.data)
                self.running_var.data.mul_(1 - ms).add_(ms*running_var.data)

        out = x - self.running_mean.view(1, -1, 1, 1)
        out = out / torch.sqrt(self.running_var + self.eps).view(1, -1, 1, 1)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        out = weight * out + bias
        return out


class BCNorm(nn.Module):
    def __init__(self,
                 num_features: int, 
                 num_groups: int = 32,
                 eps: float = 1e-7,
                 estimate: bool = False) -> None:
        """
        Batch channel normalization
        From: https://github.com/joe-siyuan-qiao/Batch-Channel-Normalization

        https://arxiv.org/abs/1911.09738

        Args:
            num_features (int):
                Number of input channels/features
            num_groups (int, default=32):
                Number of groups to group the channels.
                Typically n_features is a multiple of 32
            eps (float, default=1e-7):
                small constant for numerical stability
            estimate (bool, default=False):
                If True, Uses EstBN.
                Refer to the article
        """
        super(BCNorm, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups 
        self.eps = eps
        self.weight = Parameter(torch.ones(1, num_groups, 1))
        self.bias = Parameter(torch.zeros(1, num_groups, 1))

        if estimate:
            self.bn = EstBN(num_features=num_features)
        else:
            self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor):
        out = self.bn(x)
        out = out.view(1, x.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.view(x.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.view_as(x)
        return out