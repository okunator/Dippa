import torch
import torch.nn as nn


class GroupNorm(nn.GroupNorm):
    def __init__(self,
                 num_features: int,
                 num_groups: int=None,
                 **kwargs) -> None:
        """
        Convenience wrapper for nn.GroupNorm to make kwargs
        compatible with nn.BatchNorm

        Infers the num_groups from the num_features to avoid
        errors. By default: uses 16 channels per group. 
        If channels <= 16, squashes to layer norm

        magic number 16 comes from the paper: https://arxiv.org/abs/1803.08494

        Args:
        ----------
            num_features (int):
                Number of input channels/features
            num_groups (int, default=None):
                Number of groups to group the channels.
                Typically n_features is a multiple of 32
        """

        num_groups, remainder = divmod(num_features, 16)
        if remainder:
            num_groups = num_features // remainder

        super(GroupNorm, self).__init__(
            num_groups=num_groups,
            num_channels=num_features
        )