import torch
import torch.nn as nn


class GroupNorm(nn.GroupNorm):
    def __init__(self,
                 num_features: int,
                 num_groups: int,
                 **kwargs) -> None:
        """
        Convenience wrapper for nn.GroupNorm to make kwargs
        compatible with nn.BatchNorm

        Args:
        ----------
            num_features (int):
                Number of input channels/features
            num_groups (int, default=32):
                Number of groups to group the channels.
                Typically n_features is a multiple of 32
        """
        super(GroupNorm, self).__init__(
            num_groups=num_groups,
            num_channels=num_features
        )