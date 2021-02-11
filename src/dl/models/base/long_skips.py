
import torch
import torch.nn as nn


class UnetSkipBlock(nn.Module):
    def __init__(self, merge_policy: str = "cat") -> None:
        """
        Simple U-net like skip connection block

        Args:
            merge_policy (str, default="cat"):
                sum or concatenate the features together
        """
        super(UnetSkipBlock, self).__init__()
        assert merge_policy in ("cat", "sum")
        self.merge_policy = merge_policy

    def forward(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.merge_policy == "cat":
            x = torch.cat([x, skip], dim=1)
        elif self.merge_policy == "sum":
            x += skip
        return x