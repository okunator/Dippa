import torch
import torch.nn as nn


# Adapted from https://github.com/vqdang/hover_net/blob/master/models/hovernet/net_utils.py
class FixedUnpool(nn.Module):
    def __init__(self, scale_factor: int = 2) -> None:
        """
        Upsample input by a scale factor.
        TensorPack fixed unpooling in pytorch

        Args:
        ------------
            scale_factor (int, default=2):
                Upsampling scale factor. scale_factor*(H, W) 
        """
        super(FixedUnpool, self).__init__()
        self.scale_factor = scale_factor
        self.register_buffer(
            "unpool_mat", torch.ones(
                [scale_factor, scale_factor], 
                dtype=torch.float32
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = list(x.shape)
        x = x.unsqueeze(-1)  # (B, C, H, W)
        mat = self.unpool_mat.unsqueeze(0)  # (B, C, H, W, 1)
        ret = torch.tensordot(x, mat, dims=1)  # (B, C, H, W, SH, SW)
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((
            -1, in_shape[1], in_shape[2]*self.scale_factor, 
            in_shape[3]*self.scale_factor)
        )

        return ret

    def __repr__(self):
        return f"FixedUnpool(scale_factor={self.scale_factor})"