from torch.nn import Conv2d

from .ws_conv import WSConv2d, WSConv2dStaticSamePadding

CONV_LOOKUP = {
    "conv": "Conv2d",
    "wsconv": "WSConv2d",
    "wsconv2": "WSConv2dStaticSamePadding"
}

__all__ = ["CONV_LOOKUP", "WSConv2d", "WSConv2dStaticSamePadding", "Conv2d"]