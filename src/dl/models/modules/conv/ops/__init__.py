from .ws_conv import WSConv2d, WSConv2dStaticSamePadding

from torch.nn import Conv2d

CONV_LOOKUP = {
    "conv": "Conv2d",
    "wsconv": "WSConv2d"
}