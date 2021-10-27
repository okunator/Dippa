from torch.nn.modules.batchnorm import BatchNorm1d
from .bcn import EstBN, BCNorm
from .gn import GroupNorm

from torch.nn import (
    BatchNorm2d,
    InstanceNorm2d,
    SyncBatchNorm,
    LayerNorm,
    LocalResponseNorm
)

NORM_LOOKUP = {
    "bn": "BatchNorm2d",
    "bcn": "BCNorm",
    "gn": "GroupNorm",
    "in": "InstanceNorm2d",
    # "ln": "LayerNorm",
    "lrn": "LocalResponseNorm"
}