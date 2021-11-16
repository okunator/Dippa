from .bcn import EstBN, BCNorm
from .gn import GroupNorm

from torch.nn import (
    BatchNorm2d,
    InstanceNorm2d,
    SyncBatchNorm,
    LocalResponseNorm
)

NORM_LOOKUP = {
    "bn": "BatchNorm2d",
    "bcn": "BCNorm",
    "gn": "GroupNorm",
    "in": "InstanceNorm2d",
    "lrn": "LocalResponseNorm"
}

__all__ = [
    "NORM_LOOKUP", "BCNorm", "EstBN", "GroupNorm", "BatchNorm2d",
    "InstanceNorm2d", "SyncBatchNorm", "LocalResponseNorm"
]