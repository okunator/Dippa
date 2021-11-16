from itertools import permutations
from .criterions import SEG_LOSS_LOOKUP, REG_LOSS_LOOKUP

from ._base._joint_loss import JointLoss
from ._base._multitask_loss import MultiTaskLoss



JOINT_SEG_LOSSES = []
for i in range(1, 5):
    JOINT_SEG_LOSSES.extend(
        ["_".join(t) for t in permutations(SEG_LOSS_LOOKUP.keys(), i)]
    )
    
JOINT_AUX_LOSSES = []
for i in range(1, 5):
    JOINT_AUX_LOSSES.extend(
        ["_".join(t) for t in permutations(REG_LOSS_LOOKUP.keys(), i)]
    )


__all__ = [
    "JOINT_AUX_LOSSES", "JOINT_SEG_LOSSES", "JointLoss", "MultiTaskLoss"
]