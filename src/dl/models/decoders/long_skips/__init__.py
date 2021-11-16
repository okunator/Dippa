from .unet_skip import UnetSkip
from .unet3p_skip import Unet3pSkip


SKIP_LOOKUP = {
    "unet": "UnetSkip",
    "unet3+": "Unet3pSkip"
}


__all__ = [
    "SKIP_LOOKUP", "UnetSkip", "Unet3pSkip"
]