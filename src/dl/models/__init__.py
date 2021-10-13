from .decoders import Decoder
from .base_model import MultiTaskSegModel
from .heads import SegHead
from .encoders import TimmUniversalEncoder


MODEL_LOOKUP = {
    "unet":"UnetSmpMulti",
    "unet3+":"Unet3pMulti",
    "unet++":"UnetPlusPlusSmpMulti",
    "pspnet":"PSPNetSmpMulti",
    "fpn":"FpnSmpMulti",
    "pan":"PanSmpMulti",
    "deeplabv3":"DeepLabV3SmpMulti",
    "deeplabv3+":"DeepLabV3PlusSmpMulti",
    "hovernet":"HoverNetMulti",
}