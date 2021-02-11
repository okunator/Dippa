from src.dl.models.initialization import initialize_decoder, initialize_head
from src.dl.models.base_model import MultiTaskSegModel
from src.dl.models.linknet.model import LinknetSmpMulti
from src.dl.models.unet.model import UnetSmpMulti
from src.dl.models.unet3plus.model import Unet3pMulti
from src.dl.models.unetplusplus.model import UnetPlusPlusSmpMulti
from src.dl.models.pspnet.model import PSPNetSmpMulti
from src.dl.models.fpn.model import FpnSmpMulti
from src.dl.models.pan.model import PanSmpMulti
from src.dl.models.deeplabv3.model import DeepLabV3SmpMulti
from src.dl.models.deeplabv3plus.model import DeepLabV3PlusSmpMulti

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