from src.dl.models.initialization import initialize_decoder, initialize_head
from src.dl.models.base_model import InstSegModel, InstSegModelWithClsBranch
from src.dl.models.linknet.model import LinknetSmp, LinknetSmpWithClsBranch
from src.dl.models.unet.model import UnetSmp, UnetSmpWithClsBranch
from src.dl.models.unet3plus.model import Unet3pInst, Unet3pWithClsBranch
from src.dl.models.unetplusplus.model import UnetPlusPlusSmp, UnetPlusPlusSmpWithClsBranch
from src.dl.models.pspnet.model import PSPNetSmp, PSPNetSmpWithClsBranch 
from src.dl.models.fpn.model import FpnSmp, FpnSmpWithClsBranch 
from src.dl.models.pan.model import PanSmp, PanSmpWithClsBranch 
from src.dl.models.deeplabv3.model import DeepLabV3Smp, DeepLabV3SmpWithClsBranch 
from src.dl.models.deeplabv3plus.model import DeepLabV3PlusSmp, DeepLabV3PlusSmpWithClsBranch

MODEL_NAIVE_LOOKUP = {
    "unet":"UnetSmp",
    "unet3+":"Unet3p",
    "unet++":"UnetPlusPlusSmp",
    "pspnet":"PSPNetSmp",
    "fpn":"FpnSmp",
    "pan":"PanSmp",
    "deeplabv3":"DeepLabV3Smp",
    "deeplabv3+":"DeepLabV3PlusSmp",
    "hovernet":"HoverNet",
}

MODEL_LOOKUP = {
    "unet":"UnetSmpWithClsBranch",
    "unet3+":"Unet3pWithClsBranch",
    "unet++":"UnetPlusPlusSmpWithClsBranch",
    "pspnet":"PSPNetSmpWithClsBranch",
    "fpn":"FpnSmpWithClsBranch",
    "pan":"PanSmpWithClsBranch",
    "deeplabv3":"DeepLabV3SmpWithClsBranch",
    "deeplabv3+":"DeepLabV3PlusSmpWithClsBranch",
    "hovernet":"HoverNetWithClsBranch",
}