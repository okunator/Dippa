from ._base._multitask_model import MultiTaskSegModel

# MODEL_LOOKUP = {
#     "unet":"UnetSmpMulti",
#     "unet3+":"Unet3pMulti",
#     "unet++":"UnetPlusPlusSmpMulti",
#     "pspnet":"PSPNetSmpMulti",
#     "fpn":"FpnSmpMulti",
#     "pan":"PanSmpMulti",
#     "deeplabv3":"DeepLabV3SmpMulti",
#     "deeplabv3+":"DeepLabV3PlusSmpMulti",
#     "hovernet":"HoverNetMulti",
# }

__all__ = ["MultiTaskSegModel"]