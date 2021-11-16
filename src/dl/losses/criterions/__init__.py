from .mse import MSE
from .grad_mse import GradMSE
from .ssim import SSIM, MSSSIM
from .dice import DiceLoss
from .iou import IoULoss
from .tversky import TverskyLoss
from .focal import FocalLoss
from .ce import CELoss
from .sce import SCELoss


SEG_LOSS_LOOKUP = {
    "iou": "IoULoss",
    "dice": "DiceLoss",
    "tversky": "TverskyLoss",
    "ce": "CELoss",
    "sce": "SCELoss",
    "focal": "FocalLoss",
    "mse": "MSE",
    "gmse": "GradMSE",
    "ssim": "SSIM",
    "msssim": "MSSSIM"
}


REG_LOSS_LOOKUP = {
    "ce": "CELoss",
    "sce": "SCELoss",
    "focal": "FocalLoss",
    "mse": "MSE",
    "gmse": "GradMSE",
    "ssim": "SSIM",
    "msssim": "MSSSIM"
}


__all__ = [
    "MSE", "GradMSE", "SSIM", "MSSSIM", "DiceLoss", "IoULoss", "TverskyLoss",
    "FocalLoss", "CELoss", "SCELoss", "SEG_LOSS_LOOKUP", "REG_LOSS_LOOKUP"
]