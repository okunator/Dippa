from src.dl.losses.mse.loss import MSE
from src.dl.losses.grad_mse.loss import GradMSE
from src.dl.losses.ssim.loss import SSIM, MSSSIM
from src.dl.losses.dice.loss import DiceLoss
from src.dl.losses.iou.loss import IoULoss
from src.dl.losses.tversky.loss import TverskyLoss 
from src.dl.losses.focal.loss import WeightedFocalLoss
from src.dl.losses.ce.loss import WeightedCELoss
from src.dl.losses.sce.loss import WeightedSCELoss

LOSS_LOOKUP = {
    "iou": "IoULoss",
    "dice": "DiceLoss",
    "tversky": "TverskyLoss",
    "ce": "WeightedCELoss",
    "sce": "WeightedSCELoss",
    "focal": "WeightedFocalLoss",
    "mse": "MSE",
    "gmse": "GradMSE",
    "ssim": "SSIM",
    "msssim": "MSSSIM"
}

JOINT_SEG_LOSSES = [
    "iou",
    "dice",
    "tversky",
    "ce",
    "sce",
    "focal",
    "iou_ce",
    "iou_sce",
    "iou_focal",
    "dice_ce",
    "dice_sce",
    "dice_focal",
    "tversky_ce",
    "tversky_sce",
    "tversky_focal",
    "iou_ce_ssim",
    "iou_sce_ssim",
    "iou_focal_ssim",
    "dice_ce_ssim",
    "dice_sce_ssim",
    "dice_focal_ssim",
    "tversky_ce_ssim",
    "tversky_sce_ssim",
    "tversky_focal_ssim",
    "iou_ce_msssim",
    "iou_sce_msssim",
    "iou_focal_msssim",
    "dice_ce_msssim",
    "dice_sce_msssim",
    "dice_focal_msssim",
    "tversky_ce_msssim",
    "tversky_sce_msssim",
    "tversky_focal_msssim",
]

JOINT_AUX_LOSSES = [
    "mse",
    "gmse",
    "ssim",
    "msssim",
    "mse_ssim",
    "mse_gmse",
    "mse_msssim",
    "gmse_ssim",
    "gmse_msssim",
    "ssim_msssim",
    "mse_gmse_ssim",
    "mse_gmse_msssim"
]
