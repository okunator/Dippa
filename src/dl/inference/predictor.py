import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Tuple

import src.dl.utils as util
# import src.dl.inference.tta as tta


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """
    Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary. This weight matrix is used for merging
    individual tile predictions and helps dealing with prediction artifacts on tile
    boundaries.

    Ported from:
    https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/inference/tiles.py

    Args:
    ----------
        width (int):
            Tile width
        height (int): 
            Tile height

    Returns:
    ----------
        np.ndarray since-channel image. Shape (H, W)
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W


class Predictor:
    def __init__(self, model: nn.Module, patch_size: Tuple[int]=(256, 256)) -> None:
        """
        Helper class for predicting soft masks at inference time
        
        Contains a weight matrix that can assign bigger weight on pixels
        in center and less weight to pixels on image boundary. helps dealing with
        prediction artifacts on tile boundaries.

        Args:
        ------------
            model (nn.Module):
                nn.Module pytorch model
        """
        self.model = model
        weight_mat = compute_pyramid_patch_weight_loss(patch_size[0], patch_size[1])
        self.weight_mat = torch.from_numpy(weight_mat).float().to(self.model.device).unsqueeze(0).unsqueeze(0)


    def forward_pass(self, 
                     patch: Union[np.ndarray, torch.Tensor],
                     norm: bool=False,
                     mean: np.ndarray=None,
                     std: np.ndarray=None) -> Dict[str, torch.Tensor]:
        """
        Input an image patch or batch of patches to the network and return logits.
        patch input_size likely to be 256x256 depending how the images were patched

        Args:
        -----------
            patch (np.ndarray or torch.Tensor): 
                Image patch of shape (input_size, input_size, 3)
                or (B, input_size, input_size, 3)
            norm (bool, default=False):
                Normalize input data. Set to True only if input data was
                normalized in the training phase.
            mean (np.ndarray): 
                Means for each channel. Shape (1, 3). Ignored id norm = False
            std (np.ndarray): 
                Standard deviations for each channel. Shape (1, 3). Ignored if norm = False

        Returns:
        -----------
            A dictionary {"instances":Tensor, "types":Union[Tensor, None], "aux":Union[Tensor, None]}
        """
        if isinstance(patch, np.ndarray):
            patch = util.ndarray_to_tensor(patch)  # (B, 3, H, W) | (1, 3, H, W)

        if norm:
            for i in range(patch.shape[0]):
                patch[i] = util.minmax_normalize_torch(patch[i])
            
        input_tensor = util.to_device(patch) # to cpu|gpu
        return self.model(input_tensor)  # Dict[(B, 2, H, W), (B, C, H, W)]

    def classify(self, 
                 patch: torch.Tensor, 
                 act: Union[str, None]="softmax", 
                 apply_weights: bool=False,
                 to_cpu: bool=False) -> np.ndarray:
        """
        Take in a patch or a batch of patches of logits produced by the model and
        use sigmoid activation for instance logits and softmax for semantic logits
        and convert the output to numpy nd.array.

        Args:
        -----------
            patch (torch.Tensor): 
                a tensor of logits produced by the network.
                Shape: (B, C, input_size, input_size)
            act (str or None, default="softmax"):
                activation to be used. One of ("sigmoid", "softmax", None)
            apply_weights (bool, default=True):
                apply a weight matrix that assigns bigger weight on pixels
                in center and less weight to pixels on image boundary. helps dealing with
                prediction artifacts on tile boundaries.
            to_cpu (str, default=False):
                Detach tensor from gpu to cpu. If batch_size is very large this can be used
                to avoid memory errors during inference.

        Returns:
        -----------
            np.ndarray of the prediction
        """
        assert act in ("sigmoid", "softmax", None)
        
        # apply classification activation
        if act == "sigmoid":
            pred = torch.sigmoid(patch)
        elif act == "softmax":
            pred = F.softmax(patch, dim=1)
        else:
            pred = patch

        # Add weights to pred matrix 
        if apply_weights:
            # work out the tensor shape first for the weight mat
            B, C = pred.shape[:2]
            W = torch.repeat_interleave(self.weight_mat, repeats=C, dim=1).repeat_interleave(repeats=B, dim=0)
            pred *= W

        # from gpu to cpu
        if to_cpu:
            pred = pred.detach()
            if pred.is_cuda:
                pred = pred.cpu()

        return pred 

    # TODO: DOES NOT WORK CURRENTLY! FIX! TORCH IMPLEMENTATION PREFERRED
    def tta_classify(self, patch: np.ndarray, branch_key: str = "instances") -> np.ndarray:
        """
        Tta ensemble prediction pipeline.
    
        Args:
        -----------
            patch (np.ndarray): 
                The img patch used for ensemble prediction. 
                shape (input_size, input_size, 3)
            branch_key (str): 
                Specifies which output branch

        Returns:
        ------------
            A dictionary {"instances":Tensor, "types":Union[Tensor, None], "aux":Union[Tensor, None]}
        
        Following instructions of 'beneficial augmentations' from:
        
        "Towards Principled Test-Time Augmentation"
        D. Shanmugam, D. Blalock, R. Sahoo, J. Guttag 
        https://dmshanmugam.github.io/pdfs/icml_2020_testaug.pdf
        
        1. vflip, hflip and transpose and rotations
        2. custom fivecrops aug
        3. take the mean of predictions
        """
        pass