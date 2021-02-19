import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union

import src.dl.torch_utils as util
# import src.dl.inference.tta as tta
 

class Predictor:
    def __init__(self, model: nn.Module) -> None:
        """
        Helper class for predicting soft masks at inference time

        Args;
            model (nn.Module): nn.Module pytorch model
        """
        self.model = model


    def forward_pass(self, patch: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Input an image patch or batch of patches to the network and return logits.
        patch input_size likely to be 256x256 depending how the images were patched

        Args:
            patch (np.ndarray or torch.Tensor): 
                Image patch of shape (input_size, input_size, 3)
                or (B, input_size, input_size, 3)

        Returns:
            A dictionary {"instances":Tensor, "types":Union[Tensor, None], "aux":Union[Tensor, None]}
        """
        if isinstance(patch, np.ndarray):
            patch = util.ndarray_to_tensor(patch)  # (B, 3, H, W) | (1, 3, H, W)

        input_tensor = util.to_device(patch) # to cpu|gpu
        return self.model(input_tensor)  # Dict[(B, 2, H, W), (B, C, H, W)]

    def classify(self, 
                 patch: torch.Tensor, 
                 act: Union[str, None]="softmax", 
                 squeeze: bool=False,
                 return_type: str="numpy") -> np.ndarray:
        """
        Take in a patch or a batch of patches of logits produced by the model and
        use sigmoid activation for instance logits and softmax for semantic logits
        and convert the output to numpy nd.array.

        Args:
            patch (torch.Tensor): 
                a tensor of logits produced by the network.
                Shape: (B, C, input_size, input_size)
            act (str or None, default="softmax"):
                activation to be used. One of ("sigmoid", "softmax", None)
            squeeze (bool): 
                whether to squeeze the output batch dim if B (batch dim) = 1
                (B, C, input_size, input_size) -> (C, input_size, input_size) if B == 1
            return_type:
                One of ("torch", "numpy")
        Returns:
            np.ndarray of the prediction
        """
        assert act in ("sigmoid", "softmax", None)
        assert return_type in ("torch", "numpy")
        
        if act == "sigmoid":
            pred = torch.sigmoid(patch)
        elif act == "softmax":
            pred = F.softmax(patch, dim=1)
        else:
            pred = patch

        if return_type == "numpy":
            pred = util.tensor_to_ndarray(pred, squeeze=squeeze)
        else:
            # from gpu to cpu
            pred = pred.detach()
            if pred.is_cuda:
                pred = pred.cpu()

        return pred 

    # TODO: DOES NOT WORK CURRENTLY! FIX! TORCH IMPLEMENTATION PREFERRED
    def tta_classify(self, patch: np.ndarray, branch_key: str = "instances") -> np.ndarray:
        """
        Tta ensemble prediction pipeline.
    
        Args:
            patch (np.ndarray): 
                The img patch used for ensemble prediction. 
                shape (input_size, input_size, 3)
            branch_key (str): 
                Specifies which output branch

        Returns:
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