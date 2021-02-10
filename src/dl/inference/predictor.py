import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import src.dl.torch_utils as util
import src.img_processing.post_processing as post_proc
import src.img_processing.augmentations.test_time_augs as tta
from typing import Dict

# inheritance idea: projectfilemanager -> inferer
# predictor -> post-processor
# tilerStitcher
# benchmarker

class Predcictor:
    def __init__(self, model: nn.Module) -> None:
        """
        class to create predictions at inference time

        Args;
            model (nn.Module): nn.Module pytorch model
        """
        self.model = model

    def forward_pass(self, patch: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Input an image patch or batch of patches to the network and return logits.
        patch input_size likely to be 256x256 depending how the images were patched

        Args:
            patch (np.ndarray): 
                Image patch of shape (input_size, input_size, 3)
                or (B, input_size, input_size, 3)

        Returns:
            A dictionary {"instances":Tensor, "types":Union[Tensor, None], "aux":Union[Tensor, None]}
        """
        input_tensor = util.ndarray_to_tensor(patch)  # (B, 3, H, W) | (1, 3, H, W)
        input_tensor = util.to_device(input_tensor) # to cpu|gpu
        return self.model(input_tensor)  # Dict[(B, 2, H, W), (B, C, H, W)]

    def classify(self, patch: torch.Tensor, squeeze: bool = False) -> np.ndarray:
        """
        Take in a patch or a batch of patches of logits produced by the model and
        use sigmoid activation for instance logits and softmax for semantic logits
        and convert the output to numpy nd.array.

        Args:
            patch (torch.Tensor): 
                a tensor of logits produced by the network.
                Shape: (B, C, input_size, input_size)
            squeeze (bool): 
                whether to squeeze the output batch dim if B (batch dim) = 1
                (B, C, input_size, input_size) -> (C, input_size, input_size) if B == 1

        Returns:
            np.ndarray of the prediction
        """
        if patch.shape[1] == 2:
            pred = torch.sigmoid(patch)
        else:
            pred = F.softmax(patch, dim=1)

        return util.tensor_to_ndarray(pred, squeeze=squeeze)

    # TODO: DOES NOT WORK CURRENTLY! FIX!
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
        outputs = []
        
        # flip ttas
        for aug, deaug in zip(tta.tta_augs(), tta.tta_deaugs()):
            aug_input = aug(image = patch) # (H, W, 3)
            aug_logits = self.forward_pass(aug_input["image"]) # (1, C, H, W)

            for key, item in aug_logits.items():
                if item is not None or key is not "aux":
                    aug_output = self.classify(item, squeeze=True) # (H, W, C)
                    deaug_insts = deaug(image = aug_output) # (H, W, C)
                    outputs.append(deaug_insts["image"])
            
        # five crops tta
        scale_up = tta.resize(patch.shape[0], patch.shape[1])
        scale_down = tta.resize(patch.shape[0]//2, patch.shape[1]//2)
        out_insts = np.zeros((patch.shape[0], patch.shape[1], 2))
        for crop in tta.tta_five_crops(patch):
            cropped_im = crop(image = patch)
            scaled_im = scale_up(image = cropped_im["image"]) # (H, W, C)
            aug_logits = self.forward_pass(scaled_im["image"])  # (1, C, H, W)
            aug_insts = self.classify(aug_logits["instances"], squeeze=True) # (H, W, C)
            downscaled_insts = scale_down(image = aug_insts) # (H//2, W//2, C)
            out_insts[crop.y_min:crop.y_max, crop.x_min:crop.x_max] = downscaled_insts["image"] # (H, W, C)
            outputs.append(out_insts)

            
        # take the mean of all the predictions
        return {
            "instances":np.asarray(outputs).mean(axis=0),
            "types": np.asarray(soft_types).mean(axis=0),
            "aux": aux
        }