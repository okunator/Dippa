import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import src.dl.torch_utils as util
import src.img_processing.post_processing as post_proc
import src.img_processing.augmentations.test_time_augs as tta
from typing import Dict

# inheritance idea: projectfilemanager -> tiler -> predictor -> post-processor -> inferer

class Predcictor:
    def __init__(self, model: nn.Module) -> None:
        """
        class that provides helper methods to easily create predictions at inference

        Args;
            model (nn.Module): nn.Module pytorch model
        """
        self.model = model

    def get_logits(self, patch: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Input an image patch or batch of patches to the network and return logits.

        Args:
            patch (np.ndarray): image patch of shape (input_size, input_size, 3)
                or (B, input_size, input_size, 3)

        Returns:
            A dictionary {"instances":Tensor, "types":Union[Tensor, None], "aux":Union[Tensor, None]}
        """
        input_tensor = util.ndarray_to_tensor(patch)  # (1, 3, H, W)
        input_tensor = util.to_device(input_tensor) # to cpu|gpu
        return self.model(input_tensor)  # Dict[(B, 2, H, W), (B, C, H, W)]

    def gen_prediction(self, logits: torch.Tensor, squeeze: bool = False) -> np.ndarray:
        """
        Take in a patch or a batch of patches of logits produced by the model and
        use sigmoid activation for instance logits and softmax for semantic logits
        and convert the output to numpy nd.array.

        Args:
            logits (torch.Tensor): a tensor of logits produced by the network.
                Shape: (B, C, input_size, input_size)
            squeeze (bool): whether to squeeze the output batch if batch dim is 1

        Returns:
            np.ndarray of the result
        """
        if logits.shape[1] == 2:
            pred = torch.sigmoid(logits)
        else:
            pred = F.softmax(logits, dim=1)

        return util.tensor_to_ndarray(pred, squeeze=squeeze)

    def activate_plus_dog(self, patch: np.ndarray) -> np.ndarray:
        """
        Use activations and DoG to smoothen soft mask patch.

        Args:
            patch (np.ndarray): the soft mask pach to smoothen. Shape (H, W, C)
        """
        for c in range(patch.shape[2]):
            patch[..., c] = post_proc.activate_plus_dog(patch[..., c])  # (H, W)
        return patch
         
            
    def gen_ensemble_prediction(self, patch: np.ndarray) -> np.ndarray:
        """
        Tta ensemble prediction pipeline.
    
        Args:
            patch (np.ndarray): the img patch used for ensemble prediction. 
                   shape (input_size, input_size, 3)

        Returns:
            np.ndarray soft mask of shape (input_size, input_size, C)
        
        Following instructions of 'beneficial augmentations' from:
        
        "Towards Principled Test-Time Augmentation"
        D. Shanmugam, D. Blalock, R. Sahoo, J. Guttag 
        https://dmshanmugam.github.io/pdfs/icml_2020_testaug.pdf
        
        1. vflip, hflip and transpose and rotations
        2. custom fivecrops aug
        3. take the mean of predictions
        """
        soft_instances = []
        soft_types = []
        
        # flip ttas
        for aug, deaug in zip(tta.tta_augs(), tta.tta_deaugs()):
            aug_input = aug(image = patch) # (H, W, 3)
            aug_logits = self.get_logits(aug_input["image"]) # (1, C, H, W)
            aug_insts = self.gen_prediction(aug_logits["instances"], squeeze=True) # (H, W, C)
            deaug_insts = deaug(image = aug_insts) # (H, W, C)
            soft_instances.append(deaug_insts["image"])

            if self.class_types == "panoptic":
                aug_types = self.gen_prediction(aug_logits["types"], squeeze=True)# (H, W, C)
                deaug_types = deaug(image=aug_types)  # (H, W, C)
                soft_types.append(deaug_types["image"])
            
        # five crops tta
        scale_up = resize(patch.shape[0], patch.shape[1])
        scale_down = resize(patch.shape[0]//2, patch.shape[1]//2)
        out_insts = np.zeros((patch.shape[0], patch.shape[1], 2))
        out_types = np.zeros((patch.shape[0], patch.shape[1], len(self.classes)))
        for crop in tta.tta_five_crops(patch):
            cropped_im = crop(image = patch)
            scaled_im = scale_up(image = cropped_im["image"]) # (H, W, C)
            aug_logits = self.get_logits(scaled_im["image"])  # (1, C, H, W)
            aug_insts = self.gen_prediction(aug_logits["instances"], squeeze=True) # (H, W, C)
            downscaled_insts = scale_down(image = aug_insts) # (H//2, W//2, C)
            out_insts[crop.y_min:crop.y_max, crop.x_min:crop.x_max] = downscaled_insts["image"] # (H, W, C)
            soft_instances.append(out_insts)

            if self.class_types == "panoptic":
                aug_types = self.__gen_prediction(aug_logits["types"], squeeze=True) # (H, W, C)
                downscaled_types = scale_down(image=aug_types)  # (H//2, W//2, C)
                out_types[crop.y_min:crop.y_max,crop.x_min:crop.x_max] = downscaled_types["image"]  # (H, W, C)
                soft_types.append(out_types)

        aux = None
        if self.aux_branch == "hover":
            logits = self.get_logits(patch) # (1, C, H, W)
            aux = util.tensor_to_ndarray(logits["aux"], squeeze=True)  # (H, W, C)

        # TODO:
        # 16 crops tta
            
        # take the mean of all the predictions
        return {
            "instances":np.asarray(soft_instances).mean(axis=0),
            "types": np.asarray(soft_types).mean(axis=0),
            "aux": aux
        }