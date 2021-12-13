import torch
import numpy as np
import albumentations as A
from typing import Tuple, List, Dict, Callable

from ._inst_transforms import OnlyInstMapTransform


__all__ = ["apply_each", "compose", "to_tensor"]


class ApplyEach(A.BaseCompose):
    def __init__(
            self,
            transforms: List[A.BasicTransform],
            p: float=1.
        ) -> None:
        """
        Apply each transform to the input non-sequentially and return
        outputs for each transform
        """
        
        super().__init__(transforms, p)
        
    def __call__(self, **data):
        res = {}
        for t in self.transforms:
            res[t.name] = t(force_apply=True, **data)
        
        return res
    

class ToTensorV3(A.BasicTransform):
    """
    Convert image, masks and auxilliary inputs to tensor
    """
    def __init__(self):
        super().__init__(always_apply=True, p=1.0)
    
    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "masks": self.apply_to_masks,
            "aux": self.apply_to_aux
        }
    
    def apply(self, img: np.ndarray, **params):
        if len(img.shape) not in [2, 3]:
            raise ValueError("Images need to be in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1)).float()
    
    def apply_to_masks(self, masks: List[np.ndarray], **params):
        for i, mask in enumerate(masks):
            if mask.ndim == 3:
                mask = mask.transpose(2, 0, 1)
            masks[i] = torch.from_numpy(mask).long()
        
        return masks
    
    def apply_to_aux(self, auxilliary: Dict[str, np.ndarray], **params):
        for k, aux in auxilliary.items():
            auxilliary[k] = torch.from_numpy(aux["inst_map"]).float()

        return auxilliary

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("always_apply", "p")
    

def apply_each(transforms: List[OnlyInstMapTransform]) -> Callable:
    """
    Wrapper function to apply each transform to the input
    non-sequentially.
    
    Returns:
    -----------
        ApplyEach: ApplyEach object
    """
    
    result = ApplyEach(
        [item for sublist in transforms for item in sublist]
    )
    
    return result


def compose(transforms_to_compose: List[A.BasicTransform]) -> Callable:
    """
    Wrapper for albumentations compose func. Takes in a list of 
    albumentation transforms and composes them to one transformation 
    pipeline

    Returns:
    ----------
        A composed pipeline of albumentation transforms
    """
    result = A.Compose(
        [item for sublist in transforms_to_compose for item in sublist]
    )
    return result
