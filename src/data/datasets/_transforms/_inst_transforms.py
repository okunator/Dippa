import numpy as np
import albumentations as A
from typing import Dict, Callable

from ._functional.cellpose import gen_flow_maps
from ._functional.omnipose import gen_omni_flow_maps, smooth_distance
from ._functional.hover import gen_hv_maps
from ._functional.dist import gen_dist_maps
from ._functional.contour import gen_contour_maps
from ._functional.edge_weights import gen_weight_maps
from src.utils import remove_1px_boundary, fix_duplicates, binarize


__all__ = [
    "hover_transform", "dist_transform", "smooth_dist_transform",
    "contour_transform", "omnipose_transform", "cellpose_transform",
    "edgeweight_transform", "rm_borders_transform", "binarize_transform"
]


class OnlyInstMapTransform(A.BasicTransform):
    """
    Transforms applied to only instance labelled masks.
    """
    def __init__(self) -> None:
        super().__init__(always_apply=True, p=1.0)
    
    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "inst_map": self.apply_to_instmap,
        }
        
    def apply(self, img: np.ndarray, **params):
        return
    
    def apply_to_instmap(self, inst_map: np.ndarray, **params):
        raise NotImplementedError("`apply_to_instmap` method not implemented")


class OmniposeTrans(OnlyInstMapTransform):
    def __init__(self):
        """
        remap the labels to remove duplicate labels and apply 
        horizontal and vertical gradients to the instance labelled mask.
        """
        super().__init__()
        self.name = "omnipose"
    
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        return gen_omni_flow_maps(fix_duplicates(inst_map))
    
    
class CellposeTrans(OnlyInstMapTransform):
    def __init__(self):
        super().__init__()
        self.name = "cellpose"
    
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        """
        remove duplicate labels and apply horizontal and vertical 
        gradients to the heat diffusion of the instance labelled mask.
        """
        return gen_flow_maps(fix_duplicates(inst_map))
    

class HoVerTrans(OnlyInstMapTransform):
    def __init__(self):
        """
        Remove duplicate labels and apply horizontal and vertical
        gradients to the instance labelled mask.
        """
        super().__init__()
        self.name = "hover"
    
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        return gen_hv_maps(fix_duplicates(inst_map))
    
    
class DistTrans(OnlyInstMapTransform):
    def __init__(self) -> None:
        """
        Remove duplicate labels and apply distance transform to the
        instances in an instance labelled mask.
        """
        super().__init__()
        self.name = "dist"
           
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        return gen_dist_maps(fix_duplicates(inst_map))
    
    
class SmoothDistTrans(OnlyInstMapTransform):
    def __init__(self):
        super().__init__()
        self.name = "dist"
           
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        """
        Apply a FIM distance transform to the instances in an instance
        labelled mask.
        """
        return smooth_distance(inst_map)
    
    
class ContourTrans(OnlyInstMapTransform):
    def __init__(self):
        """
        Remove duplicate labels and find the contours of the instances
        in an instance labelled mask.
        """
        super().__init__()
        self.name = "contour"
           
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        return gen_contour_maps(fix_duplicates(inst_map))
    

class RmNucleiBorders(OnlyInstMapTransform):
    def __init__(self):
        """
        Remove one pixel around the borders of the instances in an
        instance labelled mask 
        """
        super().__init__()
        self.name = "inst"
           
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        return remove_1px_boundary(inst_map)
    

class EdgeWeightTrans(OnlyInstMapTransform):
    def __init__(self):
        """
        Remove one pixel around the borders of the instances in an
        instance labelled mask 
        """
        super().__init__()
        self.name = "weight"
           
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        return gen_weight_maps(inst_map)
    
    
class BinarizeTrans(OnlyInstMapTransform):
    def __init__(self):
        """
        Remove one pixel around the borders of the instances in an
        instance labelled mask 
        """
        super().__init__()
        self.name = "binary"
           
    def apply_to_instmap(self, inst_map: np.ndarray, **params) -> np.ndarray:
        return binarize(inst_map)
    

def hover_transform(**kwargs):
    return [HoVerTrans()]


def dist_transform(**kwargs):
    return [DistTrans()]


def smooth_dist_transform(**kwargs):
    return [SmoothDistTrans()]


def contour_transform(**kwargs):
    return [ContourTrans()]


def omnipose_transform(**kwargs):
    return [OmniposeTrans()]


def cellpose_transform(**kwargs):
    return [CellposeTrans()]


def rm_borders_transform(**kwargs):
    return [RmNucleiBorders()]


def edgeweight_transform(**kwargs):
    return [EdgeWeightTrans()]


def binarize_transform(**kwargs):
    return [BinarizeTrans()]
