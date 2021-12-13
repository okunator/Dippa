from typing import Union, Callable
from torch.utils.data import Dataset
from pathlib import Path

from .dataset import SegDataset
from . import *


ds = vars()


def prepare_dataset(
        fname: Union[str, Path],
        phase: str,
        img_transforms: Callable,
        inst_transforms: Callable,
        input_size: int,
        seg_targets: int,
        normalize_input: bool=False,
        return_weight_map: bool=False,
    ) -> Dataset:
    """
    Initialize a torch Dataset based on given args.
    
    Args:
    -----------
        fname (str or Path):
            The path to the h5/zarr db where the targets and input imgs
            are saved
        phase (str):
            The dataset phase. Allowed: ("train", "test", "valid")
        img_transforms (albu.Compose): 
            Albumentations.Compose obj (a list of transformations).
            All the transformations that are applied to the input
            images and corresponding masks
        inst_transforms (ApplyEach):
            ApplyEach obj. (a list of augmentations). All the
            transformations that are applied to only the instance
            labelled masks.
        input_size (int):
            Size of the height and width of the input images
        normalize_input (bool, default=False):
            apply minmax normalization to input images after 
            transforms
        return_weight_map (bool, default=False):
            Include a nuclear border weight map in the dataloading
            process

    Returns:
    -----------
        Dataset: Initialized torch Dataset.
    """
    if not Path(fname).exists():
        raise ValueError(
            f"Given `fname`: {fname} does not exists."
        )
        
    allowed_phase = ("train", "test", "valid")
    if not phase in allowed_phase:
       raise ValueError(
            f"Illegal phase given. Allowed: {allowed_phase}. Got {phase}."
        )
    
    if img_transforms is not None:
        allowed_augs = ds["AUGS_LOOKUP"].keys()
        if not all(aug in allowed_augs for aug in img_transforms):
            raise ValueError(f"""
                Illegal augmentation given in `img_transforms`.
                Allowed: {allowed_augs}.
                Got {img_transforms}."""
            )
        
    if inst_transforms is not None:
        allowed_augs = ds["AUX_LOOKUP"].keys()
        if not all(aug in allowed_augs for aug in inst_transforms):
            raise ValueError(f"""
                Illegal augmentation given in `inst_transforms`.
                Allowed: {allowed_augs}.
                Got {inst_transforms}."""
            )
    
    # init augmentations
    aug_list = []
    if img_transforms is not None and phase == "train":
        kwargs={"height": input_size, "width": input_size}
        aug_list = [
            ds[ds["AUGS_LOOKUP"][aug]](**kwargs) for aug in img_transforms
        ]
    aug_list = ds["compose"](aug_list) # to A.Compose
        
    # init aux transforms
    aux_list = []
    if inst_transforms is not None:
        aux_list = [
            ds[ds["AUX_LOOKUP"][aug]]() for aug in inst_transforms
        ]
    
    if return_weight_map:
        aux_list.append(ds[ds["AUX_LOOKUP"]["edge_weight"]]())
    
    aux_list = ds["apply_each"](aux_list)
        
    return SegDataset(
        fname=fname,
        img_transforms=aug_list,
        inst_transforms=aux_list,
        seg_targets=seg_targets,
        normalize_input=normalize_input,
    )
