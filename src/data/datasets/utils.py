from typing import List, Dict, Union
from torch.utils.data import Dataset
from pathlib import Path

from . import *


ds = vars()


def prepare_dataset(
        fname: Union[str, Path],
        name: str,
        phase: str,
        augs: List[str],
        input_size: int,
        target_types: List[str],
        normalize_input: bool,
        return_weight_map: bool,
        rm_touching_nuc_borders: bool,
    ) -> Dataset:
    """
    Initialize a torch Dataset based on given args.
    
    Args:
    -----------
        fname (str or Path):
            The path to the h5/zarr db where the targets and input imgs
            are saved
        name (str):
            The name of the dataset type. Use lowercase letters.
            Allowed: ("hover", "unet", "basic", "contour", "dist")
        phase (str):
            The dataset phase. Allowed: ("train", "test", "valid")
        augs (List[str]):
            List of augmentations. Allowed: "rigid", "non_rigid", 
            "hue_sat", "blur", "non_spatial", "random_crop", 
            "center_crop", "resize"
        input_size (int):
            Size of the height and width of the input images
        target_types (List[str]):
            A list of the targets that are loaded during dataloading
            process. Allowed values: "inst", "type", "sem".
        normalize_input (bool, default=False):
            apply minmax normalization to input images after 
            transforms
        return_weight_map (bool, default=False):
            Include a nuclear border weight map in the dataloading
            process
        rm_touching_nuc_borders (bool, default=False):
            If True, the pixels that are touching between distinct
            nuclear objects are removed from the masks.

    Returns:
    -----------
        Dataset: Initialized torch Dataset.
    """
    assert Path(fname).exists(), (
        f"Given `fname`: {fname} does not exists."
    )
    
    allowed_ds = ds["DS_LOOKUP"].keys()
    assert name in allowed_ds, (
        f"Illegal dataset name given. Allowed: {allowed_ds}. Got {name}."
    )
    
    allowed_phase = ("train", "test", "valid")
    assert phase in allowed_phase, (
        f"Illegal phase given. Allowed: {allowed_phase}. Got {phase}."
    )
    
    if augs is not None:
        allowed_augs = ds["AUGS_LOOKUP"].keys()
        assert all(aug in allowed_augs for aug in augs), (
            f"Illegal augmentation given. Allowed: {allowed_augs}. Got {augs}."
        )
    
    # init augmentations
    aug_list = []
    if augs is not None and phase == "train":
        kwargs={"height": input_size, "width": input_size}
        aug_list = [ds[ds["AUGS_LOOKUP"][aug]](**kwargs) for aug in augs]
        
    aug_list.append(ds["to_tensor"]())
    aug_list = ds["compose"](aug_list) # to A.Compose
    
    # return dataset
    key = ds["DS_LOOKUP"][name]
    return ds[key](
        fname=fname,
        transforms=aug_list,
        target_types=target_types,
        normalize_input=normalize_input,
        return_weight_map=return_weight_map,
        rm_touching_nuc_borders=rm_touching_nuc_borders
    )
