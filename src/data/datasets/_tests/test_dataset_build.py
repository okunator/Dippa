import pytest
import torch
from albumentations.pytorch import ToTensorV2
from typing import List, Dict
from pathlib import Path

from src.data.datasets.utils import prepare_dataset
from src.dl.utils import binarize


@pytest.mark.parametrize("fname", ["tiny_test.h5"])
@pytest.mark.parametrize("name", ["basic", "hover", "dist", "contour", "unet"])
@pytest.mark.parametrize("phase", ["train", "test", "valid"])
@pytest.mark.parametrize("normalize_input", [True, False])
@pytest.mark.parametrize("rm_touching_nuc_borders", [True, False])
@pytest.mark.parametrize("return_weight_map", [True, False])
@pytest.mark.parametrize("target_types", [
    ["inst"], ["inst", "type"], ["inst", "type", "sem"],
    ["type"], ["type", "sem"], ["sem"], ["sem", "inst"] 
])
@pytest.mark.parametrize("augs", [
    ["hue_sat", "non_rigid", "rigid"], [],
    ["hue_sat", "non_rigid", "rigid", "blur"],
    ["random_crop", "non_spatial"]
])
def test_dataset_build_load(
        fname: str,
        name: str,
        phase: str,
        augs: List[str],
        target_types: List[str],
        normalize_input: bool,
        return_weight_map: bool,
        rm_touching_nuc_borders: bool,
    ) -> None:
    """
    Test the `prepare_dataset` func with different params and one
    iter of data loading
    """
    folder = Path(__file__).parents[0]
    fname = folder / fname
    input_size = 256 # tiny_test.h5 contains only (256, 256) inputs
    
    dataset = prepare_dataset(
        fname, name, phase, augs, input_size, target_types,
        normalize_input, return_weight_map, rm_touching_nuc_borders
    )
    
    dat = next(iter(dataset))
    
    # check that the borders get removed if specified
    if rm_touching_nuc_borders:
        if "type" in target_types and "inst" in target_types:
            tmap = binarize(dat["type_map"])
            assert not torch.all(torch.isclose(dat["binary_map"], tmap))
    
    # check that weight map gets loaded
    if return_weight_map:
        assert "weight_map" in dat.keys()
        
    # check that binary map gets loaded
    if "inst" in target_types:
        assert "binary_map" in dat.keys()
        
    # check that sem map gets loaded
    if "sem" in target_types:
        assert "sem_map" in dat.keys()
        
    # check that type map gets loaded
    if "type" in target_types:
        assert "type_map" in dat.keys()
        
    # check that aux map is loaded
    if name in ("hover", "dist", "contour"):
        assert "aux_map" in dat.keys()
    
    # check that input is normalized
    if normalize_input:
        assert torch.min(dat["image"]) >= 0.  
        assert torch.max(dat["image"]) <= 1.
        
    # check that the only aug for "test" and "valid" is ToTensorV2
    if phase in ("test", "valid"):
        assert isinstance(next(iter(dataset.transforms)), ToTensorV2)
    elif phase == "train" and augs:
        assert not isinstance(next(iter(dataset.transforms)), ToTensorV2)
        
        
        
    
    
    