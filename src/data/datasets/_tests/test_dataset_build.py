import pytest
import torch
from typing import List
from pathlib import Path

from src.data.datasets.utils import prepare_dataset


@pytest.mark.parametrize("fname", ["tiny_test.h5"])
@pytest.mark.parametrize("phase", ["train", "test", "valid"])
@pytest.mark.parametrize("normalize_input", [True, False])
@pytest.mark.parametrize("return_weight_map", [True, False])
@pytest.mark.parametrize("seg_targets", [
    ["type", "sem"], ["inst", "type"], ["inst"]
])
@pytest.mark.parametrize("img_transforms", [
    ["hue_sat", "non_rigid", "rigid"], []
])
@pytest.mark.parametrize("inst_transforms", [
    ["hover"], ["dist"], ["omnipose"], ["cellpose"],
    [], ["contour"], ["cellpose", "hover", "dist"]
])
def test_dataset_build_load(
        fname: str,
        phase: str,
        seg_targets: List[str],
        img_transforms: List[str],
        inst_transforms: List[str],
        normalize_input: bool,
        return_weight_map: bool,
    ) -> None:
    """
    Test the `prepare_dataset` func with different params and one
    iter of data loading
    """
    folder = Path(__file__).parents[0]
    fname = folder / fname
    input_size = 256 # tiny_test.h5 contains only (256, 256) inputs
    
    dataset = prepare_dataset(
        fname=fname,
        phase=phase,
        img_transforms=img_transforms,
        inst_transforms=inst_transforms,
        input_size=input_size,
        seg_targets=seg_targets,
        normalize_input=normalize_input,
        return_weight_map=return_weight_map,
    )
    
    dat = next(iter(dataset))
    
    # assert all(m in dat.keys() for m in )
    for t in seg_targets:
        assert f"{t}_map" in dat.keys()
    
    if inst_transforms is not None:
        assert all(f"{tr}_map" in dat.keys() for tr in inst_transforms)
        
    # check that input is normalized
    if normalize_input:
        assert torch.min(dat["image"]) >= 0.  
        assert torch.max(dat["image"]) <= 1.
