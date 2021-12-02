import pytest
from typing import List

from src.utils import FileHandler
from src.settings import MODULE_DIR
from src.dl.inference.post_processing.utils import post_processor


def _get_samples(target_types: List[str], aux_type: str):
    path1 = MODULE_DIR / "dl/inference/post_processing/_tests/_test_data/test_out1.mat"
    path2 = MODULE_DIR / "dl/inference/post_processing/_tests/_test_data/test_out2.mat"
    masks1 = FileHandler.read_mask(path1, return_all=True)
    masks2 = FileHandler.read_mask(path2, return_all=True)
    
    out = {}
    for t in target_types:
        out[f"{t}_map"] = {"sample1": masks1[f"{t}_map"], "sample2": masks2[f"{t}_map"]}

    out["aux_map"] = {"sample1": masks1[f"{aux_type}_map"], "sample2": masks2[f"{aux_type}_map"]}

    return out
    

# @pytest.mark.parametrize("method", ["hover", "cellpose", "drfns", "dcan", "dran", "basic"])
@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::RuntimeWarning")
@pytest.mark.parametrize("method", ["cellpose", "hover", "basic"])
@pytest.mark.parametrize("aux_type", ["hover"])
@pytest.mark.parametrize("thresh_method", ["naive", "argmax", "sauvola", "niblack"])
@pytest.mark.parametrize("target_types", [
    ["inst", "sem", "type"], ["inst", "sem"], ["inst", "type"], ["sem", "type"]
])
def test_post_proccer(
        method: str,
        aux_type: str,
        target_types: List[str],
        thresh_method: str
    ) -> None:
    """
    Test different post-processors
    """
    post_proccer = post_processor(method, thresh_method=thresh_method)
    samples = _get_samples(target_types, aux_type)
    out_maps = post_proccer.run_post_processing(samples)
    
    for outd in out_maps:
        for k, out in outd.items():
            if k != "fn":
                assert out.dtype == "uint32"
                assert len(out.shape) == 2
