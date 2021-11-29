import pytest
import numpy as np
import pytorch_lightning as pl

from typing import Dict, Tuple

from src.settings import MODULE_DIR

from src.dl.inference import Inferer
from src.dl.lightning import SegExperiment
from src.dl.models import MultiTaskSegModel


def model(input_size: int) -> pl.LightningModule:
    model = MultiTaskSegModel(
        dec_branches={"aux": 2, "inst": 2, "sem": 3, "type": 4},
        model_input_size=input_size
    )
    
    lightning_model = SegExperiment(
        model, "test", "test", inference_mode=True, hparams_to_yaml=False
    )
    
    return lightning_model


@pytest.mark.required
@pytest.mark.parametrize("loader_batch_size", [1, 2])
@pytest.mark.parametrize("stride_size", [80, 256])
@pytest.mark.parametrize("patch_size", [(256, 256), (320, 320)])
@pytest.mark.parametrize("fn_pattern", ["*_l*"])
@pytest.mark.parametrize("branch_weights", [
    {"type": False, "sem": False, "aux": True, "inst": False},
])
@pytest.mark.parametrize("branch_acts", [
    {"type": "softmax", "sem": "softmax", "aux": None, "inst": "sigmoid"}
])
def test_infer(
        fn_pattern: str,
        branch_weights: Dict[str, bool],
        branch_acts: Dict[str, str],
        loader_batch_size: int,
        stride_size: int,
        patch_size: Tuple[int, int]
    ) -> None:

    in_dir = MODULE_DIR / "dl/inference/_tests/_test_data"
    test_model = model(patch_size[0])
    
    inferer = Inferer(
        test_model,
        post_proc_method="hover",
        in_data_dir=in_dir,
        patch_size=patch_size,
        stride_size=stride_size,
        branch_weights=branch_weights,
        branch_acts=branch_acts,
        loader_batch_size=loader_batch_size,
        loader_num_workers=1,
        model_batch_size=8,
        fn_pattern=fn_pattern,
        test_mode=True
    )
    
    n_images_real = int(np.ceil(inferer.n_images / inferer.loader_batch_size))
    loader = inferer._chunks(iterable=inferer.dataloader, size=n_images_real)
    
    inferer._infer(next(loader))
    