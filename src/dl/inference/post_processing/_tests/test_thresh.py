import pytest
import numpy as np
from typing import Callable

from src.dl.inference.post_processing._base._thresholding import (
    naive_thresh, naive_thresh_prob, sauvola_thresh, 
    morph_chan_vese_thresh, smoothed_thresh, niblack_thresh,
)


@pytest.fixture
def prob_map() -> np.ndarray:
    return np.random.rand(256, 256)


@pytest.mark.parametrize("method", [
    naive_thresh, naive_thresh_prob, sauvola_thresh,
    morph_chan_vese_thresh, smoothed_thresh, niblack_thresh
])
def test_thresh(prob_map: np.ndarray, method: Callable) -> None:
    """
    Quick tests for the different thresholding methods
    """
    binary = method(prob_map)
    
    if len(np.unique(binary)) == 2:
        assert np.amax(binary) == 1
        assert np.amin(binary) == 0
        
    assert binary.dtype == "uint8"
