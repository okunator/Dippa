from .hover.processor import HoverNetPostProcessor
from .cellpose.processor import CellposePostProcessor
from .cellpose._old.processor import CellposePostProcessorOld
from .omnipose.processor import OmniposePostProcessor
from .drfns.processor import DRFNSPostProcessor
from .dcan.processor import DCANPostProcessor
from .dran.processor import DRANPostProcessor
from .basic.processor import BasicPostProcessor

from ._base._thresholding import *
from ._base._combine_type_inst import *


POST_PROC_LOOKUP = {
    "hover": "HoverNetPostProcessor",
    "cellpose": "CellposePostProcessor",
    "cellpose0": "CellposePostProcessorOld",
    "omnipose": "OmniposePostProcessor",
    "drfns": "DRFNSPostProcessor",
    "dcan": "DCANPostProcessor",
    "dran": "DRANPostProcessor",
    "basic": "BasicPostProcessor",
}


THRESH_LOOKUP = {
    "argmax": "argmax",
    "sauvola": "sauvola_thresh",
    "niblack": "niblack_thresh",
    "naive": "naive_thresh_prob"
}


__all__ = [
    "THRESH_LOOKUP", "POST_PROC_LOOKUP", "combine_inst_type",
    "naive_thresh_prob", "niblack_thresh", "sauvola_thresh", "argmax",
    "BasicPostProcessor", "DRANPostProcessor", "DCANPostProcessor",
    "DRFNSPostProcessor", "CellposePostProcessor", "HoverNetPostProcessor",
    "OmniposePostProcessor", "CellposePostProcessorOld"
]