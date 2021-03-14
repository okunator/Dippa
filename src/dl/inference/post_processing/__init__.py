from .hover.processor import HoverNetPostProcessor
from .cellpose.processor import CellposePostProcessor
from .drfns.processor import DRFNSPostProcessor
from .dcan.processor import DCANPostProcessor
from .dran.processor import DRANPostProcessor
from .basic.processor import BasicPostProcessor

from .thresholding import *
from .combine_type_inst import *


POST_PROC_LOOKUP = {
    "hover":"HoverNetPostProcessor",
    "cellpose":"CellposePostProcessor",
    "drfns":"DRFNSPostProcessor",
    "dcan":"DCANPostProcessor",
    "dran":"DRANPostProcessor",
    "basic":"BasicPostProcessor",
}
