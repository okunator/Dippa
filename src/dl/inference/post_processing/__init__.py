from .hover.processor import HoverNetPostProcessor
from .cellpose.processor import CellposePostProcessor
from .dist.processor import DistPostProcessor
# from .contour.processor import ContourPostProcessor
# from .basic.processor import BasicPostProcessor


POST_PROC_LOOKUP = {
    "hover":"HoverNetPostProcessor",
    "cellpose":"CellposePostProcessor",
    "dist":"DistPostProcessor",
    "contour":"ContourPostProcessor",
    "basic":"BasicPostProcessor",
}
