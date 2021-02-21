from .thresholding import *
from .utils import *
from .hover.post_proc import post_proc_hover
# from .hover.processor import HoverNetPostProcessor
from .basic.post_proc import inv_dist_watershed, shape_index_watershed2
from .combine_type_inst import combine_inst_semantic

THRESH_LOOKUP = {
    "argmax":"argmax",
    "sauvola":"sauvola_thresh",
    "niblack":"niblack_thresh",
    "naive":"naive_thresh_prob"
}

POST_PROC_LOOKUP = {
    "hover":"HoverNetPostProcessor",
    "cellpose":"CellposePostProcessor",
    "dist":"DistPostProcessor",
    "contour":"ContourPostProcessor",
    "basic":"BasicPostProcessor",
}
