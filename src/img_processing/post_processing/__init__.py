from src.img_processing.post_processing.thresholding import *
from src.img_processing.post_processing.utils import *
from src.img_processing.post_processing.hover.post_proc import *
from src.img_processing.post_processing.basic.post_proc import *
from src.img_processing.post_processing.heuristics.combine_type_inst import *

THERSH_LOOKUP = {
    "argmax":"argmax",
    "sauvola":"sauvola_thresh",
    "niblack":"niblack_thresh",
    "naive":"naive_thresh_prob"
}


POST_PROC_LOOKUP = {
    "hover":{"default":"post_proc_hover", "experimental":"post_proc_hover2"},
    "cellpose":{"default":"post_proc_cellpose"},
    "regular":{"default":"shape_index_ws", "experimental":"inv_dist_watershed"}
}