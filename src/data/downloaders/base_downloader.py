import scipy.io
import numpy as np
from pathlib import Path
from typing import Union

from src.utils import (
    FileHandler, 
    get_inst_centroid, 
    bounding_box, 
    get_inst_types
)


class BaseDownloader(FileHandler):
    """
    Base downloader class.
    """    
    def _add_slots2masks(self, ann_dir: Union[str, Path]) -> None:
        """
        Takes in a file path of the mask and adds np.arrays
        under "inst_type", "inst_centroid" and "bbox" keys 
        of the .mat file

        Args:
        ----------
            ann_dir (str or Path obj): 
                The path to the annotation mask directory
        """
        for ann_path in sorted(Path(ann_dir).glob("*.mat")):
            m = scipy.io.loadmat(ann_path)
            inst_map = m["inst_map"].astype("int32")
            type_map = m["type_map"].astype("int32")
            inst_ids = list(np.unique(inst_map)[1:])
            centroids = get_inst_centroid(inst_map)
            bboxes = np.array([bounding_box(np.array(inst_map == id_, np.uint8)) for id_ in inst_ids])
            inst_types = get_inst_types(inst_map, type_map)

            scipy.io.savemat(
                file_name=ann_path,
                mdict={
                    "inst_map": inst_map,
                    "type_map": type_map,
                    "inst_type": inst_types,
                    "inst_centroid": centroids,
                    "inst_bbox": bboxes
                }
            )