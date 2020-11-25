import tables
import collections
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader

from src.utils.file_manager import FileHandler
from src.img_processing.pre_processing import gen_unet_labels, gen_hv_maps
from src.img_processing.process_utils import instance_contours, binarize


class SegmentationDataset(Dataset, FileHandler):
    def __init__(self,
                 fname: str,
                 transforms: List,
                 aux_branch: Optional[str]) -> None:
        """
        PyTorch Dataset object that loads items from pytables db

        Args:
            fname (str): path to the pytables database
            transforms (albu.Compose): albumentations.Compose object of 
                                       augmentations from albumentations pkg
            aux_branch (str, optional): one of "hover", "micro"
        """
        self.fname = fname
        self.transforms = transforms
        self.aux_branch = aux_branch
        self.tables = tables.open_file(self.fname)
        self.numpixels = self.tables.root.numpixels[:]
        self.n_items = self.tables.root.img.shape[0]
        self.tables.close()
        self.img = None
        self.mask = None
        
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Patched images and masks are stored in the pytables HDF5 
        database as numpy arrays of shape
        
        1. Read inst_map, type_map and an image from the pytables database corresponding to index
        2. Pre process input masks as in Unet paper (remove touching borders and create weight maps)
        3. If aux branch is used in the network then create the input maps for that branch.
        4. Augment the image, mask, mask weight and the rest of the input data
        5. insert input maps to a dict

        Args:
            index (int) : the index where to slice the numpy array in the database
            
        Returns:
            result (dict) : dictionary containing the augmented image, mask, weight map and filename  
        """
        im_patch, inst_patch, type_patch = self.read_hdf5_patch(self.fname, index)
        unet_inst, weight = gen_unet_labels(inst_patch)

        datalist = []
        datalist.append(type_patch)
        datalist.append(weight)

        # pre-process the inst_maps if aux branch is not None
        if self.aux_branch == "micro":
            datalist.append(unet_inst)
            contour = instance_contours(inst_patch)
            datalist.append(contour)
        elif self.aux_branch == "hover":
            datalist.append(inst_patch)
            hvmaps = gen_hv_maps(inst_patch)
            datalist.append(hvmaps["ymap"])
            datalist.append(hvmaps["xmap"])
        else:
            datalist.append(unet_inst)

        # Augment
        augmented = self.transforms(image=im_patch, masks=datalist)
        img_new = augmented['image']
        masks_new = augmented["masks"]
        
        result = {}
        result['image'] = img_new
        result['type_map'] = masks_new[0]
        result['weight_map'] = masks_new[1]
        result['inst_map'] = masks_new[2]
        result['binary_map'] = binarize(masks_new[2])
        result['filename'] = self.fname
        
        if self.aux_branch == "micro":
            result['contour'] = masks_new[-1]
        elif self.aux_branch == "hover":
            result["xmap"] = masks_new[-1]
            result["ymap"] = masks_new[-2]

        return result
    
    def __len__(self): return self.n_items
    
