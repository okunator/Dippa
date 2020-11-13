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
        database as numpy arrays of shape:

        See the database_prep_notebooks for more info.
        
        1. Read inst_map, type_map and an image from the pytables database corresponding to index
        2. Create instance contours for the instance mask that are used as a weight map for the nuclei edges
        3. Process the mask as described in the Unet paper
        4. Augment the image, mask and mask weight and the rest
        5. insert maps to a dict

        Args:
            index (int) : the index where to slice the numpy array in the database
            
        Returns:
            result (dict) : dictionary containing the augmented image, mask, weight map and filename  
        """
        im_patch, inst_patch, type_patch = self.read_hdf5_patch(self.fname, index)

        datalist = []
        inst_map, weight = gen_unet_labels(inst_patch)
        datalist.append(inst_patch)
        datalist.append(type_patch)
        datalist.append(weight)

        # pre-process the inst_maps if aux branch is not None
        if self.aux_branch == "micro":
            contour = instance_contours(inst_patch)
            datalist.append(contour)
        elif self.aux_branch == "hover":
            hvmaps = gen_hv_maps(inst_patch)
            datalist.append(hvmaps["ymap"])
            datalist.append(hvmaps["xmap"])
                
        # Augment
        augmented = self.transforms(image=im_patch, masks=datalist)
        img_new = augmented['image']
        masks_new = augmented["masks"]
        
        result = {}
        result['image'] = img_new
        result['inst_map'] = masks_new[0]
        result['binary_map'] = binarize(masks_new[0])
        result['type_map'] = masks_new[1]
        result['weight_map'] = masks_new[2]
        result['filename'] = self.fname
        
        if self.aux_branch == "micro":
            result['contour'] = masks_new[-1]
        elif self.aux_branch == "hover":
            result["xmap"] = masks_new[-1]
            result["ymap"] = masks_new[-2]

        return result
    
    def __len__(self): return self.n_items
    
