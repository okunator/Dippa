import tables
import collections
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from src.utils.file_manager import FileHandler
from src.img_processing.pre_processing import gen_unet_labels
from src.img_processing.process_utils import instance_contours, binarize


class SegmentationDataset(Dataset, FileHandler):
    def __init__(self, fname, transforms):
        """
        PyTorch Dataset object that loads items from pytables db

        Args:
            fname (str): path to the pytables database
            transforms (albu.Compose): albumentations.Compose object of 
                                       augmentations from albumentations pkg
            class_types (str):  one of ("binary", "types")
        """
        self.fname = fname
        self.transforms = transforms
        self.tables = tables.open_file(self.fname)
        self.numpixels = self.tables.root.numpixels[:]
        self.n_items = self.tables.root.img.shape[0]
        self.tables.close()
        self.img = None
        self.mask = None
        
    def __getitem__(self, index):
        """
        Patched images and masks are stored in the pytables HDF5 
        database as numpy arrays of shape:

        See the database_prep_notebooks for more info.
        
        1. Read inst_map, type_map and an image from the pytables database corresponding to index
        2. Create instance contours for the instance mask that are used as a weight map for the nuclei edges
        3. Process the mask as described in the Unet paper
        4. Augment the image, mask and mask weight
        5. binarize the mask

        Args:
            index (int) : the index where to slice the numpy array in the database
            
        Returns:
            result (dict) : dictionary containing the augmented image, mask, weight map and filename  
        """
        im_patch, inst_patch, type_patch = self.read_hdf5_patch(self.fname, index)
        
        # process the inst_maps to find nuclei borders
        contour = instance_contours(inst_patch)
        inst_map, weight = gen_unet_labels(inst_patch)
                
        # Augment
        augmented = self.transforms(image=im_patch, masks=[inst_map, type_patch, weight, contour])
        img_new = augmented['image']
        imap_new, tmap_new, weight_new, contour_new = augmented['masks']
        
        # Binarize mask
        binary_map = binarize(imap_new)
        
        result = {}
        result['image'] = img_new
        result['inst_map'] = imap_new
        result['binary_map'] = binary_map
        result['type_map'] = tmap_new
        result['weight_map'] = weight_new
        result['contour'] = contour_new
        result['filename'] = self.fname
        return result
    
    def __len__(self): return self.n_items
    
