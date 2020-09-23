import tables
import collections
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from src.img_processing.pre_processing import gen_unet_labels
from src.img_processing.process_utils import instance_contours


class BinarySegmentationDataset(Dataset):
    def __init__(self, fname, transforms):
        """
        PyTorch Dataset object that loads items from pytables db

        Args:
            fname (str) : path to the pytables database
            transforms (albu.Compose) : albumentations.Compose object of 
                                        augmentations from albumentations pkg
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
        img: [npatch x patch_size x patch_size x 3]. 
        mask: [npatch x patch_size x patch_size x 1] 
        
        See the database_prep_notebooks for more info.
        
        1. Read a mask and an image from the pytables database corresponding to index
        2. Create instance contours for the instance mask that are used as a weight map for the nuclei edges
        3. Process the mask as described in the Unet paper
        4. Augment the image, mask and mask weight
        5. binarize the mask

        Args:
            index (int) : the index where to slice the numpy array in the database
            
        Returns:
            result (dict) : dictionary containing the augmented image, mask, weight map and filename  
        """
        with tables.open_file(self.fname,'r') as db:
            self.img = db.root.img
            self.mask = db.root.mask
            img = self.img[index, ...]
            mask = self.mask[index, ...]
        
        # process the masks as in unet paper
        contour = instance_contours(mask)
        mask, weight = gen_unet_labels(mask)
                
        # Augment
        augmented = self.transforms(image=img, masks=[mask, weight, contour])
        img_new = augmented['image']
        mask_new, weight_new, contour_new = augmented['masks']
        
        # Binarize mask
        mask_new[mask_new > 0] = 1
        
        result = {}
        result['image'] = img_new
        result['mask'] = mask_new
        result['mask_weight'] = weight_new
        result['contour'] = contour_new
        result['filename'] = self.fname
        
        return result
    
    def __len__(self): return self.n_items
    