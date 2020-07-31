import tables
import collections
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from .pre_processing import gen_unet_labels
from .process_utils import instance_contours


class BinarySegmentationDataset(Dataset):
    def __init__(self, fname, transforms):
        """
        PyTorch Dataset object that loads items from pytables db

        Args:
            fname (str) : path to the pytables database
            transforms (albu.Compose) : albumentations.Compose object of augmentations from albumentations pkg
        """
        self.fname = fname
        self.transforms = transforms
        self.tables = tables.open_file(self.fname)
        self.numpixels = self.tables.root.numpixels[:]
        self.nitems = self.tables.root.img.shape[0]
        self.tables.close()
        self.img = None
        self.mask = None
        
    def __getitem__(self, index):
        """
        Patched images and masks are stored in the pytables HDF5 database as numpy arrays of shape:
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
    
    def __len__(self): return self.nitems
    
    
# def visualize_batch(loader):
#     """
#     Visualize a batch of a datloader
#     Args:
#         loader (DataLoader) : torch DataLoader obj that contains custom DataSet object defined above
#     """
#     assert isinstance(loader, DataLoader)
#     
#     batch = next(iter(loader))
# 
#     # [6 x 3 x patch_size x patch_size]
#     image_b = batch['image']
#     # [batch_size x patch_size x patch_size]
#     mask_b = batch['mask']
#     # [batch_size x patch_size x patch_size]
#     contour_b = batch['contour']
#     # [batch_size x patch_sizex patch_size]
#     wmap_b = batch['mask_weight']
# 
#     fig, axes = plt.subplots(2, 3, figsize=(30, 20))
#     for i, ax in enumerate(axes.flat):
#         plt.subplot(2, 3, i+1)
#         plt.imshow(image_b[i, ...].permute(1, 2, 0), interpolation='none')
#         plt.imshow(mask_b[i, ...], interpolation='none', alpha=0.3)
#         plt.imshow(contour_b[i, ...], interpolation='none', alpha=0.3, cmap="jet")
#         plt.imshow(wmap_b[i, ...], interpolation='none', alpha=0.3, cmap="magma")
#         plt.tight_layout(w_pad=4, h_pad=4)
    

    
# def get_loaders(phases, dbs, transforms, batch_size=6, edge_weight=True):
#     """
#     Initialize PyTorch dataloaders and datsets
#     
#     Args:
#         phases (list[str]) : list of the phases used in training e.g. ['train', 'valid']
#         dbs dict(phase:path) : e.g {'train': "/mypath/train.pytable", 'valid':"/pathto/valid.pytable"}
#         transforms (dict[phase:list]) : dict of albu.compose objs of augs for each phase of training
#         batch_size (int) : size of the batch (number of images) per iteration in training
#         edge_weight (bool) : apply weights on the edges of the nuclei instances in cross-entropy loss
#     Returns:
#         loaders (OrderedDict[phase:DataLoader]) : dict of dataloaders for each phase of training
#         datasets (OrderedDict[phase:DataSet]) : dict of PyTorch Datasets for each phase of training
#     """
# 
#     assert all(phase in ('train', 'valid') for phase in phases)
#     assert all(phase in ('train', 'valid') for phase in dbs.keys())
#     assert all(phase in ('train', 'valid') for phase in transforms.keys())
#         
#     dataset = collections.OrderedDict()
#     loaders = collections.OrderedDict()
#     
#     for phase in phases:
#         dataset[phase] = BinarySegmentationDataset(
#             fname = dbs[phase], 
#             transforms = transforms[phase],
#             edge_weight = edge_weight
#         )
#         
#         loaders[phase] = DataLoader(
#             dataset[phase], 
#             batch_size = batch_size, 
#             shuffle = True, 
#             num_workers = 8, 
#             pin_memory = True
#         )
#         
#     return loaders, dataset