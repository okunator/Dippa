import numpy as np
from typing import Tuple
from albumentations import Compose

from src.dl.datasets.augs import rigid_augs_and_crop


class BaseWriter:
    """
    Base class for writing datasets to disk
    """
    def _augment_patches(self, 
                         patches_im: np.ndarray, 
                         patches_mask: np.ndarray, 
                         crop_shape: Tuple[int]=(256, 256)) -> Tuple[np.ndarray]:
        """
        Rotations, flips and other rigid augmentations followed by a center crop.
        For dataloading performance, if patches are big it's good to apply rigid 
        augs and crop to rather than in the dataset class.

        Args:
        ---------
            patches_im (np.ndarray):
                Image patches. Shape (n_patches, pH, pW, 3)
            patches_mask (np.ndarray):
                Patches of the masks. Shape (n_patches, pH, pW, n_masks).
                inst_maps in patches_mask[0], type_maps in patches_mask[1]
            crop_shape (Tuple[int], default=(256, 256)):
                Shape of the center crop.

        Returns:
            A tuple of nd.arrays of the transformed patches.
        """
        
        imgs, inst_maps, type_maps, overlays = [], [], [], []

        patches_im = patches_im.astype("uint8")
        for i in range(patches_im.shape[0]):
            cropped_patches = rigid_augs_and_crop(
                image=patches_im[i],
                masks=patches_mask[i],
                crop_shape=crop_shape
            )
            imgs.append(cropped_patches["image"])
            inst_maps.append(cropped_patches["masks"][0][..., 0])
            type_maps.append(cropped_patches["masks"][0][..., 1])

        imgs, insts, types = np.array(imgs), np.array(inst_maps), np.array(type_maps)
        full_data = np.concatenate((imgs, insts[..., None], types[..., None]), axis=-1)
        return full_data
        

    def _pixels_per_classes(self, type_map: np.ndarray) -> np.ndarray:
        """
        Compute pixels belonging to each class

        Args:
        ---------
        type_map (np.ndarray):
            Type map of shape (H, W).
        
        Returns:
        ---------
            np.ndarray of shape (C, ). indices are classes, values teh number of pixels per cls
        """
        totals = np.zeros(len(self.classes))
        for j, val in enumerate(self.classes.values()):
            totals[j] += sum(sum(type_map == val))
            
        return totals.astype("int32")