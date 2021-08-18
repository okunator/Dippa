import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from albumentations import Compose

from src.utils import FileHandler
from src.dl.datasets.augs import rigid_augs_and_crop


class BaseWriter(ABC, FileHandler):
    def __init__(self):
        """
        Base class for writing datasets to disk
        """
        super(BaseWriter, self).__init__()

    @abstractmethod
    def write2db(self):
        raise NotImplementedError

    def _patch_stats(self, patches_im: np.ndarray, num_channels: int=3) -> np.ndarray:
        """
        Compute the number of pixels. Channel-wise sum and channel-wise squared sum
        for dataset mean & std computations.

        Args:
        ----------
            patches_im (np.ndarray):
                Image patches. Shape (n_patches, pH, pW, num_channels)
            num_channels (int, default=3):
                number of channels in the patches

        Returns: 
        ----------
            Tuple[Union[int, np.ndarray]] : The computed statistics. np.ndarrays have shape (3, ).
        """
        pixel_num = 0 
        channel_sum = np.zeros(num_channels)
        channel_sum_sq = np.zeros(num_channels)
        for i in range(patches_im.shape[0]):
            img = patches_im[i] / 255
            pixel_num += (img.size/num_channels)
            channel_sum += np.sum(img, axis=(0, 1))
            channel_sum_sq += np.sum(np.square(img), axis=(0, 1))
        
        return pixel_num, channel_sum.astype("f4"), channel_sum_sq.astype("f4")

    def _augment_patches(self, 
                         patches_im: np.ndarray, 
                         patches_mask: np.ndarray, 
                         crop_shape: Tuple[int]=(256, 256)) -> Tuple[np.ndarray]:
        """
        Rotations, flips and other rigid augmentations followed by a center crop.
        Rigid augmentations on large images can kill the datalaoding performance, 
        so it's worth to apply rigid augmentations and crop beforehand rather when
        loading data with the dataloader. 

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
        ---------
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
        pixels = np.zeros(len(self.classes))
        for j, val in enumerate(self.classes.values()):
            pixels[j] += sum(sum(type_map == val))
            
        return pixels.astype("int32")