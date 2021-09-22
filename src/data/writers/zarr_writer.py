import zarr
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Union, Dict

from src.patching import TilerStitcher
from .base_writer import BaseWriter


# TODO: update to match the HDF%writer
class ZarrWriter(BaseWriter):
    def __init__(self,
                 img_dir: Union[str, Path],
                 mask_dir: Union[str, Path],
                 save_dir: Union[str, Path],
                 file_name: str,
                 classes: Dict[str, int],
                 patch_shape: Tuple[int]=(512, 512),
                 stride_size: int=80,
                 n_copies: int=None,
                 rigid_augs_and_crop: bool = True,
                 crop_shape: Tuple[int]=(256, 256),
                 chunk_size: int=1,
                 chunk_synchronization: bool=True) -> None:
        """
        Iterates image and mask folders, patches them, and appends the patches 
        to a zarr dataset.  
        
        Args:
        ------------
            img_dir (str, or Path obj):
                Path to the image dir. Image reading is performed with cv2, so 
                image format needs to be cv2 readable.
            mask_dir (str or Path obj):
                directory of the corresponding masks for the images. (Make sure
                the mask filenames are the same or at least contain a part of 
                the corresponding image file names. Masks need to be stored in 
                .mat files that contain at least the key: "inst_map".
            save_dir (str, Path):
                The directory, where the zarr array is written/saved.
            file_name (str):
                name of the zarr array
            classes (Dist[str, int]):
                Dictionary of class integer key-value pairs
                e.g. {"background":0, "inflammatory":1, "epithel":2}
            patch_shape (Tuple[int], default=(512, 512)):
                specifies the height and width of the patches that are stored 
                in zarr-arrays.
            stride_size (int, default=80):
                Stride size for the sliding window patcher. Needs to be less or
                equal to patch_shape. If < patch_shape, patches are created 
                with overlap.
            n_copies (int, default=None):
                Number of copies created per one input image & corresponding 
                mask. This argument is ignored if patch_shape is provided. 
                This is used for already patched data such as Pannuke data. If 
                patch_shape and n_copies are None, no additional data is 
                created but transforms may still be applied to the patches.
            rigid_augs_and_crop (bool, default=True):
                If True, rotations, flips etc are applied to the patches which 
                is followed by a center cropping. 
            crop_shape (Tuple[int], default=(256, 256)):
                If rigid_augs_and_crop is True, this is the crop shape for the
                center crop. 
            chunk_size (int, default=1):
                The chunk size of the zarr array of shape: (npatches, H, W, C).
                This param defines the num_patches i.e. How many patches are 
                included in one read of the array.
            chunk_synchronization (bool: default=True):
                Make chunks thread safe. No concurrent writes and reads to the
                same chunk.
        """
        super(ZarrWriter, self).__init__()
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.save_dir = Path(save_dir)
        self.patch_shape = patch_shape
        self.stride_size = stride_size
        self.n_copies = n_copies
        self.file_name = file_name
        self.chunk_sync = chunk_synchronization
        self.chunk_size = chunk_size
        self.classes = classes
        self.rigid_augs_and_crop = rigid_augs_and_crop
        self.crop_shape = crop_shape
        
        if self.patch_shape is not None:
            assert self.stride_size <= self.patch_shape[0]

    def write2db(self, skip: bool=False) -> Path:
        """
        Write the the images and masks to zarr groups. 
        Mimics HDF5 filesystem inside a file

        Args:
        ---------
            skip (bool, default=False):
                If True, skips the db writing and just returns the filename of the db
        """
        fname = Path(self.save_dir / f"{self.file_name}.zarr")

        # Open zarr group
        root = zarr.open(fname.as_posix(), mode="w")

        # save some params as metadata
        root.attrs["stride_size"] = self.stride_size
        root.attrs["img_dir"] = self.img_dir.as_posix()
        root.attrs["mask_dir"] = self.mask_dir.as_posix()
        root.attrs["classes"] = self.classes

        # init zarrays for data
        ph, pw = self.patch_shape if not self.rigid_augs_and_crop and self.patch_shape is not None else self.crop_shape
        imgs = root.zeros(
            "imgs", 
            mode="w", 
            shape=(0, ph, pw, 3), 
            chunks=(self.chunk_size, None, None, None), 
            dtype="u1",
            synchronizer=zarr.ThreadSynchronizer() if self.chunk_sync else None
        )

        imaps = root.zeros(
            "insts", 
            mode="w", 
            shape=(0, ph, pw), 
            chunks=(self.chunk_size, None, None), 
            dtype="i4",
            synchronizer=zarr.ThreadSynchronizer() if self.chunk_sync else None
        )

        tmaps = root.zeros(
            "types", 
            mode="w", 
            shape=(0, ph, pw), 
            chunks=(self.chunk_size, None, None), 
            dtype="i4",
            synchronizer=zarr.ThreadSynchronizer() if self.chunk_sync else None
        )

        npixels = root.zeros(
            "npixels", 
            mode="w", 
            shape=(1, len(self.classes)), 
            chunks=False,
            dtype="i4"
        )

        dataset_mean = root.zeros(
            "dataset_mean", 
            mode="w", 
            shape=(1, 3), 
            chunks=False,
            dtype="f4"
        )

        dataset_std = root.zeros(
            "dataset_std", 
            mode="w", 
            shape=(1, 3), 
            chunks=False,
            dtype="f4"
        )

        # Iterate imgs and masks -> patch -> save to zarr
        imgs = self.get_files(self.img_dir)
        masks = self.get_files(self.mask_dir)

        # For dataset stats computations
        channel_sum = np.zeros(3)
        channel_sum_sq = np.zeros(3)
        pixel_num = 0
        
        with tqdm(total=len(imgs), unit="file") as pbar:
            for img_path, mask_path in zip(imgs, masks):
                im = self.read_img(img_path)
                inst_map = self.read_mask(mask_path, key="inst_map")
                type_map = self.read_mask(mask_path, key="type_map")
                npixels[:] += self._pixels_per_classes(type_map)

                full_data = np.concatenate(
                    (im, inst_map[..., None], type_map[..., None]), 
                    axis=-1
                )
                
                # Do patching or create copies of input images 
                if self.patch_shape is not None:
                    H, W, C = full_data.shape
                    tiler = TilerStitcher((H, W, C), self.patch_shape, self.stride_size)
                    patches = tiler.extract_patches_quick(full_data)
                elif self.n_copies is not None:
                    patches = np.stack([full_data]*self.n_copies)
                else:
                    patches = full_data[None, ...]

                if self.rigid_augs_and_crop:
                    patches = self._augment_patches(
                        patches_im=patches[..., :3], 
                        patches_mask=patches[..., 3:],
                        crop_shape=self.crop_shape
                    )

                # Compute stats from the patches
                pixel_stats = self._patch_stats(patches[..., :3])
                pixel_num += pixel_stats[0]
                channel_sum += pixel_stats[1]
                channel_sum_sq += pixel_stats[2]
                
                im_p = patches[..., :3].astype("uint8")
                inst_p = patches[..., 3].astype("int32")
                type_p = patches[..., 4].astype("int32")

                imgs.append(im_p, axis=0)
                imaps.append(inst_p, axis=0)
                tmaps.append(type_p, axis=0)

                # Update tqdm pbar
                npatch = im_p.shape[0] + inst_p.shape[0] + type_p.shape[0]
                pbar.set_postfix(
                    info=f"Writing {npatch} mask and image patches from file: {Path(img_path).name} to zarr array"
                )
                pbar.update(1)

        dataset_mean[:] += channel_sum / pixel_num
        dataset_std[:] += np.sqrt(channel_sum_sq / pixel_num - np.square(dataset_mean))

        # Add # of patches to attrs
        root.attrs["n_items"] = len(imgs)

        return fname