import numpy as np
import tables as tb
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Union, Optional, Dict

from src.patching import TilerStitcher
from .base_writer import BaseWriter


class HDF5Writer(BaseWriter):
    def __init__(self,
                 img_dir: Union[str, Path],
                 mask_dir: Union[str, Path],
                 save_dir: Union[str, Path],
                 file_name: str,
                 classes: Dict[str, int],
                 patch_shape: Tuple[int]=(512, 512),
                 stride_size: int=80,
                 n_copies: int=None,
                 rigid_augs_and_crop: bool=True,
                 crop_shape: Tuple[int]=(256, 256),
                 chunk_size: int=1) -> None:
        """
        Iterates image and mask folders, patches them (if specified), and appends the patches to
        a hdf5 dataset. This h5-file does not support thread locks, so the torch dataset class 
        should handle that (if locking is needed at all. Likely not needed).
        
        Args:
        ------------
            img_dir (str, or Path obj):
                Path to the image directory. Image reading is performed with cv2, so image format
                needs to be cv2 readable.
            mask_dir (str or Path obj):
                directory of the corresponding masks for the images. (Make sure the mask filenames
                are the same or at least contain a part of the corresponding image file names. 
                Masks need to be stored in .mat files that contain at least the key: "inst_map".
            save_dir (str, Path):
                The directory, where the h5 array is written/saved.
            file_name (str):
                name of the h5 array
            classes (Dist[str, int]):
                Dictionary of class integer key-value pairs
                e.g. {"background":0, "inflammatory":1, "epithel":2}
            patch_shape (Tuple[int], default=(512, 512)):
                Specifies the height and width of the patches. If this is None, 
                no patching is applied. 
            stride_size (int, default=80):
                Stride size for the sliding window patcher. Needs to be <= patch_shape.
                If < patch_shape, patches are created with overlap. Ignored if patch_shape
                is None.
            n_copies (int, default=None):
                Number of copies created per one input image & corresponding mask.
                Used for already patched data such as Pannuke data. If patch_shape
                and n_copies are None, no additional data is created but transforms 
                may still be applied to the patches.
            rigid_augs_and_crop (bool, default=True):
                If True, rotations, flips etc are applied to the patches which is followed by a
                center cropping. 
            crop_shape (Tuple[int], default=(256, 256)):
                If rigid_augs_and_crop is True, this is the crop shape for the center crop. 
            chunk_size (int, default=1):
                The chunk size of the h5 array of shape: (num_patches, H, W, C). This param defines
                the num_patches i.e. How many patches are included in one read of the array.
        """
        super(HDF5Writer, self).__init__()
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.save_dir = Path(save_dir)
        self.patch_shape = patch_shape
        self.stride_size = stride_size
        self.n_copies = n_copies
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.classes = classes
        self.rigid_augs_and_crop = rigid_augs_and_crop
        self.crop_shape = crop_shape

        if self.patch_shape is not None:
            assert self.stride_size <= self.patch_shape[0]


    def write2db(self, skip: bool=False) -> Path:
        """
        Write the the images and masks to zarr group. Mimics HDF5 file

        Args:
        ---------
            skip (bool, default=False):
                If True, skips the db writing and just returns the filename of the db
        """
        self.create_dir(self.save_dir)
        fname = Path(self.save_dir / f"{self.file_name}.h5")

        if skip:
            return fname

        h5 = tb.open_file(fname.as_posix(), mode="w")
        root = h5.root

        # save some params as metadata
        root._v_attrs.stride_size = self.stride_size
        root._v_attrs.img_dir = self.img_dir.as_posix()
        root._v_attrs.mask_dir = self.mask_dir.as_posix()
        root._v_attrs.classes = self.classes

        ph, pw = self.patch_shape if not self.rigid_augs_and_crop and self.patch_shape is not None else self.crop_shape
        imgs = h5.create_earray(
            where=root, 
            name="imgs", 
            atom=tb.UInt8Atom(),  
            shape=np.append([0], (ph, pw, 3)), 
            chunkshape=np.append([self.chunk_size], (ph, pw, 3)),
            filters=tb.Filters(complevel=5, complib="blosc:lz4")
        )

        insts = h5.create_earray(
            where=root, 
            name="insts", 
            atom=tb.Int32Atom(),  
            shape=np.append([0], (ph, pw)), 
            chunkshape=np.append([self.chunk_size], (ph, pw)),
            filters=tb.Filters(complevel=5, complib="blosc:lz4")
        )

        types = h5.create_earray(
            where=root, 
            name="types", 
            atom=tb.Int32Atom(),  
            shape=np.append([0], (ph, pw)), 
            chunkshape=np.append([self.chunk_size], (ph, pw)),
            filters=tb.Filters(complevel=5, complib="blosc:lz4")
        )

        npixels = h5.create_carray(
            where=root, 
            name="npixels", 
            atom=tb.Int32Atom(), 
            shape=(1, len(self.classes))
        )

        dataset_mean = h5.create_carray(
            where=root, 
            name="dataset_mean", 
            atom=tb.Float32Atom(), 
            shape=(1, 3)
        )

        dataset_std = h5.create_carray(
            where=root, 
            name="dataset_std", 
            atom=tb.Float32Atom(), 
            shape=(1, 3)
        )

        # Iterate imgs and masks -> patch -> save to hdf5
        img_files = self.get_files(self.img_dir)
        mask_files = self.get_files(self.mask_dir)

        # Channel-wise mean & std for the whole dataset
        channel_sum = np.zeros(3)
        channel_sum_sq = np.zeros(3)
        pixel_num = 0
                
        with tqdm(total=len(img_files), unit="file") as pbar:
            for i, (img_path, mask_path) in enumerate(zip(img_files, mask_files), 1):
                # get data
                im = self.read_img(img_path)
                inst_map = self.read_mask(mask_path, key="inst_map")
                type_map = self.read_mask(mask_path, key="type_map")
                npixels[:] += self._pixels_per_classes(type_map)

                im = im[:-1, :] if im.shape[0] > inst_map.shape[0] else im
                im = im[:, :-1] if im.shape[1] > inst_map.shape[1] else im

                full_data = np.concatenate((im, inst_map[..., None], type_map[..., None]), axis=-1)
                
                # Do patching or create copies of input images 
                if self.patch_shape is not None:
                    H, W, C = full_data.shape
                    tiler = TilerStitcher((H, W, C), self.patch_shape, self.stride_size)
                    patches = tiler.extract_patches_quick(full_data)
                elif self.n_copies is not None:
                    patches = np.stack([full_data]*self.n_copies)
                else:
                    patches = full_data[None, ...]

                if self.self.rigid_augs_and_crop:
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

                imgs.append(im_p)
                insts.append(inst_p)
                types.append(type_p)

                # Update tqdm pbar
                npatch = im_p.shape[0] + inst_p.shape[0] + type_p.shape[0]
                pbar.set_postfix(
                    info=f"Writing {npatch} mask and image patches from file: {Path(img_path).name} to hdf5 storage"
                )
                pbar.update(1)

        # Compute dataset level mean & std 
        dataset_mean[:] += channel_sum / pixel_num
        dataset_std[:] += np.sqrt(channel_sum_sq / pixel_num - np.square(dataset_mean))

        # Add # of patches to attrs
        root._v_attrs.n_items = imgs.shape[0]
        h5.close()
        return fname