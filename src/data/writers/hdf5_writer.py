import numpy as np
import tables as tb
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Union, Optional, Dict

from src.utils import FileHandler
from src.patching import TilerStitcher
from .base_writer import BaseWriter


class HDF5Writer(BaseWriter, FileHandler):
    def __init__(self,
                 img_dir: Union[str, Path],
                 mask_dir: Union[str, Path],
                 save_dir: Union[str, Path],
                 file_name: str,
                 classes: Dict[str, int],
                 patch_shape: Tuple[int]=(512, 512),
                 rigid_augs_and_crop: bool = True,
                 crop_shape: Tuple[int]=(256, 256),
                 stride_size: int=80,
                 chunk_size: int=1) -> None:
        """
        Iterates image and mask folders, patches them, and appends the patches to a hdf5 dataset.
        The dataset does not contain any threading locks, so the datset class should handle that 
        
        Args:
        ------------
            img_dir (str, or Path obj):
                Path to the image directory. Image reading is performed with cv2, so image format
                needs to be cv2 readable.
            mask_dir (str or Path obj):
                directory of the corresponding masks for the images. (Make sure the mask filenames
                are the same or at least contain a part of the corresponding image file names. 
                (Suffix of the mask file name can be different). Masks need to be stored in .mat files
                that contain at least the key: "inst_map". Also the "type_map".
            save_dir (str, Path):
                The directory, where the zarr array is written/saved.
            file_name (str):
                name of the zarr array
            classes (Dist[str, int]):
                Dictionary of class integer key-value pairs
                e.g. {"background":0, "inflammatory":1, "epithel":2}
            patch_shape (Tuple[int], default=(512, 512)):
                specifies the height and width of the patches that are stored in zarr-arrays.
            rigid_augs_and_crop (bool, default=True):
                If True, rotations, flips etc are applied to the patches which is followed by a
                center cropping to smaller patch. 
            crop_shape (Tuple[int], default=(256, 256)):
                If rigid_augs_and_crop is True, this is the crop shape for the center crop. 
            stride_size (int, default=80):
                Stride size for the sliding window patcher. Needs to be <= patch_shape.
                If < patch_shape, patches are created with overlap.
            chunk_size (int, default=1):
                The chunk size of the zarr array. i.e. How many patches are included in
                one read of the array.
        """
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.save_dir = Path(save_dir)
        self.patch_shape = patch_shape
        self.stride_size = stride_size
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.classes = classes
        self.rac = rigid_augs_and_crop
        self.crop_shape = crop_shape

        assert self.img_dir.exists(), f"img_dir: {img_dir} does not exist."
        assert self.mask_dir.exists(), f"mask_dir: {mask_dir} does not exist."
        assert self.save_dir.exists(), f"write_dir: {save_dir} does not exist."
        assert self.stride_size <= self.patch_shape[0]


    def write2hdf5(self) -> None:
        """
        Write the the images and masks to zarr group. Mimics HDF5 file
        """
        # Open hdf5 root
        h5 = tb.open_file(self.save_dir / f"{self.file_name}.h5", mode="w")
        root = h5.root

        # save some params as metadata
        root._v_attrs.stride_size = self.stride_size
        root._v_attrs.img_dir = self.img_dir.as_posix()
        root._v_attrs.mask_dir = self.mask_dir.as_posix()
        root._v_attrs.classes = self.classes
 
        ph, pw = self.patch_shape if not self.rac else self.crop_shape
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

        # Iterate imgs and masks -> patch -> save to hdf5
        img_files = self.get_files(self.img_dir)
        mask_files = self.get_files(self.mask_dir)
        
        with tqdm(total=len(img_files), unit="file") as pbar:
            for i, (img_path, mask_path) in enumerate(zip(img_files, mask_files), 1):
                im = self.read_img(img_path)
                inst_map = self.read_mask(mask_path, key="inst_map")
                type_map = self.read_mask(mask_path, key="type_map")
                npixels[:] += self._pixels_per_classes(type_map)

                full_data = np.concatenate((im, inst_map[..., None], type_map[..., None]), axis=-1)
                H, W, C = full_data.shape
                tiler = TilerStitcher((H, W, C), self.patch_shape, self.stride_size)
                patches = tiler.extract_patches_quick(full_data)

                if self.rac:
                    im_p, inst_p, type_p = self._augment_patches(
                        patches_im=patches[..., :3], 
                        patches_mask=patches[..., 3:],
                        crop_shape=self.crop_shape
                    )
                else:
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

        # Add # of patches to attrs
        root._v_attrs.n_items = imgs.shape[0]
        h5.close()