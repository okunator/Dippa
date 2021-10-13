import numpy as np
import tables as tb
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Union, Dict, BinaryIO, Optional

from src.patching import TilerStitcher
from src.data.writers.base_writer import BaseWriter


class HDF5Writer(BaseWriter):
    def __init__(
            self,
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
            chunk_size: int=1,
            sem_classes: Optional[Dict[str, int]]=None
        ) -> None:
        """
        Iterates image and mask folders, patches them (if specified), 
        and appends the patches to a hdf5 dataset. 
        
        Args:
        ------------
            img_dir (str, or Path obj):
                Path to the image directory. Image reading is performed 
                with cv2, so image format needs to be cv2 readable.
            mask_dir (str or Path obj):
                directory of the corresponding masks for the images. 
                Make sure the mask filenames correspond to image file 
                names. Masks need to be stored in .mat files that 
                contain at least the key: "inst_map".
            save_dir (str, Path):
                The directory, where the h5 array is written.
            file_name (str):
                name of the h5 array.
            classes (Dist[str, int]):
                Dictionary of class integer key-value pairs
                e.g. {"background":0, "inflammatory":1, "epithel":2}
            patch_shape (Tuple[int], default=(512, 512)):
                Specifies the height and width of the patches. If this 
                is None, no patching is applied. 
            stride_size (int, default=80):
                Stride size for the sliding window patcher. Needs to be 
                less or equal to the patch_shape. If less than 
                patch_shape, patches are created with overlap. This arg
                is ignored if patch_shape is None.
            n_copies (int, default=None):
                Number of copies created per one input image and the 
                corresponding mask. This argument is ignored if 
                patch_shape is provided. This arg is used for already 
                patched data such as Pannuke to multiply the # of imgs.
            rigid_augs_and_crop (bool, default=True):
                If True, rotations, flips etc are applied to the patches
                which is followed by a center cropping. 
            crop_shape (Tuple[int], default=(256, 256)):
                If rigid_augs_and_crop is True, this is the crop shape 
                for the center crop. 
            chunk_size (int, default=1):
                The chunk size of the h5 arrays. 
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
        self.sem_classes = sem_classes

        if self.patch_shape is not None:
            self.n_copies = None
            assert self.stride_size is not None, "stride is not set."
            assert self.stride_size <= self.patch_shape[0], (
                "stride_size needs to be less or equal to the patch_shape"
            )
        
        if self.rigid_augs_and_crop:
            assert self.crop_shape is not None, (
                "If rigids_augs_and_crop is True, crop_shape can't be None"
            )


    def _write_files(self, 
                     h5: BinaryIO, 
                     img_dir: str, 
                     mask_dir: str) -> None:
        """
        Writes img and mask files to a hdf5 db

        Args:
        ---------
            h5 (BinaryIO):
                H5 filehandle 
            img_dir (str):
                Directory of the images
            mask_dir (str):
                Directory of the masks
        """
        root = h5.root
        
        # Iterate imgs and masks -> patch -> save to hdf5
        imgs = sorted(Path(img_dir).glob("*"))
        masks = sorted(Path(mask_dir).glob("*"))

        with tqdm(total=len(imgs), unit="file") as pbar:
            for img_path, mask_path in zip(imgs, masks):
                im = self.read_img(img_path)
                insts = self.read_mask(mask_path, key="inst_map")
                types = self.read_mask(mask_path, key="type_map")

                # most of the time there are no semantic maps
                try:
                    areas = self.read_mask(mask_path, key="sem_map")
                except:
                    areas = np.zeros_like(insts)

                # add # of pixels per class to db for class weighting
                root.npixels[:] += self._pixels_per_classes(types)

                # workout shape inconsistencies that occur 
                im = im[:-1, :] if im.shape[0] > insts.shape[0] else im
                im = im[:, :-1] if im.shape[1] > insts.shape[1] else im
                
                full_data = np.concatenate(
                    (im, insts[..., None], types[..., None], areas[..., None]), 
                    axis=-1
                )

                # Do patching or create copies of input images
                if self.patch_shape is not None:
                    tiler = TilerStitcher(
                        full_data.shape, 
                        self.patch_shape, 
                        self.stride_size
                    )
                    patches = tiler.extract_patches_quick(full_data)
                elif self.n_copies is not None:
                    patches = np.stack([full_data]*self.n_copies)
                else:
                    patches = full_data[None, ...]

                # apply rigid augmentations and center cropping
                if self.rigid_augs_and_crop:
                    patches = self._augment_patches(
                        patches_im=patches[..., :3], 
                        patches_mask=patches[..., 3:],
                        crop_shape=self.crop_shape
                    )

                # Compute stats from the patches
                pixel_stats = self._patch_stats(patches[..., :3])
                root.pixel_num[:] += pixel_stats[0]
                root.channel_sum[:] += pixel_stats[1]
                root.channel_sum_sq[:] += pixel_stats[2]
                
                im_p = patches[..., :3].astype("uint8")
                inst_p = patches[..., 3].astype("int32")
                type_p = patches[..., 4].astype("int32")
                sem_p = patches[..., 5].astype("int32")

                root.imgs.append(im_p)
                root.insts.append(inst_p)
                root.types.append(type_p)
                root.areas.append(sem_p)

                pbar.set_postfix(
                    info=(
                        f"Writing mask and image patches to hdf5 file", 
                        f"{h5.root._v_attrs.file_name}.h5"
                    )
                )
                pbar.update(1)

        # Compute dataset level mean & std 
        # TODO: This is failing when add2db is used
        root.dataset_mean[:] = root.channel_sum[:] / root.pixel_num[:]
        root.dataset_std[:] = np.sqrt(
            (root.channel_sum_sq[:] / root.pixel_num[:])
            - np.square(root.dataset_mean[:])
        )

        # Add the number of patches to the metadata
        root._v_attrs.n_items = root.imgs.shape[0]
        h5.close()


    @classmethod
    def add2db(cls,
               fname: str,
               img_dir: str,
               mask_dir: str,
               **kwargs) -> Path:
        """
        Add data to a previously created hdf5 database. This uses the 
        metadata of the existing h5 db to sort out the required params 
        for writing the file. kwargs can be used to modify the params 
        for the data writing

        Args:
        ---------
            fname (str):
                Path to the existing HDF5 database
            img_dir (str):
                Directory of the images
            mask_dir (str):
                Directory of the masks

        Returns:
        ---------
            Path: path to the HDF5 database
        """
        fname = Path(fname)
        h5 = tb.open_file(fname.as_posix(), mode="a")
        root = h5.root
        
        # set kwargs for the writer class
        _kwargs = {
            "save_dir": root._v_attrs.save_dir,
            "file_name": root._v_attrs.file_name,
            "classes": root._v_attrs.classes,
            "patch_shape": root._v_attrs.patch_shape,
            "stride_size": root._v_attrs.stride_size,
            "n_copies": root._v_attrs.n_copies,
            "rigid_augs_and_crop": root._v_attrs.rigid_augs,
            "crop_shape": root._v_attrs.crop_shape,
            "chunk_size": root._v_attrs.chunk_size 
        }

        # change the default _kwargs if kwargs were given 
        if kwargs:
            for key, val in kwargs.items():
                _kwargs[key] = val
        
        c = cls(
            img_dir=img_dir,
            mask_dir=mask_dir,
            **_kwargs
        )

        c._write_files(h5, img_dir, mask_dir)
        return fname
 
    def write2db(self) -> Path:
        """
        Creates a HDF5 db and writes the images, masks and metadata

        Returns:
        ----------
            Path: path to the HDF5 database
        """
        self.create_dir(self.save_dir)
        fname = Path(self.save_dir / self.file_name).with_suffix(".h5")

        h5 = tb.open_file(fname.as_posix(), mode="w")
        root = h5.root

        # save some params as metadata
        root._v_attrs.stride_size = self.stride_size
        root._v_attrs.patch_shape = self.patch_shape
        root._v_attrs.n_copies = self.n_copies
        root._v_attrs.chunk_size = self.chunk_size
        root._v_attrs.rigid_augs = self.rigid_augs_and_crop
        root._v_attrs.crop_shape = self.crop_shape
        root._v_attrs.img_dir = self.img_dir.as_posix()
        root._v_attrs.mask_dir = self.mask_dir.as_posix()
        root._v_attrs.save_dir = self.save_dir.as_posix()
        root._v_attrs.file_name = self.file_name
        root._v_attrs.classes = self.classes
        root._v_attrs.sem_classes = self.sem_classes

        # set patch width and height
        ph, pw = self.crop_shape
        if not self.rigid_augs_and_crop and self.patch_shape is not None:
            ph, pw = self.patch_shape

        h5.create_earray(
            where=root, 
            name="imgs", 
            atom=tb.UInt8Atom(),  
            shape=np.append([0], (ph, pw, 3)), 
            chunkshape=np.append([self.chunk_size], (ph, pw, 3)),
            filters=tb.Filters(complevel=5, complib="blosc:lz4")
        )

        h5.create_earray(
            where=root, 
            name="insts", 
            atom=tb.Int32Atom(),  
            shape=np.append([0], (ph, pw)), 
            chunkshape=np.append([self.chunk_size], (ph, pw)),
            filters=tb.Filters(complevel=5, complib="blosc:lz4")
        )

        h5.create_earray(
            where=root, 
            name="types", 
            atom=tb.Int32Atom(),  
            shape=np.append([0], (ph, pw)), 
            chunkshape=np.append([self.chunk_size], (ph, pw)),
            filters=tb.Filters(complevel=5, complib="blosc:lz4")
        )

        h5.create_earray(
            where=root, 
            name="areas", 
            atom=tb.Int32Atom(),  
            shape=np.append([0], (ph, pw)), 
            chunkshape=np.append([self.chunk_size], (ph, pw)),
            filters=tb.Filters(complevel=5, complib="blosc:lz4")
        )

        # pixels per classes
        h5.create_carray(
            where=root, 
            name="npixels", 
            atom=tb.Int32Atom(), 
            shape=(1, len(self.classes))
        )

        h5.create_carray(
            where=root, 
            name="dataset_mean", 
            atom=tb.Float32Atom(), 
            shape=(1, 3)
        )

        h5.create_carray(
            where=root, 
            name="dataset_std", 
            atom=tb.Float32Atom(), 
            shape=(1, 3)
        )

        # total number of pixels in data
        h5.create_carray(
            where=root, 
            name="pixel_num", 
            atom=tb.Float32Atom(), 
            shape=(1, ),
        )

        h5.create_carray(
            where=root, 
            name="channel_sum", 
            atom=tb.Float32Atom(), 
            shape=(1, 3),
        )

        h5.create_carray(
            where=root, 
            name="channel_sum_sq", 
            atom=tb.Float32Atom(), 
            shape=(1, 3),
        )

        self._write_files(h5, self.img_dir, self.mask_dir)
        return fname
