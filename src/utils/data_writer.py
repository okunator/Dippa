import tables
import cv2
import albumentations as A
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_extraction.image
from pathlib import Path
from typing import List, Dict, Tuple
from omegaconf import DictConfig
from src.utils.patch_extractor import PatchExtractor
from src.utils.file_manager import ProjectFileManager
from src.img_processing.viz_utils import viz_patches
from src.img_processing.process_utils import overlays
from src.img_processing.augmentations import rigid_augs_and_crop


# Lots of spadghetti b/c different datasets need different ad hoc processing to get stuff aligned
class PatchWriter(ProjectFileManager, PatchExtractor):
    def __init__(self, 
                 dataset_args: DictConfig,
                 experiment_args: DictConfig,
                 patching_args: DictConfig,
                 **kwargs: Dict,) -> None:
        """
        A class to patch input images and to write them to either hdf5 tables
        or .npy files that are are used in the training of the networks used in this project. 
        The torch dataset class is written to read from the files created by this class.
        
            Args:
                dataset_args (DictConfig): omegaconfig DictConfig specifying arguments
                                           related to the dataset that is being used.
                                           Check config.py for more info
                experiment_args (DictConfig): omegaconfig DictConfig specifying arguments
                                              that are used for creating result folders
                                              files. Check config.py for more info
                patching_args (DictConfig): omegaconfig DictConfig specifying arguments
                                            that are used for patching input images.
                                            Check config.py for more info
                
        """
        super(PatchWriter, self).__init__(dataset_args, experiment_args)
        self.patch_size = patching_args.patch_size
        self.stride_size = patching_args.stride_size
        self.input_size = patching_args.model_input_size
        self.verbose = patching_args.verbose
        self.crop_to_input = patching_args.crop_to_input       
        self.__validate_patch_args(self.patch_size, self.stride_size, self.input_size)

    @classmethod
    def from_conf(cls, conf: DictConfig):
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        patching_args = conf.patching_args
        
        return cls(
            dataset_args,
            experiment_args,
            patching_args
        )

    @property
    def save_size(self):
        """Size of the patch to be saved to db"""
        if self.crop_to_input:
            size = self.input_size
        else:
            size = self.patch_size
        return size
    
    def __validate_patch_args(self, patch_size: int, stride_size: int, input_size: int) -> None:
        shapes = [self.read_img(f).shape
                  for phase in self.phases 
                  for key, val in self.data_folds[phase].items()
                  for f in val if key =='img']
        
        assert stride_size <= patch_size, "stride_size must be <= patch_size"
        assert input_size <= patch_size, "input_size must be <= patch_size"
        assert all(s[0] >= patch_size and s[1] >= patch_size for s in shapes), (
            f"height or width of given imgs is < patch_size ({patch_size}). Check your image shapes."
        )

    def __augment_patches(self, 
                          patches_im: np.ndarray, 
                          patches_mask: np.ndarray, 
                          do_overlay: bool = False) -> Tuple[np.ndarray]:
        
        imgs, inst_maps, type_maps, overlays = [], [], [], []
        for i in range(patches_im.shape[0]):
            cropped_patches = rigid_augs_and_crop(
                image=patches_im[i],
                mask=patches_mask[i],
                input_size=self.save_size
            )
            imgs.append(cropped_patches["image"])
            inst_maps.append(cropped_patches["mask"][..., 0])
            type_maps.append(cropped_patches["mask"][..., 1])

            if do_overlay:
                image = cropped_patches["image"]
                inst_map = cropped_patches["inst_map"]
                overlay = np.where(inst_map[..., None], image, 0)
                overlays.append(overlay)
      
        return np.array(imgs), np.array(inst_maps), np.array(type_maps), np.array(overlays)

    def __extract_patches(self, 
                          phase: str,
                          img_path: str,
                          mask_path: str) -> np.ndarray:

        im = self.read_img(img_path)
        inst_map = self.read_mask(mask_path, key="inst_map")
        type_map = self.read_mask(mask_path, key="type_map")
        full_data = np.concatenate([im, inst_map[..., None], type_map[..., None]], axis=-1)

        # compute number of pixels in each class
        totals = np.zeros((2, len(self.classes)))
        totals[0, :] = list(self.classes.values())
        for j, val in enumerate(self.classes.values()):
            totals[1, j] += sum(sum(type_map == val))

        if phase is not "test":
            patches = self.extract_mirror(
                full_data, 
                (self.patch_size, self.patch_size),
                (self.stride_size, self.stride_size)
            )
        else:
            patches = self.extract_test_patches(full_data, self.input_size)

        patches_im = patches[..., :3].astype("uint8")
        patches_mask = patches[..., 3:]

        # no augmentations for test patches
        if self.crop_to_input and phase is not "test":
            patches_im, patches_imap, patches_tmap, _ = self.__augment_patches(patches_im, patches_mask)
        else:
            patches_imap = patches_mask[..., 0]
            patches_tmap = patches_mask[..., 1]

        return patches_im, patches_imap, patches_tmap, totals
         
    def __create_db(self, phase: str) -> None:
        storage = {}
        imgs = self.data_folds[phase]["img"]
        masks = self.data_folds[phase]["mask"]
                    
        block_shape = {
            "img": np.array((self.save_size, self.save_size, 3)),
            "inst_map": np.array((self.save_size, self.save_size)),
            "type_map": np.array((self.save_size, self.save_size))
        }
        
        fn = f"patch{self.save_size}_{phase}_{self.dataset}.pytable"
        hdf5 = tables.open_file(self.database_dir.joinpath(fn).as_posix(), mode="w")
        
        # Create empty earray for he filename
        storage["filename"] = hdf5.create_earray(
            hdf5.root, 
            "filename", 
            tables.StringAtom(itemsize=255),
            (0,)
        )
        
        # Create distinct earrays for ground truths and images
        for img_type in self.img_types:
            # Uint8 for images Uint16 for instance labels
            if img_type == "img":
                img_dtype = tables.UInt8Atom()
            elif img_type == "inst_map":
                img_dtype = tables.Int32Atom()
            elif img_type == "type_map":
                img_dtype = tables.Int32Atom()
            
            storage[img_type] = hdf5.create_earray(
                hdf5.root, 
                img_type, 
                img_dtype,  
                shape=np.append([0], block_shape[img_type]), 
                chunkshape=np.append([1], block_shape[img_type]),
                filters=tables.Filters(complevel=6, complib="zlib")
            )

        # Do the patching for the files
        for img_path, mask_path in zip(imgs, masks):
            if self.verbose:
                print(f"patching {img_path.split('/')[-1]} and writing to db")
                print(f"patching {mask_path.split('/')[-1]} and writing patches to db")

            ims, imaps, tmaps, totals = self.__extract_patches(phase, img_path, mask_path)
            # TODO: add pre processing functions to patches
            storage["img"].append(ims)
            storage["inst_map"].append(imaps)
            storage["type_map"].append(tmaps)
            
        # lastly store the number of non-zero pixels to a carray
        # (in case we want to train with class weights)
        n_pixels = hdf5.create_carray(
            hdf5.root, 
            "numpixels", 
            tables.Atom.from_dtype(totals.dtype), 
            totals.shape
        )
        
        n_pixels[:] = totals
        hdf5.close()

    def write_dbs(self) -> None:
        """
        Writes the hdf5 databases to correct folders.
        """
        self.create_dir(self.database_dir)
        self.remove_existing_files(self.database_dir)
        for phase in self.phases:
            if self.verbose:
                print("phase: ", phase)
            self.__create_db(phase)
                
    def write_npys(self) -> None:
        # TODO
        pass

    def viz_patched_img(self,
                        phase: str,
                        img_type: str,
                        index: int) -> Tuple:
        """
        Visualize an image in patches.
        Args:
            phase (str): one of ('train', 'valid', 'test')
            img_type (str): one of ("img", "inst_map", "type_map", "overlay")
            index (int): index number for file path list
        Returns:
            the shape of patches.
        """
        assert img_type in ("img", "inst_map", "type_map", "overlay")
        assert phase in ("train", "test", "valid")

        img_path = self.data_folds[phase]["img"][index]
        mask_path = self.data_folds[phase]["mask"][index]
        ims, imaps, tmaps, _ = self.__extract_patches(phase, img_path, mask_path)

        if img_type == "img":
            patches = ims
        elif img_type == "inst_map":
            patches = imaps
        elif img_type == "type_map":
            patches = tmaps
        elif img_type == "overlay":
            patches = overlays(ims, imaps)

        viz_patches(patches)
            
    def viz_patch_from_db(self, phase: str, index: int) -> Tuple[np.ndarray]:
        """
        Opens the hdf5 file and queries for an image and a mask with an index and plots these.
        Args:
            phase (str): one of ('train', 'valid', 'test')
            index (int): index for the patch array in the hdf5 db
        Returns:
            the unique labels of the mask
        """
        db_path = list(self.databases[phase].values())[0]
        im_patch, inst_patch, type_patch = self.read_hdf5_patch(db_path.as_posix(), index)

        matplotlib.rcParams.update({"font.size": 22})
        fig, ax = plt.subplots(1, 4, figsize=(45,45))
        ax[0].imshow(im_patch)
        ax[1].imshow(inst_patch)
        ax[2].imshow(type_patch)
        ax[3].imshow(overlays(im_patch, inst_patch))
        ax[0].title.set_text("IMG")
        ax[1].title.set_text("INST MAP")
        ax[2].title.set_text("TYPE MAP")
        ax[3].title.set_text("OVERLAY")
        plt.show()
        return np.unique(inst_patch), np.unique(type_patch)

