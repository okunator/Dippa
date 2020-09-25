import tables
import cv2
import albumentations as A
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from src.utils.patch_extractor import PatchExtractor
from src.utils.file_manager import ProjectFileManager
from src.img_processing.augmentations import rigid_transforms, random_crop, compose


class PatchWriter(ProjectFileManager):
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
        self.xtractor = PatchExtractor(
            (self.patch_size, self.patch_size), 
            (self.stride_size, self.stride_size)
        )


        
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

            
    def __rigid_augs_and_crop(self, im: np.ndarray, mask: np.ndarray):
        """
        Do rigid augmentations and crop the patch to the size of the input_size.
        The database access is several times faster if the patches are smaller.
        """
        transforms = compose([
            rigid_transforms(),
            random_crop(self.input_size)
        ])
        
        return transforms(image=im, mask=mask)

    
    def __augment_patches(self, 
                          patches_im: np.ndarray, 
                          patches_mask: np.ndarray,
                          do_overlay: bool = False) -> None:
        
        imgs = []
        masks = []
        overlays = []
        for i in range(patches_im.shape[0]):
            cropped_patches = self.__rigid_augs_and_crop(im=patches_im[i], mask=patches_mask[i])
            image = cropped_patches["image"]
            mask = cropped_patches["mask"]
            imgs.append(image)
            masks.append(mask)
            
            if do_overlay:
                overlay = np.where(mask[..., None], image, 0)
                overlays.append(overlay)
                
        return np.array(imgs), np.array(masks), np.array(overlays)
    
            
    def __create_db(self, phase: str) -> None:
        storage = {}
        imgs = self.data_folds[phase]["img"]
        masks = self.data_folds[phase]["mask"]
        
        # Choose the image size to be saved to db
        if self.crop_to_input:
            size = self.input_size
        else:
            size = self.patch_size
            
        block_shape = {
            "img": np.array((size, size, 3)),
            "mask": np.array((size, size))
        }
        
        totals = np.zeros((2, len(self.classes)))
        totals[0, :] = list(self.classes.values())
        fn = f"patch{size}_{phase}_{self.dataset}.pytable"
        hdf5 = tables.open_file(self.database_dir.joinpath(fn).as_posix(), mode="w")
        
        # Create empty earray for he filename
        storage["filename"] = hdf5.create_earray(
            hdf5.root, 
            "filename", 
            tables.StringAtom(itemsize=255),
            (0,)
        )
        
        # Create distinct earrays for ground truths and images
        for img_type in ["img", "mask"]:
            # Uint8 for images Uint16 for instance labels
            if img_type == "img":
                img_dtype = tables.UInt8Atom()
            else:
                img_dtype = tables.Int16Atom()
            
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
                
            im = self.read_img(img_path)
            mask = self.read_mask(mask_path)
            mask = mask[..., None].repeat(3, axis=2)

            for j, key in enumerate(self.classes.values()): 
                totals[1, j] += sum(sum(mask[:, :, 0] == key))
            
            if self.dataset == "pannuke":
                # No need for rigid augmentations for pannuke
                patches_im = im[None, ...]
                patches_mask = mask[None, :, :, 0]
            else:
                patches_im = self.xtractor.extract(im, "mirror")
                patches_mask = self.xtractor.extract(mask, "mirror")
                patches_mask = patches_mask[..., 0].squeeze()
                
                if self.crop_to_input:
                    patches_im, patches_mask, _ = self.__augment_patches(patches_im, patches_mask)
                    
            storage["img"].append(patches_im)
            storage["mask"].append(patches_mask)
            
        
        # lastly  store the number of non-zero pixels to a carray
        # (in case we want to train with class weights)
        n_pixels = hdf5.create_carray(
            hdf5.root, 
            "numpixels", 
            tables.Atom.from_dtype(totals.dtype), 
            totals.shape
        )
        
        n_pixels[:] = totals
        hdf5.close()
        
    
    def __viz_pannuke_patches(self, phase: str, img_type: str) -> None:
        img_paths = self.data_folds[phase]["img"]
        mask_paths = self.data_folds[phase]["mask"]
        idxs = np.random.randint(low = 0, high=len(img_paths), size=25)
        
        fig, ax = plt.subplots(5 , 5, figsize=(50,50))
        ax = ax.flatten()
        for i, idx in enumerate(idxs):
            if img_type == "img":
                io = self.read_img(img_paths[idx])
            elif img_type == "mask":
                io = self.read_mask(mask_paths[idx])
            else:
                im = self.read_img(img_paths[idx])
                mask = self.read_mask(mask_paths[idx])
                io = np.where(mask[..., None], im, 0)
            ax[i].imshow(io)

    
    
    def viz_patches_example(self, 
                            index: int = 1, 
                            img_type: str ="img", 
                            phase: str = "train") -> np.ndarray:
        """
        A visualization of what the patches look like that are written to the hdf5 database.
        Args:
            index (int): the index number of the filepath in the list of files belonging to a specific dataset
            img_type (str): whether to use masks or images or a juxtaposition of both
            phase (str): One of ('train', 'test', 'valid').
            
        Returns:
            the shape of the patched array
        """
        # This is fugly as heck... 
        assert img_type in ("img", "mask", "overlay") 
        assert phase in ("train", "test", "valid")            
        
        if self.dataset == "pannuke":
            self.__viz_pannuke_patches(phase, img_type)
        else:
            img_path = self.data_folds[phase]["img"][index]
            mask_path = self.data_folds[phase]["mask"][index]
            im = self.read_img(img_path)
            mask = self.read_mask(mask_path)
            overlay = np.where(mask[..., None], im, 0)
            mask = mask[..., None].repeat(3, axis=2)
            patches_im = self.xtractor.extract(im, "mirror")
            patches_mask = self.xtractor.extract(mask, "mirror")
            patches_mask = patches_mask[..., 0].squeeze()

            if self.crop_to_input:
                patches_im, patches_mask, patches_overlay = self.__augment_patches(
                    patches_im, patches_mask, do_overlay=True
                )
            else:
                patches_overlay = self.xtractor.extract(overlay, "mirror")

            if img_type == "img":
                patches = patches_im
            elif img_type == "mask":
                patches = patches_mask
            else:
                patches = patches_overlay

            fignum=200
            low=0
            high=len(patches)

            # Visualize
            fig_patches = plt.figure(fignum, figsize=(35,35))
            pmin, pmax = patches.min(), patches.max()
            dims = np.ceil(np.sqrt(high - low))
            for idx in range(high - low):
                spl = plt.subplot(dims, dims, idx + 1)
                ax = plt.axis("off")
                imm = plt.imshow(patches[idx].astype("uint8"))
                cl = plt.clim(pmin, pmax)
            plt.show()
            return patches.shape


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
            
            

def visualize_db_patches(path: str, index: int) -> np.ndarray:
    """
    This function opens the hdf5 file and queries for an image and a mask with an index and plots these.
    Args:
        path (str): path to the hdf5 database
        index (int): index for the patch array in the hdf5 db
    
    Returns:
        the unique labels of the mask
    """
    matplotlib.rcParams.update({"font.size": 22})
    with tables.open_file(path,"r") as db:
        img = db.root.img
        maskd = db.root.mask
        im = img[index, ...]
        mask = maskd[index, ...]
        
    image_overlayed = np.where(mask[..., None], im, 0)
    fig, ax = plt.subplots(1,3, figsize=(45,45))
    ax[0].imshow(im)
    ax[1].imshow(mask)
    ax[2].imshow(image_overlayed)
    ax[0].title.set_text("ORIGINAL")
    ax[1].title.set_text("MASK")
    ax[2].title.set_text("SUPERIMPOSED")
    plt.show()
    return np.unique(mask)


# How to get correct types of instances example
#t1 = gt['type_map'] == 4
#inst_map = np.copy(gt['inst_map'])
#inst_map[~t1] = 0