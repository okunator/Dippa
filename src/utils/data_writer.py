import tables
import cv2
import albumentations as A
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_extraction.image
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split
from patch_extractor import PatchExtractor
from file_manager import ProjectFileManager
from img_processing.augmentations import *


class PatchWriter(ProjectFileManager):
    def __init__(self, 
                 dataset: str,
                 data_dirs: Dict[str, str],
                 database_root: str,
                 experiment_root: str,
                 experiment_version: str,
                 model_name: str,
                 phases: List[str],
                 patch_size: int,
                 stride_size: int,
                 input_size: int,
                 class_dict: Dict[str, str],
                 crop_to_input: bool,
                 verbose: bool,
                 pannuke_folds: Dict[str, str] = {"fold1":"train", "fold2":"valid", "fold3":"test"},
                 **kwargs: Dict,
                 ) -> None:
        """
        This class is used to patch input images and to write them to either hdf5 tables
        or .npy files that are are used in the training of the networks used in this project. 
        The torch dataset class is written to read from the files created by this class.
        
            Args:
                dataset (str) : one of ("kumar", "consep", "pannuke", "other")
                data_dirs (dict) : dictionary of directories containing masks and images. Keys of this
                                   dict must be the same as ("kumar", "consep", "pannuke", "other")
                database_root_dir (str) : directory where the databases are written
                experiment_root (str) : directory where results from a network training 
                                        and inference experiment are written
                experiment_version (str) : a name for the experiment you want to conduct. e.g. 
                                           'FPN_test_pannuke' or 'Unet_consep' etc. This name will be used
                                           for results folder of training and inference results
                model_name (str) : The name of the model used in the experiment. This name will be used
                                   for results folder of training and inference results
                phases (list) : list of the phases (["train", "valid", "test"] or ["train", "test"])     
                patch_size (int) : size of a single image patch saved to an HDF5 db
                stride_size (int) : size of the stride when patching
                input_size (int) : size of the network input patch
                class_dict (Dict) : the dict specifying pixel classes. e.g. {"background":0,"nuclei":1}
                crop_to_input (bool) : Whether to crop the extracted patch to the network
                                       input size before writing to database. Also rigid
                                       augmentations are applied to the patch before cropping
                                       since it is useful to tranform bigger patches to avoid
                                       problems with affine transformations on square matrices.
                                       This might be useful since the acces time will be a lot
                                       faster in the db if the patches are small.
                verbose (bool) : Print files being patched to console
                pannuke_folds (Dict[str, str]) : if dataset == "pannuke", this dict will define what 
                                                 fold is used as train, valid and test folds. Otherwise
                                                 this is ignored.
        """
        super(PatchWriter, self).__init__(dataset, data_dirs, database_root, experiment_root,
                                          experiment_version, model_name, phases, pannuke_folds)
        self.__validate_patch_args(patch_size, stride_size, input_size)
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.input_size = input_size
        self.verbose = verbose
        self.crop_to_input = crop_to_input        
        self.classes = class_dict
        self.xtractor = PatchExtractor((patch_size, patch_size), (stride_size, stride_size))

        
    @classmethod
    def from_conf(cls, conf):
        dataset = conf["dataset"]["args"]["dataset"]
        data_dirs = conf["paths"]["data_dirs"]
        database_root = conf["paths"]["database_root_dir"]
        experiment_root = conf["paths"]["experiment_root_dir"]
        experiment_version = conf["experiment_args"]["experiment_version"]
        model_name = conf["experiment_args"]["model_name"]
        phases = conf["dataset"]["args"]["phases"]
        patch_size = conf["patching_args"]["patch_size"]
        stride_size = conf["patching_args"]["stride_size"]
        input_size = conf["patching_args"]["input_size"]
        class_type = conf["dataset"]["args"]["class_types"]
        class_dict = conf["dataset"]["class_dicts"][class_type] # clumsy
        crop_to_input = conf["patching_args"]["crop_to_input"]
        verbose = conf["patching_args"]["verbose"]
        pannuke_folds = conf["dataset"]["pannuke_folds"]
        
        return cls(
            dataset,
            data_dirs,
            database_root,
            experiment_root,
            experiment_version,
            model_name,
            phases,
            patch_size,
            stride_size,
            input_size,
            class_dict,
            crop_to_input,
            verbose,
            pannuke_folds
        )
    
    
    def __validate_patch_args(self, patch_size, stride_size, input_size):
        shapes = [self.read_img(f).shape
                  for phase in self.phases 
                  for key, val in self.data_folds[phase].items()
                  for f in val if key =='img']
        
        assert stride_size <= patch_size, "stride_size must be <= patch_size"
        assert input_size <= patch_size, "input_size must be <= patch_size"
        assert all(s[0] >= patch_size and s[1] >= patch_size for s in shapes), ("height or width of given "
                                                                                "imgs is < patch_size "
                                                                                f"({patch_size})."
                                                                                " Check your image shapes.")
            
            
    def __extract_patches(self, io):
        # OLD, USE HOVERNET PATCH EXTRACTOR INSTEAD
        # add reflect padding 
        mirror_pad = self.patch_size // 2
        pad = (mirror_pad, mirror_pad)
        io = np.pad(io, [pad, pad, (0, 0)], mode="reflect")
        
        # convert input image into overlapping tiles
        # size is ntile_row x ntile_col x 1 x patch_size x patch_size x 3
        io_arr = sklearn.feature_extraction.image.extract_patches(
            io,
            (self.patch_size, self.patch_size, 3),
            self.stride_size
        )
        # resize it into a ntile_total x patch_size x patch_size x 3
        io_arr = io_arr.reshape(-1, self.patch_size, self.patch_size, 3)
        return io_arr
    
    
    def __rigid_augs_and_crop(self, im, mask):
        """
        Do rigid augmentations and crop the patch to the size of the input_size.
        The database access is several times faster if the patches are smaller.
        """
        transforms = compose([
            rigid_transforms(),
            random_crop(self.input_size)
        ])
        
        return transforms(image=im, mask=mask)

    
    def __augment_patches(self, patches_im, patches_mask, do_overlay=False):
        """
        Run the rigid augmentations for the patches
        """
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
    
            
    def __create_db(self, phase):
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
        
    
    def __viz_pannuke_patches(self, phase, img_type):
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
        
    
    
    def viz_patches_example(self, index=1, img_type="img", phase="train"):
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


    def write_dbs(self):
        self.create_dir(self.database_dir)
        self.remove_existing_files(self.database_dir)
        for phase in self.phases:
            if self.verbose:
                print("phase: ", phase)
            self.__create_db(phase)
            
            
    def write_npys(self):
        # TODO
        pass
            
            

def visualize_db_patches(path, index):
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