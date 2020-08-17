import tables
import cv2
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image
from pathlib import Path
from sklearn.model_selection import train_test_split

class DB_writer:
    """
    This class can be used to create the databases where the torch
    DataLoader loads the data in the training loop
    """
    def __init__(self, conf, verbose=True):
        assert conf["dataset"] in ("kumar", "consep", "pannuke", "other")
        
        self.verbose = verbose
        self.dataset = conf["dataset"]
        self.patch_size = conf["patch_size"]
        self.classes = conf["binary_classes"]
        self.phases = conf["phases"]
        self.data_dirs = conf['data_dirs']
        self.database_root = Path(conf['database_root_dir'])
        self.database_dir = self.database_root.joinpath(f"{self.dataset}/patch_{self.patch_size}")
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        train_imgs = self._get_files(self.data_dirs[self.dataset]["train_im"])
        train_masks = self._get_files(self.data_dirs[self.dataset]["train_gt"])
        test_imgs = self._get_files(self.data_dirs[self.dataset]["test_im"])
        test_masks = self._get_files(self.data_dirs[self.dataset]["test_gt"])
        
        # optionally create validation set from training set if dataset is not pan-Nuke
        if "valid" in self.phases and self.dataset == "pannuke":
            valid_imgs = self._get_files(self.data_dirs[self.dataset]["valid_im"])
            valid_masks = self._get_files(self.data_dirs[self.dataset]["valid_gt"])   
        elif "valid" in self.phases:
            train_imgs, train_masks, valid_imgs, valid_masks = self._split_training_set(train_imgs, train_masks)
        else:
            valid_imgs = None
            valid_masks = None
            
        self.data_folds = {
            "train":{"img":train_imgs, "mask":train_masks},
            "valid":{"img":valid_imgs, "mask":valid_masks},
            "test":{"img":test_imgs, "mask":test_masks},
        }
        
        self.stride_size = self.patch_size//2
        self.mirror_pad_size = self.patch_size//2
        self.n_classes = len(self.classes)
    
        
    def _get_suffix(self, directory):
        assert all([file.suffix for file in directory.iterdir()]), "All files should be in same format"
        return [file.suffix for file in directory.iterdir()][0]
        
        
    def _get_files(self, directory):
        directory = Path(directory)
        assert directory.is_dir(), "Provided directory for image files is not a directory."
        assert all([file.exists() for file in directory.iterdir()]), "Some files do not exist in dir"
        file_suffix = self._get_suffix(directory)
        return sorted([x.as_posix() for x in directory.glob(f"*{file_suffix}")])
    
    
    def _remove_existing_files(self, directory):
        for file in directory.iterdir():
            file.unlink()
    
    
    def _read_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    
    def _read_mask(self, path):
        mask = scipy.io.loadmat(path)
        return mask["inst_map"]
    
    
    def _split_training_set(self, train_imgs, train_masks, seed=42, size=0.2):
        """
        Split training set into training and validation set (correct way to train).
        Not needed for pan-Nuke (has 3 folds). This might affect training accuracy
        if training set is small. 
        """
        indices = np.arange(len(train_imgs))
        train_indices, valid_indices = train_test_split(
            indices, test_size=size, random_state=42, shuffle=True
        )
                      
        np_train_imgs = np.array(train_imgs)
        np_train_masks = np.array(train_masks)
        train_imgs = sorted(np_train_imgs[train_indices].tolist())
        train_masks = sorted(np_train_masks[train_indices].tolist())
        valid_imgs = sorted(np_train_imgs[valid_indices].tolist())
        valid_masks = sorted(np_train_masks[valid_indices].tolist())
        return train_imgs, train_masks, valid_imgs, valid_masks
        
    
    def _extract_patches(self, io):
        # add reflect padding 
        pad = (self.mirror_pad_size, self.mirror_pad_size)
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
      
        
    def _create_db(self, phase):
        storage = {}
        imgs = self.data_folds[phase]["img"]
        masks = self.data_folds[phase]["mask"]
        block_shape = {
            "img": np.array((self.patch_size, self.patch_size, 3)),
            "mask": np.array((self.patch_size, self.patch_size))
        }
        
        totals = np.zeros((2, self.n_classes))
        totals[0, :] = list(self.classes.values())
        root = self.database_root.as_posix()
        
        fn = f"{phase}_{self.dataset}.pytable"
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
                img_dtype = tables.UInt16Atom()
            
            storage[img_type] = hdf5.create_earray(
                hdf5.root, 
                img_type, 
                img_dtype,  
                shape=np.append([0], block_shape[img_type]), 
                chunkshape=np.append([1], block_shape[img_type]),
                filters=tables.Filters(complevel=6, complib="zlib")
            )
            
        for img_path in imgs:
            if self.verbose:
                print(f"patching {img_path.split('/')[-1]} and writing to db")
                
            io = self._read_img(img_path)
            
            if self.dataset == "pannuke":
                patches = io[None, ...]
            else:
                patches = self._extract_patches(io)
        
            storage["img"].append(patches)
        
        for mask_path in masks:
            if self.verbose:
                print(f"patching {mask_path.split('/')[-1]} and writing patches to db")
                
            io = self._read_mask(mask_path)
            io = io[..., None].repeat(3, axis=2)
            
            # Get the number of non-zero pixels in the mask
            for j, key in enumerate(self.classes.values()): 
                totals[1, j] += sum(sum(io[:, :, 0] == key))
                
            if self.dataset == "pannuke":
                patches = io[None, :, :, 0]
            else:
                patches = self._extract_patches(io)
                patches = patches[..., 0].squeeze()
            
            storage["mask"].append(patches)
            
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
        
    
    def viz_patches_example(self, index, img_type="img", phase="train"):
        assert self.dataset != "pannuke", "pannuke is patched. Patch extractor is not used for it"
        assert img_type in ("img", "mask") 
        assert phase in ("train", "test", "valid")
        
        img_path = self.data_folds[phase][img_type][index]
        if img_type == "img":
            io = self._read_img(img_path)
        else:
            io = self._read_mask(img_path)
            io = io[..., None].repeat(3, axis=2)
            
        patches = self._extract_patches(io)
        fignum=200
        low=0
        high=len(patches)

        # Visualize
        fig_patches = plt.figure(fignum, figsize=(35,35))
        pmin, pmax = patches.min(), patches.max()
        dims = np.ceil(np.sqrt(high - low))
        for idx in range(high - low):
            spl = plt.subplot(dims, dims, idx + 1)
            ax = plt.axis('off')
            imm = plt.imshow(patches[idx], cmap=matplotlib.cm.gray)
            cl = plt.clim(pmin, pmax)
        plt.show()

    def write_dbs(self):
        self._remove_existing_files(self.database_dir)
        for phase in self.phases:
            if self.verbose:
                print("phase: ", phase)
            self._create_db(phase)
            
            

def visualize_db_patches(path, index):
    matplotlib.rcParams.update({'font.size': 22})
    with tables.open_file(path,'r') as db:
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
    fig.show()
    return np.unique(mask)

            