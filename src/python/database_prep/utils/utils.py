import math
import tables
import cv2
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image
from pathlib import Path
from sklearn.model_selection import train_test_split

# Patch Extractor from hover-net repo
# https://github.com/vqdang/hover_net/blob/master/src/misc/patch_extractor.py
#####
class PatchExtractor(object):
    """
    Extractor to generate patches with or without padding.
    Turn on debug mode to see how it is done.
    Args:
        x         : input image, should be of shape HWC
        win_size  : a tuple of (h, w)
        step_size : a tuple of (h, w)
        debug     : flag to see how it is done
    Return:
        a list of sub patches, each patch has dtype same as x
    Examples:
        >>> xtractor = PatchExtractor((450, 450), (120, 120))
        >>> img = np.full([1200, 1200, 3], 255, np.uint8)
        >>> patches = xtractor.extract(img, 'mirror')
    """
    def __init__(self, win_size, step_size, debug=False):

        self.patch_type = 'mirror'
        self.win_size  = win_size
        self.step_size = step_size
        self.debug   = debug
        self.counter = 0

    def __get_patch(self, x, ptx):
        pty = (ptx[0]+self.win_size[0],
               ptx[1]+self.win_size[1])
        win = x[ptx[0]:pty[0], 
                ptx[1]:pty[1]]
        
        assert win.shape[0] == self.win_size[0] and \
               win.shape[1] == self.win_size[1],    \
               '[BUG] Incorrect Patch Size {0}'.format(win.shape)
        
        return win
    
    def __extract_valid(self, x):
        """
        Extracted patches without padding, only work in case win_size > step_size
        
        Note: to deal with the remaining portions which are at the boundary a.k.a
        those which do not fit when slide left->right, top->bottom), we flip 
        the sliding direction then extract 1 patch starting from right / bottom edge. 
        There will be 1 additional patch extracted at the bottom-right corner
        Args:
            x         : input image, should be of shape HWC
            win_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x
        """

        im_h = x.shape[0] 
        im_w = x.shape[1]

        def extract_infos(length, win_size, step_size):
            flag = (length - win_size) % step_size != 0
            last_step = math.floor((length - win_size) / step_size)
            last_step = (last_step + 1) * step_size
            return flag, last_step

        h_flag, h_last = extract_infos(im_h, self.win_size[0], self.step_size[0])
        w_flag, w_last = extract_infos(im_w, self.win_size[1], self.step_size[1])    

        sub_patches = []
        #### Deal with valid block
        for row in range(0, h_last, self.step_size[0]):
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        #### Deal with edge case
        if h_flag:
            row = im_h - self.win_size[0]
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        if w_flag:
            col = im_w - self.win_size[1]
            for row in range(0, h_last, self.step_size[0]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        if h_flag and w_flag:
            ptx = (im_h - self.win_size[0], im_w - self.win_size[1])
            win = self.__get_patch(x, ptx)
            sub_patches.append(win)  
        return np.asarray(sub_patches)
    
    def __extract_mirror(self, x):
        """
        Extracted patches with mirror padding the boundary such that the 
        central region of each patch is always within the orginal (non-padded)
        image while all patches' central region cover the whole orginal image
        Args:
            x         : input image, should be of shape HWC
            win_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x
        """

        diff_h = self.win_size[0] - self.step_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = self.win_size[1] - self.step_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        pad_type = 'constant' if self.debug else 'reflect'
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.__extract_valid(x)
        return sub_patches

    def extract(self, x, patch_type):
        patch_type = patch_type.lower()
        self.patch_type = patch_type
        if patch_type == 'valid':
            return self.__extract_valid(x)
        elif patch_type == 'mirror':
            return self.__extract_mirror(x)
        else:
            assert False, 'Unknown Patch Type [%s]' % patch_type
        return
#####


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
        self.stride_size = conf['stride_size']
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
        
        # self.mirror_pad_size = self.patch_size//2
        self.n_classes = len(self.classes)
        self.xtractor = PatchExtractor(
            (self.patch_size, self.patch_size),
            (self.stride_size,self.stride_size)
        )
        
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
        Split training set into trainingagen(self, batch_size, mode='train', view=False):
        if mode == 'train':
            augmentors = self.get_train_augmentors(
                                            self.train_input_shape,
                                            self.train_mask_shape,
                                            view)
            data_files = get_files(self.train_dir, self.data_ext)
            data_generator = loader.train_generator
            nr_procs = self.nr_procs_train
        else: and validation set (correct way to train).
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
        # OLD, USE HOVERNET PATCH EXTRACTOR INSTEAD
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
                # patches = self._extract_patches(io)
                patches = self.xtractor.extract(io, 'mirror')
        
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
                # patches = self._extract_patches(io)
                patches = self.xtractor.extract(io, 'mirror')
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
            
        #patches = self._extract_patches(io)
        patches = self.xtractor.extract(io, 'mirror')
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
            imm = plt.imshow(patches[idx].astype('uint8'))
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
    plt.show()
    return np.unique(mask)

            