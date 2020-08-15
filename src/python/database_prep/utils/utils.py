import tables
import numpy as np
from pathlib import Path

class DB_writer(object):
    """
    This class can be used to create the databases where the torch
    DataLoader loads the data in the training loop
    """
    def __init__(self, conf, verbose=True):
        assert conf['dataset'] in ("kumar", "consep", "pannuke")
        
        self.dataset = conf['dataset']
        self.patch_size = conf['patch_size']
        self.classes = conf['classes']
        self.db_name = conf['db_name']
        self.phases = conf['phases']
        
        self.train_mask_dir = Path(conf['train_mask_dir'])
        self.valid_mask_dir = Path(conf['valid_mask_dir'])
        self.test_mask_dir = Path(conf['test_mask_dir'])
        self.train_img_dir = Path(conf['train_img_dir'])
        self.valid_img_dir = Path(conf['valid_img_dir'])
        self.test_img_dir = Path(conf['test_img_dir'])
        self.outdir = Path(conf['outdir'])
        
        # All masks need to be in .mat format
        self.train_masks = sorted([x.as_posix() for x in self.train_dir.glob("*.mat")])
        self.valid_masks = sorted([x.as_posix() for x in self.valid_dir.glob("*.mat")])
        self.test_masks = sorted([x.as_posix() for x in self.test_dir.glob("*.mat")])
        self.db_path = self.outdir.joinpath(self.db_name)
        
        
        # Dataset directories
        self.data_dirs = {
            'kumar':{
                'kumar_im':"../../../../datasets/Nucleisegmentation-Kumar/train/Images",
                'kumar_gt':"../../../../datasets/Nucleisegmentation-Kumar/train/Labels",
                'gt_sfx':'/*.mat',
                'im_sfx':'/*.tif'
            },
            'consep':{
                'consep_im':"../../../../datasets/Nucleisegmentation-CoNSeP/test/Images",
                'consep_gt':"../../../../datasets/Nucleisegmentation-CoNSeP/test/Labels",
                'gt_sfx':'/*.mat',
                'im_sfx':'/*.png'
            },
            'pannuke': {
                'pannuke_im':"../../../../datasets/Nucleisegmentation-PanNuke/test/Images",
                'pannuke_gt':"../../../../datasets/Nucleisegmentation-PanNuke/test/Labels",
                'gt_sfx':'/*.mat',
                'im_sfx':'/*.png'
            }
        }
        
        self.stride_size = self.patch_size//2
        self.mirror_pad_size = self.patch_size//2
        self.n_classes = len(self.classes)
        
        
    def _read_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    
    def _read_mask(self, path):
        mask = scipy.io.loadmat(path)
        return mask['inst_map']
    
    
    def _extract_patches(self, io):
        # add reflect padding 
        pad = (self.mirror_pad_size, self.mirror_pad_size)
        io = np.pad(io, [pad, pad, (0, 0)], mode="reflect")
        
        # convert input image into overlapping tiles
        # size is ntile_row x ntile_col x 1 x patch_size x patch_size x 3
        io_arr = sklearn.feature_extraction.image.extract_patches(
            io,
            (self.patch_size, self.patch_size, 3),
            stride_size
        )
        # resize it into a ntile_total x patch_size x patch_size x 3
        io_arr = io_arr.reshape(-1, self.patch_size, self.patch_size, 3)
        return io_arr
      
        
    def _create_db(self, phase, imgs, masks):
        storage = {}
        block_shape = {
            'img': np.array((self.patch_size, self.patch_size, 3)),
            'mask': np.array((self.patch_size, self.patch_size))
        }
        
        totals = np.zeros((2, self.n_classes))
        totals[0, :] = list(self.classes.items())
        hdf5 = tables.open_file(f"{}", mode='w')
        
        # Create empty earray for he filename
        storage["filename"] = hdf5.create_earray(
            hdf5.root, 
            'filename', 
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
                filters=tables.Filters(complevel=6, complib='zlib')
            )
            
        for img_path in imgs:
            io = _read_img(img_path)
            if self.dataset == 'pannuke':
                patches = io
            else:
                patches = _extract_patches(io)
            storage['img'].append(patches)
        
        for mask_path in masks:
            io = _read_mask(mask_path)
            io = io[..., None].repeat(3, axis=2)
            
            # Get the number of non-zero pixels in the mask
            for j, key in enumerate(self.classes.items()): 
                totals[1, j] += sum(sum(io[..., 0] == key))
                
            patches = _extract_patches(io)
            storage['mask'].append(patches[..., 0].squeeze())
            
        # lastly, we store the number of non-zero pixels to a carray
        # (in case we want to train with class weights)
        npixels = hdf5.create_carray(
            hdf5.root, 
            'numpixels', 
            tables.Atom.from_dtype(totals.dtype), 
            totals.shape
        )
        
        npixels[:] = totals
        hdf5.close()
        
    def write_dbs(self, dataset):
        pass
        # for phase in self.phases:
        #     pass
            
            
            
        
            
            