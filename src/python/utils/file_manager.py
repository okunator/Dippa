import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split


class ProjectFileManager:
    def __init__(self, 
                 dataset : str, 
                 data_dirs : Dict, 
                 database_root : str, 
                 phases : List,
                 **kwargs : Dict) -> None:
        """
        This class is used for managing the files and folders needed in this project. 
        
            Args:
                dataset (str) : one of ('kumar', 'consep', 'pannuke', 'other')
                data_dirs (dict) : dictionary of directories containing masks and images. Keys of this
                                   dict must be the same as ('kumar', 'consep', 'pannuke', 'other')
                database_root_dir (str) : directory where the databases are written
                phases (list) : list of the phases (['train', 'valid', 'test'] or ['train', 'test'])
        """
        
        self.validate_data_args(dataset, data_dirs, phases, database_root)
        self.dataset = dataset
        self.data_dirs = data_dirs
        self.phases = phases
        self.database_root = Path(database_root)
        self.database_dir = self.database_root.joinpath(f"{self.dataset}")
        super().__init__(**kwargs)
        
        # TODO:
        # self.patches_root = Path(patches_root)
        # self.patches_npy_dir = self.patches_root.joinpath(f"{self.dataset}")
        
                        
    @classmethod
    def from_conf(cls, conf):
        dataset = conf['dataset']['args']['dataset']
        data_dirs = conf['paths']['data_dirs']
        database_root = conf['paths']['database_root_dir']
        phases = conf['dataset']['args']['phases']
        
        cls.validate_data_args(dataset, data_dirs, phases, database_root)
        
        return cls(
            dataset,
            data_dirs,
            database_root,
            phases
        )
    
    
    @staticmethod
    def validate_data_args(dataset, data_dirs, phases, database_root):        
        assert dataset in ("kumar", "consep", "pannuke", "other"), f"input dataset: {dataset}"
        assert list(data_dirs.keys()) == ["kumar", "consep", "pannuke", "other"], f"{data_dirs.keys()}"
        assert phases in (['train', 'valid', 'test'], ['train', 'test']), f"{phases}"
        assert Path(database_root).exists(), "config file 'database_root_dir' does not exist"
        
        data_paths_bool = [Path(p).exists() for d in data_dirs.values() for p in d.values()]
        assert any(data_paths_bool), ("None of the config file 'data_dirs' paths exist. "
                                      "Make sure they are created, by runnning these... TODO")
        
        
    @staticmethod
    def remove_existing_files(directory):
        assert Path(directory).is_dir(), f"{directory} is not a folder that can be iterated"
        for item in Path(directory).iterdir():
            # Do not touch the folders inside the directory
            if item.is_file():
                item.unlink()


    @staticmethod      
    def create_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
                
    def __get_suffix(self, directory):
        # Find out the suffix of the image and mask files
        d = Path(directory)
        assert all([file.suffix for file in d.iterdir()]), "All image files should be in same format"
        return [file.suffix for file in directory.iterdir()][0]
        
        
    def __get_files(self, directory):
        # Get the image and mask files from the directory that was provided 
        d = Path(directory)
        assert d.is_dir(), f"Provided directory: {d.as_posix()} for image files is not a directory."
        file_suffix = self.__get_suffix(d)
        return sorted([x.as_posix() for x in d.glob(f"*{file_suffix}")])
    
            
    def __split_training_set(self, train_imgs, train_masks, seed=42, size=0.2):
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
        
        return {
            'train_imgs':train_imgs,
            'train_masks':train_masks,
            'valid_imgs':valid_imgs,
            'valid_masks':valid_masks
        }
    
        
    @property
    def data_folds(self):
        train_imgs = self.__get_files(self.data_dirs[self.dataset]["train_im"])
        train_masks = self.__get_files(self.data_dirs[self.dataset]["train_gt"])
        test_imgs = self.__get_files(self.data_dirs[self.dataset]["test_im"])
        test_masks = self.__get_files(self.data_dirs[self.dataset]["test_gt"])
        
        # optionally create validation set from training set if dataset is not pan-Nuke
        if "valid" in self.phases and self.dataset == "pannuke":
            valid_imgs = self.__get_files(self.data_dirs[self.dataset]["valid_im"])
            valid_masks = self.__get_files(self.data_dirs[self.dataset]["valid_gt"])   
        elif "valid" in self.phases:
            d = self.__split_training_set(train_imgs, train_masks)
            train_imgs = d['train_imgs']
            train_masks = d['train_masks']
            valid_imgs = d['valid_imgs']
            valid_masks = d['valid_masks']
        else:
            valid_imgs = None
            valid_masks = None
            
        return {
            "train":{"img":train_imgs, "mask":train_masks},
            "valid":{"img":valid_imgs, "mask":valid_masks},
            "test":{"img":test_imgs, "mask":test_masks},
        }
    
    
    @property
    def databases(self):
        assert self.database_dir.exists(), ("Database directory not found, Create " 
                                            "the dbs first. Check instructions.")
        def get_input_sizes(paths):
            path_dict = {}
            for path in paths:
                fn = path.as_posix().split('/')[-1]
                size = int(''.join(filter(str.isdigit, fn)))
                path_dict[size] = path
            return path_dict
                    
        if 'valid' in self.phases:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_valid_*"))
            test_dbs = list(self.database_dir.glob("*_test_*"))
        else:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_test_*"))
            
        assert train_dbs, (f"{train_dbs} HDF5 training db not found. Create the dbs first. " 
                            "Check instructions.")
        
        assert valid_dbs, (f"{valid_dbs} HDF5 validation db not found. Create the dbs first. "
                            "Check instructions")
        
        
        
        return {
            "train":get_input_sizes(train_dbs),
            "valid":get_input_sizes(valid_dbs),
            "test":get_input_sizes(test_dbs)
        }
        
        
    # Methods for raw data downloaded from the links provided
    @staticmethod 
    def handle_kumar_xml(path):
        pass
    
    
    @staticmethod
    def to_mat():
        pass
    
    
    @staticmethod
    def to_png():
        pass
    
    
    def convert_imgs(self, format=".png"):
        #assert format in (".png", ".tif")
        pass
    
    def convert_masks(self, format=".mat"):
        # assert format in (".mat", ".npy")
        pass

    
