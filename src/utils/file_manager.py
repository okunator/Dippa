import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split


class ProjectFileManager:
    def __init__(self, 
                 dataset: str, 
                 data_dirs: Dict, 
                 database_root: str,
                 experiment_root: str,
                 experiment_version: str,
                 model_name: str,
                 phases: List,
                 **kwargs: Dict) -> None:
        """
        This class is used for managing the files and folders needed in this project. 
        
            Args:
                dataset (str) : one of ("kumar", "consep", "pannuke", "other")
                data_dirs (dict) : dictionary of directories containing masks and images. Keys of this
                                   dict must be the one of ("kumar", "consep", "pannuke", "other").
                                   Check config.
                database_root (str) : directory where the databases are written
                experiment_root (str) : directory where results from a network training 
                                        and inference experiment are written
                experiment_version (str) : a name for the experiment you want to conduct. e.g. 
                                           'FPN_test_pannuke' or 'Unet_consep' etc. This name will be used
                                           for results folder of training and inference results
                model_name (str) : The name of the model used in the experiment. This name will be used
                                   for results folder of training and inference results
                phases (list) : list of the phases (["train", "valid", "test"] or ["train", "test"])
        """
        super().__init__(**kwargs)
        self.validate_data_args(dataset, data_dirs, phases, database_root, experiment_root)
        self.dataset = dataset
        self.data_dirs = data_dirs
        self.database_root = database_root
        self.experiment_root = experiment_root
        self.experiment_version = experiment_version
        self.model_name = model_name
        self.phases = phases
        
        # TODO:
        # self.patches_root = Path(patches_root)
        # self.patches_npy_dir = self.patches_root.joinpath(f"{self.dataset}")
        
    
    @classmethod
    def from_conf(cls, conf):
        dataset = conf["dataset"]["args"]["dataset"]
        data_dirs = conf["paths"]["data_dirs"]
        database_root = conf["paths"]["database_root_dir"]
        experiment_root = conf["paths"]["experiment_root_dir"]
        experiment_version = conf["experiment_args"]["experiment_version"]
        model_name = conf["experiment_args"]["model_name"]
        phases = conf["dataset"]["args"]["phases"]
        
        return cls(
            dataset,
            data_dirs,
            database_root,
            experiment_root,
            experiment_version,
            model_name,
            phases
        )
    
    
    @staticmethod
    def validate_data_args(dataset, data_dirs, phases, database_root, experiment_root):        
        assert dataset in ("kumar", "consep", "pannuke", "other"), f"input dataset: {dataset}"
        assert list(data_dirs.keys()) == ["kumar", "consep", "pannuke", "other"], f"{data_dirs.keys()}"
        assert phases in (["train", "valid", "test"], ["train", "test"]), f"{phases}"
        assert Path(database_root).exists(), f"database_root: {database_root} not found. Check config"
        assert Path(experiment_root).exists(), f"experiment_root: {experiment_root} not found. Check config"
        
        data_paths_bool = [Path(p).exists() for d in data_dirs.values() for p in d.values()]
        assert any(data_paths_bool), ("None of the config file 'data_dirs' paths exist. "
                                      "Make sure they are created, by runnning these... TODO")
        
        
    @property
    def database_dir(self):
        return Path(self.database_root).joinpath(f"{self.dataset}")
    
    
    @property
    def experiment_dir(self):
        return Path(self.experiment_root).joinpath(f"{self.model_name}/version_{self.experiment_version}")
        
        
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
            train_imgs = d["train_imgs"]
            train_masks = d["train_masks"]
            valid_imgs = d["valid_imgs"]
            valid_masks = d["valid_masks"]
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
        
        def get_input_sizes(paths):
            # Paths to dbs are named as 'patchsize_dataset'. e.g. 'patch256_kumar.pytable' 
            # This will parse the name and use patchsize as key for a dict
            path_dict = {}
            for path in paths:
                fn = path.as_posix().split("/")[-1]
                size = int("".join(filter(str.isdigit, fn)))
                path_dict[size] = path
            return path_dict
                    
        assert self.database_dir.exists(), ("Database directory not found, Create " 
                                            "the dbs first. Check instructions.")
            
        if "valid" in self.phases:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_valid_*"))
            test_dbs = list(self.database_dir.glob("*_test_*"))
        else:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_test_*")) # test set used for both valid and test
            test_dbs = list(self.database_dir.glob("*_test_*")) # remember to not use the best model w this
            
        assert train_dbs, (f"{train_dbs} HDF5 training db not found. Create the dbs first. " 
                            "Check instructions.")
        
        assert valid_dbs, (f"{valid_dbs} HDF5 validation db not found. Create the dbs first. "
                            "Check instructions")
        
        return {
            "train":get_input_sizes(train_dbs),
            "valid":get_input_sizes(valid_dbs),
            "test":get_input_sizes(test_dbs)
        }
    
        
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
        
    
    def model_checkpoint(self, which='last'):
        assert self.experiment_dir.exists(), "Experiment dir does not exist. Train the model first."
        assert which in ("best", "last"), f"param which: {which} not one of ('best', 'last')"
        
        ckpt = None
        for item in self.experiment_dir.iterdir():
            if item.is_file() and item.suffix == ".ckpt":
                if which == "last" and "last" in str(item) and ckpt is None:
                    ckpt = item
                elif which == "best" and "last" not in str(item) and ckpt is None:
                    ckpt = item
        
        assert ckpt is not None, f"{ckpt} checkpoint is None."
        return ckpt
        
                
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
            "train_imgs":train_imgs,
            "train_masks":train_masks,
            "valid_imgs":valid_imgs,
            "valid_masks":valid_masks
        }
    
                
    # Methods for raw data downloaded from the links provided
    @staticmethod 
    def kumar_xml2mat(path):
        """
        From https://github.com/vqdang/hover_net/blob/master/src/misc/proc_kumar_ann.py
        """
        
        imgs_dir = "../../datasets/Nucleisegmentation-Kumar/train/Images/"
        anns_dir = "../../datasets/Nucleisegmentation-Kumar/train/Annotations_xml/"
        save_dir = "../../datasets/Nucleisegmentation-Kumar/train/Labels/"

        file_list = glob.glob(imgs_dir + '*.png')
        file_list.sort() # ensure same order [1]

        for filename in file_list: # png for base
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]

            print(basename)
            img = cv2.imread(imgs_dir + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            hw = img.shape[:2]

            xml = ET.parse(anns_dir + basename + '.xml')

            contour_dbg = np.zeros(hw, np.uint8)

            insts_list = []
            for idx, region_xml in enumerate(xml.findall('.//Region')):
                vertices = []
                for vertex_xml in region_xml.findall('.//Vertex'):
                    attrib = vertex_xml.attrib
                    vertices.append([float(attrib['X']), 
                                     float(attrib['Y'])])
                vertices = np.array(vertices) + 0.5
                vertices = vertices.astype('int32')
                contour_blb = np.zeros(hw, np.uint8)

                # fill both the inner area and contour with idx+1 color
                cv2.drawContours(contour_blb, [vertices], 0, idx+1, -1)
                insts_list.append(contour_blb)

            insts_size_list = np.array(insts_list)
            insts_size_list = np.sum(insts_size_list, axis=(1 , 2))
            insts_size_list = list(insts_size_list)

            pair_insts_list = zip(insts_list, insts_size_list)

            # sort in z-axis basing on size, larger on top
            pair_insts_list = sorted(pair_insts_list, key=lambda x: x[1])
            insts_list, insts_size_list = zip(*pair_insts_list)

            ann = np.zeros(hw, np.int32)
            for idx, inst_map in enumerate(insts_list):
                ann[inst_map > 0] = idx + 1

            mask_fn = save_dir+'/'+basename+'_mask.mat'
            scipy.io.savemat(mask_fn, mdict={'inst_map': ann})
    
    
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

    
