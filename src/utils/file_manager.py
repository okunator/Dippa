import tables
import zipfile
import cv2
import scipy.io
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from src.img_processing.viz_utils import draw_contours
from src.img_processing.process_utils import overlays
from src.settings import DATA_DIR, CONF_DIR, PATCH_DIR, RESULT_DIR


class FileHandler:
    """
    Class for handling different file formats that are needed in the project.
    """
    @staticmethod
    def read_img(path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    @staticmethod
    def read_mask(path: str, key: str = "inst_map") -> np.ndarray:
        assert key in ("inst_map", "type_map", "inst_centroid", "inst_type")
        dtypes = {"inst_map":"int32", "type_map":"int32", "inst_centroid":"float64", "inst_type":"int32"}
        return scipy.io.loadmat(path)[key].astype(dtypes[key])
    
    @staticmethod
    def remove_existing_files(directory: str) -> None:
        assert Path(directory).exists(), f"{directory} does not exist"
        for item in Path(directory).iterdir():
            # Do not touch the folders inside the directory
            if item.is_file():
                item.unlink()

    @staticmethod
    def read_hdf5_patch(path: str, index: int) -> Tuple[np.ndarray]:
        with tables.open_file(path, "r") as db:
            img = db.root.img
            inst_map = db.root.inst_map
            type_map = db.root.type_map
            im_patch = img[index, ...]
            inst_patch = inst_map[index, ...]
            type_patch = type_map[index, ...]
        return im_patch, inst_patch, type_patch

    @staticmethod
    def get_class_pixels_num(path: str) -> np.ndarray:
        with tables.open_file(path, "r") as db:
            weights = db.root.numpixels[:]
        class_weight = weights[1, ]
        return class_weight

    @staticmethod
    def create_dir(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_zips(folder: str, rm: bool = False) -> None:
        for f in Path(folder).iterdir():
            if f.is_file() and f.suffix == ".zip":
                with zipfile.ZipFile(f, 'r') as z:
                    z.extractall(folder)
                if rm:
                    f.unlink()


# Lots of spadghetti b/c different datasets need different ad hoc processing to get stuff aligned
class ProjectFileManager(FileHandler):
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig,
                 **kwargs) -> None:
        """
        A class for managing the file and folder paths needed in this project.
        
            Args:
                dataset_args (DictConfig): omegaconfig DictConfig specifying arguments
                                           related to the dataset that is being used.
                                           config.py for more info
                experiment_args (DictConfig): omegaconfig DictConfig specifying arguments
                                              that are used for creating result folders and
                                              files. Check config.py for more info
        """
        super().__init__(**kwargs)
        self.dsargs = dataset_args
        self.exargs = experiment_args
    
    @classmethod
    def from_conf(cls, conf: DictConfig):
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        
        return cls(
            dataset_args,
            experiment_args
        )

    @property
    def experiment_dir(self) -> Path:
        ex_version = self.exargs.experiment_version
        model_name = self.exargs.model_name
        return RESULT_DIR.joinpath(f"{model_name}/version_{ex_version}")
    
    @property
    def dataset(self) -> str:
        assert self.dsargs.dataset in ("kumar","consep","pannuke","dsb2018","cpm")
        return self.dsargs.dataset
    
    @property
    def data_dirs(self) -> Dict:
        assert Path(DATA_DIR / self.dataset).exists(), (
            f"The folder {Path(DATA_DIR / self.dataset)} for the dataset {self.dataset} does not exist"
        )
        return {
            "raw_data_dir":Path(DATA_DIR / self.dataset),
            "train_im":Path(DATA_DIR / self.dataset / "train" / "images"),
            "train_gt":Path(DATA_DIR / self.dataset / "train" / "labels"),
            "test_im":Path(DATA_DIR / self.dataset / "test" / "images"),
            "test_gt":Path(DATA_DIR / self.dataset / "test" / "labels"),
        }
    
    @property
    def database_dir(self) -> Path:
        assert self.dsargs.patches_dtype in ("npy", "hdf5"), f"{self.dsargs.patches_dtype}"
        return Path(PATCH_DIR / self.dsargs.patches_dtype / self.dataset)
    
    @property
    def phases(self) -> List[str]:
        assert self.dsargs.phases in (["train", "valid", "test"], ["train", "test"]), f"{self.dsargs.phases}"
        return self.dsargs.phases
    
    @property
    def classes(self) -> Dict[str, int]:
        assert self.dsargs.class_types in ("instance", "panoptic")
        yml_path = [f for f in Path(CONF_DIR).iterdir() if self.dataset in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        if self.dataset == "consep":
            key = data_conf.types_to_use
        else:
            key = self.dsargs.class_types
        return data_conf.class_types[key]

    @property
    def img_types(self) -> Dict[str, str]:
        yml_path = [f for f in Path(CONF_DIR).iterdir() if self.dataset in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        return data_conf.img_types

    @property
    def class_types(self):
        return self.dsargs.class_types
    
    @property
    def pannuke_folds(self) -> Dict[str, str]:
        assert self.dataset == "pannuke", f"dataset: {self.dataset}. dataset not pannuke"
        yml_path = [f for f in Path(CONF_DIR).iterdir() if self.dataset in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        return data_conf.folds
        
    @property
    def pannuke_tissues(self) -> List[str]:
        yml_path = [f for f in Path(CONF_DIR).iterdir() if self.dataset in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        return data_conf.tissues
        
    @property
    def data_folds(self) -> Dict[str, Dict[str, str]]:
        """
        Get the 'train', 'valid', 'test' fold paths. This also splits training set to train and valid
        if 'valid is in self.phases list'
        
        Returns:
            Dict[str, Dict[str, str]]
            Dictionary where keys (train, test, valid) point to dictionaries with keys img and mask.
            The innermost dictionary values where the img and mask -keys point to are sorted lists 
            containing paths to the image and mask files. 
        """
        assert all(Path(p).exists() for p in self.data_dirs.values()), (
            "Data folders do not exists Convert the data first to right format. See step 1."
        )
        
        train_imgs = self.__get_files(self.data_dirs["train_im"])
        train_masks = self.__get_files(self.data_dirs["train_gt"])
        test_imgs = self.__get_files(self.data_dirs["test_im"])
        test_masks = self.__get_files(self.data_dirs["test_gt"])
        
        if "valid" in self.phases:
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
    def databases(self) -> Dict[str, Dict[int, str]]:
        """
        Get the the databases that were written by the PatchWriter
        
        Returns:
            Dict[str, Dict[int, str]]
            A dictionary where the keys (train, test, valid) are pointing to dictionaries with
            values are the paths to the hdf5 databases and keys are the size of the square 
            patches saved in the databass
        """
        def get_input_sizes(paths):
            # Paths to dbs are named as 'patchsize_dataset'. e.g. 'patch256_kumar.pytable' 
            # This will parse the name and use patchsize as key for a dict
            path_dict = {}
            for path in paths:
                fn = path.as_posix().split("/")[-1]
                size = int("".join(filter(str.isdigit, fn)))
                path_dict[size] = path
            return path_dict
                    
        assert self.database_dir.exists(), (
            f"Database dir: {self.database_dir} not found, Create the dbs first. Check instructions."
        )
            
        if "valid" in self.phases:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_valid_*"))
            test_dbs = list(self.database_dir.glob("*_test_*"))
        else:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_test_*")) # test set used for both valid and test
            test_dbs = list(self.database_dir.glob("*_test_*")) # remember to not use the best model w this
            
        assert train_dbs, (
            f"{train_dbs} HDF5 training db not found. Create the dbs first. Check instructions."
        )
        
        assert valid_dbs, (
            f"{valid_dbs} HDF5 validation db not found. Create the dbs first. Check instructions."
        )
        
        return {
            "train":get_input_sizes(train_dbs),
            "valid":get_input_sizes(valid_dbs),
            "test":get_input_sizes(test_dbs)
        }

    def model_checkpoint(self, which: str = "last") -> Path:
        """
        Get the best or last checkpoint of a trained network.
        Args:
            which (str): one of ("best", "last"). Specifies whether to use last epoch model or best model on
                         validation data.
        Returns:
            Path object to the pytorch checkpoint file.
        """
        assert which in (
            "best", "last"), f"param which: {which} not one of ('best', 'last')"
        assert self.experiment_dir.exists(), (
            f"Experiment dir: {self.experiment_dir} does not exist. Train the model first."
        )

        ckpt = None
        for item in self.experiment_dir.iterdir():
            if item.is_file() and item.suffix == ".ckpt":
                if which == "last" and "last" in str(item) and ckpt is None:
                    ckpt = item
                elif which == "best" and "last" not in str(item) and ckpt is None:
                    ckpt = item

        assert ckpt is not None, (
            f"ckpt: {ckpt}. Checkpoint is None. Make sure that a .ckpt file exists in experiment dir"
        )
        return ckpt

    def __get_img_suffix(self, directory: Path) -> str:
        assert all([f.suffix for f in directory.iterdir()]), "All image files should be in same format"
        return [f.suffix for f in directory.iterdir()][0]
           
    def __get_files(self, directory: str) -> List[str]:
        d = Path(directory)
        assert d.exists(), f"Provided directory: {d.as_posix()} does not exist."
        file_suffix = self.__get_img_suffix(d)
        return sorted([x.as_posix() for x in d.glob(f"*{file_suffix}")])
           
    def __split_training_set(self, 
                             train_imgs: List, 
                             train_masks: List,
                             seed: int = 42,
                             size: float = 0.2) -> Dict[str, List[str]]:
        """
        Split training set into training and validation set. This might affect training 
        accuracy if training set is small. Pannuke is split by the folds specified in the pannuke.yml
        """
        def split_pannuke(paths):
            phases = {}
            for fold, phase in self.pannuke_folds.items():
                phases[phase] = []
                for item in paths:
                    if fold in Path(item).name:
                        phases[phase].append(item)
            return phases
        
        if self.dataset == "pannuke":
            imgs = split_pannuke(train_imgs)
            masks = split_pannuke(train_masks)
            train_imgs = sorted(imgs["train"])
            train_masks = sorted(masks["train"])
            valid_imgs = sorted(imgs["valid"])
            valid_masks = sorted(masks["valid"])
        else:
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


    
