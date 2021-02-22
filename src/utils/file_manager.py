import tables
import zipfile
import cv2
import scipy.io
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.settings import DATA_DIR, CONF_DIR, PATCH_DIR, RESULT_DIR


class FileHandler:
    """
    Class for handling different file formats that are needed in the project.
    """
    @staticmethod
    def read_img(path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB)

    @staticmethod
    def read_mask(path: Union[str, Path], key: str = "inst_map") -> np.ndarray:
        assert key in ("inst_map", "type_map", "inst_centroid", "inst_type")
        dtypes = {"inst_map":"int32", "type_map":"int32", "inst_centroid":"float64", "inst_type":"int32"}
        path = Path(path)
        return scipy.io.loadmat(path.as_posix())[key].astype(dtypes[key])
    
    @staticmethod
    def remove_existing_files(directory: Union[str, Path]) -> None:
        assert Path(directory).exists(), f"{directory} does not exist"
        for item in Path(directory).iterdir():
            # Do not touch the folders inside the directory
            if item.is_file():
                item.unlink()

    @staticmethod
    def read_hdf5_patch(path: Union[str, Path], index: int) -> Tuple[np.ndarray]:
        path = Path(path)
        with tables.open_file(path.as_posix(), "r") as db:
            img = db.root.img
            inst_map = db.root.inst_map
            type_map = db.root.type_map
            im_patch = img[index, ...]
            inst_patch = inst_map[index, ...]
            type_patch = type_map[index, ...]
        return im_patch, inst_patch, type_patch

    @staticmethod
    def get_class_pixels(path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        with tables.open_file(path.as_posix(), "r") as db:
            weights = db.root.numpixels[:]
        class_weight = weights[1, ]
        return class_weight

    @staticmethod
    def create_dir(path: Union[str, Path]) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_zips(path: Union[str, Path], rm: bool = False) -> None:
        for f in Path(path).iterdir():
            if f.is_file() and f.suffix == ".zip":
                with zipfile.ZipFile(f, 'r') as z:
                    z.extractall(path)
                if rm:
                    f.unlink()

    @staticmethod
    def suffix(path: Union[str, Path]) -> str:
        path = Path(path)
        assert all([f.suffix for f in path.iterdir()]), "Image files should be in same format"
        return [f.suffix for f in path.iterdir()][0]

    def get_files(self, path: Union[str, Path]) -> List[str]:
        path = Path(path)
        assert path.exists(), f"Provided directory: {path.as_posix()} does not exist."
        file_suffix = self.suffix(d)
        return sorted([x.as_posix() for x in path.glob(f"*{file_suffix}")])


class FileManager(FileHandler):
    def __init__(self,
                 experiment_args: DictConfig,
                 dataset_args: DictConfig) -> None:
        """
        File hadling and managing

        Args:
            experiment_args (DictConfig): 
                Omegaconfig DictConfig specifying arguments that
                are used for creating result folders and files. 
            dataset_args (DictConfig): 
                Omegaconfig DictConfig specifying arguments related 
                to the dataset that is being used.
        """
        self.__ex_name: str = experiment_args.experiment_name
        self.__ex_version: str = experiment_args.experiment_version
        self.__train_ds: str = dataset_args.train_dataset

    @classmethod
    def from_conf(cls, conf: DictConfig):
        exargs = conf.experiment_args
        dsargs = conf.dataset_args

        return cls(
            exargs,
            dsargs
        )

    @property
    def result_folder(self):
        return RESULT_DIR

    @property
    def experiment_dir(self) -> Path:
        return RESULT_DIR / f"{self.__ex_name}" / f"version_{self.__ex_version}"

    @property
    def train_dataset(self):
        assert self.__train_ds in ("kumar","consep","pannuke","dsb2018", "monusac", "other")
        return self.__train_ds

    @property
    def classes(self):
        yml_path = [f for f in Path(CONF_DIR).iterdir() if self.train_dataset in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        classes = data_conf.class_types.type
        return classes

    @property
    def pannuke_folds(self) -> Dict[str, str]:
        yml_path = [f for f in Path(CONF_DIR).iterdir() if "pannuke" in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        return data_conf.folds
        
    @property
    def pannuke_tissues(self) -> List[str]:
        yml_path = [f for f in Path(CONF_DIR).iterdir() if "pannuke" in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        return data_conf.tissues
    
    def __split_training_set(self,
                             dataset: str,
                             train_imgs: List[str], 
                             train_masks: List[str],
                             seed: int = 42,
                             size: float = 0.2) -> Dict[str, List[str]]:
        """
        Split training set into training and validation set. This might affect training 
        accuracy if training set is small. Pannuke is split by the folds specified in the pannuke.yml
        """

        assert dataset in ("kumar","consep","pannuke","dsb2018", "monusac", "other")

        def split_pannuke(paths):
            phases = {}
            for fold, phase in self.pannuke_folds.items():
                phases[phase] = []
                for item in paths:
                    if fold in Path(item).name:
                        phases[phase].append(item)
            return phases
        
        if dataset == "pannuke":
            imgs = split_pannuke(train_imgs)
            masks = split_pannuke(train_masks)
            train_imgs = sorted(imgs["train"])
            train_masks = sorted(masks["train"])
            valid_imgs = sorted(imgs["valid"])
            valid_masks = sorted(masks["valid"])
        else:
            ixs = np.arange(len(train_imgs))
            train_ixs, valid_ixs = train_test_split(ixs, test_size=size, random_state=42, shuffle=True)
            np_train_imgs = np.array(train_imgs)
            np_train_masks = np.array(train_masks)
            train_imgs = sorted(np_train_imgs[train_ixs].tolist())
            train_masks = sorted(np_train_masks[train_ixs].tolist())
            valid_imgs = sorted(np_train_imgs[valid_ixs].tolist())
            valid_masks = sorted(np_train_masks[valid_ixs].tolist())
        
        return {
            "train_imgs":train_imgs,
            "train_masks":train_masks,
            "valid_imgs":valid_imgs,
            "valid_masks":valid_masks
        }

    def get_data_dirs(self, dataset: str) -> Dict[str, Path]:
        """
        Get the paths of data dirs for given dataset
        """
        assert dataset in ("kumar","consep","pannuke","dsb2018", "monusac", "other")
        assert Path(DATA_DIR / dataset).exists(), "No train data found"

        return {
            "raw_data_dir":Path(DATA_DIR / dataset),
            "train_im":Path(DATA_DIR / dataset / "train" / "images"),
            "train_gt":Path(DATA_DIR / dataset / "train" / "labels"),
            "test_im":Path(DATA_DIR / dataset / "test" / "images"),
            "test_gt":Path(DATA_DIR / dataset / "test" / "labels"),
        }

    def get_data_folds(self, dataset: str, split_train: bool=False) -> Dict[str, Dict[str, str]]:
        """
        Get train data folds

        Args:
            dataset (str):
                name of the dataset. One of ("kumar","consep","pannuke","dsb2018", "monusac", "other")
            split_train (bool):
                split training data into training and validation data

        Returns:
            Dict[str, Dict[str, str]]
            Dictionary where keys (train, test, valid) point to dictionaries with keys img and mask.
            The innermost dictionary values where the img and mask keys point to are sorted lists 
            containing paths to the image and mask files. 
        """

        dirs = self.get_data_dirs(dataset)
        assert all(Path(p).exists() for p in dirs.values()), (
            "Data folders do not exists. Convert the data first to right format."
        )
        assert dataset in ("kumar","consep","pannuke","dsb2018", "monusac", "other")

        train_imgs = self.get_files(dirs["train_im"])
        train_masks = self.get_files(dirs["train_gt"])
        test_imgs = self.get_files(dirs["test_im"])
        test_masks = self.get_files(dirs["test_gt"])
        valid_imgs = None
        valid_masks = None

        if split_train:
            d = self.__split_training_set(train_imgs, train_masks)
            train_imgs = d["train_imgs"]
            train_masks = d["train_masks"]
            valid_imgs = d["valid_imgs"]
            valid_masks = d["valid_masks"]

        return {
            "train":{"img":train_imgs, "mask":train_masks},
            "valid":{"img":valid_imgs, "mask":valid_masks},
            "test":{"img":test_imgs, "mask":test_masks},
        }

    
    def get_databases(self, dataset: str) -> Dict[str, Dict[int, str]]:
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

        db_dir = Path(PATCH_DIR / "hdf5" / dataset)         
        assert db_dir.exists(), (
            f"Database dir: {db_dir} not found, Create the dbs first. Check instructions."
        )
            
        train_dbs = list(db_dir.glob("*_train_*"))
        valid_dbs = list(db_dir.glob("*_valid_*"))
        test_dbs = list(db_dir.glob("*_test_*"))
        
        if not valid_dbs:
            valid_dbs = test_dbs

        assert train_dbs, (
            f"{train_dbs} HDF5 training db not found. Create the dbs first. Check instructions."
        )
        
        return {
            "train":get_input_sizes(train_dbs),
            "valid":get_input_sizes(valid_dbs),
            "test":get_input_sizes(test_dbs)
        }

    def get_model_checkpoint(self, which: str = "last") -> Path:
        """
        Get the best or last checkpoint of a trained network.

        Args:
            which (str): 
                One of ("best", "last"). Specifies whether to use 
                last epoch model or best model on validation data.

        Returns:
            Path object to the pytorch checkpoint file.
        """
        assert which in ("best", "last"), f"param which: {which} not one of ('best', 'last')"
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
            f"ckpt: {ckpt}. Checkpoint is None. Make sure that a .ckpt file exists in experiment dir",
            "If you dont want to resume training, check that resume_training = False in experiment.yml"
        )
        return ckpt

