import zipfile
import zarr
import cv2
import scipy.io
import numpy as np
import tables as tb
from pathlib import Path
from typing import List, Dict, Tuple, Union
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.settings import DATA_DIR, CONF_DIR, PATCH_DIR, RESULT_DIR


class FileHandler:
    """
    Class for handling flie reading
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
    def read_zarr_patch(path: str, index: int):
        d = zarr.open(path, mode="r")
        im_patch, inst_patch, type_patch = d["imgs"][index, ...], d["insts"][index, ...], d["types"][index, ...]
        return im_patch, inst_patch, type_patch 

    @staticmethod
    def read_h5_patch(path: str, index: int):
        with tb.open_file(path, "r") as h5:
            img = h5.root.imgs
            inst_map = h5.root.insts
            type_map = h5.root.types
            im_patch = img[index, ...]
            inst_patch = inst_map[index, ...]
            type_patch = type_map[index, ...]
        return im_patch, inst_patch, type_patch

    @staticmethod
    def get_class_weights(path: Union[str, Path], binary: bool=False) -> np.ndarray:
        path = Path(path)
        if path.suffix == ".h5": 
            with tb.open_file(path.as_posix(), "r") as db:
                npixels = db.root.numpixels[:]
        elif path.suffix == ".zarr":
            z = zarr.open(path.as_posix(), mode="r")
            npixels = z["npixels"][:]

        if binary:
            fg_npixels = np.sum(npixels, where=[False] + [True]*(npixels.shape[1]-1))
            bg_npixels = npixels[0][0]
            npixels = np.array([[bg_npixels, fg_npixels]])

        class_weights = 1 - npixels / npixels.sum()
        return class_weights

    @staticmethod
    def get_dataset_stats(path: Union[str, Path]) -> Tuple[np.ndarray]:
        path = Path(path)
        if path.suffix == ".h5": 
            with tb.open_file(path.as_posix(), "r") as db:
                mean = db.root.dataset_mean[:]
                std = db.root.dataset_std[:]

        elif path.suffix == ".zarr":
            z = zarr.open(path.as_posix(), mode="r")
            mean = z["dataset_mean"][:]
            std = z["dataset_std"][:]
        
        return mean, std

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
        assert all([f.suffix for f in path.iterdir()]), "All files should be in same format"
        return [f.suffix for f in path.iterdir()][0]

    def get_files(self, path: Union[str, Path]) -> List[str]:
        path = Path(path)
        assert path.exists(), f"Provided directory: {path.as_posix()} does not exist."
        file_suffix = self.suffix(path)
        return sorted([x.as_posix() for x in path.glob(f"*{file_suffix}")])


# TODO: REWRITE
class FileManager(FileHandler):
    def __init__(self,
                 experiment_name: str,
                 experiment_version: str) -> None:
        """
        File hadling and managing

        Args:
        ------------
            experiment_name (str):
                Name of the experiment
            experiment_version (str):
                Name of the experiment version
        """
        self.__ex_name = experiment_name
        self.__ex_version = experiment_version

    @property
    def result_folder(self):
        return RESULT_DIR

    @property
    def experiment_dir(self) -> Path:
        return RESULT_DIR / f"{self.__ex_name}" / f"version_{self.__ex_version}"
    
    def get_classes(self, dataset):
        yml_path = [f for f in Path(CONF_DIR).iterdir() if dataset in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        classes = data_conf.class_types.type
        return classes
    
    def __split_training_set(self,
                             dataset: str,
                             train_imgs: List[str], 
                             train_masks: List[str],
                             seed: int=42,
                             size: float=0.2) -> Dict[str, List[str]]:
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
        assert Path(DATA_DIR / dataset).exists(), f"No data found for dataset {dataset}"

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
        ------------
            dataset (str):
                name of the dataset. One of ("kumar","consep","pannuke","dsb2018", "monusac", "other")
            split_train (bool):
                split training data into training and validation data

        Returns:
        ------------
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

    
    def get_databases(self, dataset: str, db_type: str="zarr") -> Dict[str, Dict[int, str]]:
        """
        Get the the databases that were written by the PatchWriter
        
        Args:
        -------------
        dataset (str):
            Name of the dataset. One of ("consep", "kumar", "pannuke")
        db_type (str):
            One of ("zarr", "hdf5"). Depends on what db's have been written
            by the datawriters.

        Returns:
        -------------
            Dict[str, Path]
            A dictionary where the keys (train, test, valid) are pointing to Path objs of dbs.
        """

        db_dir = Path(PATCH_DIR / db_type / dataset)       
        assert db_dir.exists(), (f"Database dir: {db_dir} not found, Create the dbs first. Check instructions.")
            
        train_dbs = list(db_dir.glob(f"*train_{dataset}*"))
        valid_dbs = list(db_dir.glob(f"*valid_{dataset}*"))
        test_dbs = list(db_dir.glob(f"*test_{dataset}*"))
        
        assert train_dbs, (f"{train_dbs} training db not found. Create the dbs first. Check instructions.")
        assert test_dbs, (f"{test_dbs} test db not found. Create the dbs first. Check instructions.")
        
        if not valid_dbs:
            valid_dbs = test_dbs

        dbs = {
            "train":train_dbs[0],
            "valid":valid_dbs[0],
            "test":test_dbs[0]
        }

        return dbs
        

    def get_model_checkpoint(self, which: str = "last") -> Path:
        """
        Get the best or last checkpoint of a trained network.

        Args:
        ------------
            which (str): 
                One of ("best", "last"). Specifies whether to use 
                last epoch model or best model on validation data.

        Returns:
        ------------
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

