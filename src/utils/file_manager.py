import zipfile
import zarr
import cv2
import scipy.io
import numpy as np
import tables as tb
from pathlib import Path
from typing import Dict, Tuple, Union

from src.settings import DATA_DIR, RESULT_DIR


class FileHandler:
    """
    Class for handling flie reading
    """
    @staticmethod
    def read_img(path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB)

    @staticmethod
    def read_mask(path: Union[str, Path], key: str="inst_map") -> np.ndarray:
        assert key in ("inst_map", "type_map", "inst_centroid", "inst_type")

        dtypes = {
            "inst_map":"int32", 
            "type_map":"int32", 
            "inst_centroid":"float64", 
            "inst_type":"int32"
        }

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
        im_patch = d["imgs"][index, ...]
        inst_patch = d["insts"][index, ...]
        type_patch = d["types"][index, ...]
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
    def get_class_weights(path: Union[str, Path], 
                          binary: bool=False) -> np.ndarray:
        path = Path(path)
        if path.suffix == ".h5": 
            with tb.open_file(path.as_posix(), "r") as db:
                npixels = db.root.npixels[:]
        elif path.suffix == ".zarr":
            z = zarr.open(path.as_posix(), mode="r")
            npixels = z["npixels"][:]

        if binary:
            bixs = [False] + [True]*(npixels.shape[1]-1)
            fg_npixels = np.sum(npixels, where=bixs)
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
    def get_experiment_dir(experiment: str, version: str) -> Path:
        experiment_dir = Path(f"{RESULT_DIR}/{experiment}/version_{version}")
        return experiment_dir

    @staticmethod
    def result_dir() -> Path:
        return Path(RESULT_DIR)

    @staticmethod
    def get_model_checkpoint(experiment: str, 
                             version: str, 
                             which: str = "last") -> Path:

        assert which in ("best", "last"), (
            f"param which: {which} not one of ('best', 'last')"
        )

        experiment_dir = Path(f"{RESULT_DIR}/{experiment}/version_{version}")  
        assert experiment_dir.exists(), (
            f"Experiment dir: {experiment_dir.as_posix()} does not exist."
        )

        cpt = None
        for it in experiment_dir.iterdir():
            if it.is_file() and it.suffix == ".ckpt":
                if which == "last" and "last" in str(it) and cpt is None:
                    cpt = it
                elif which == "best" and "last" not in str(it) and cpt is None:
                    cpt = it

        assert cpt is not None, (
            f"ckpt: {cpt}. Checkpoint is None. Make sure that a .ckpt file",
            f"exists in {experiment_dir}. If you dont want to resume training,"
             "check that resume_training = False in experiment.yml"
        )
        return cpt
