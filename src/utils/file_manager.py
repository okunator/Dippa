import zipfile
import zarr
import cv2
import scipy.io as sio
import numpy as np
import tables as tb
from pathlib import Path
from typing import Tuple, Union

from src.settings import RESULT_DIR


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
        assert key in (
            "inst_map", "type_map", "inst_centroid", "inst_type", "sem_map"
        )

        dtypes = {
            "inst_map":"int32", 
            "type_map":"int32", 
            "sem_map":"int32",
            "inst_centroid":"float64", 
            "inst_type":"int32"
        }

        path = Path(path)
        return sio.loadmat(path.as_posix())[key].astype(dtypes[key])
    
    @staticmethod
    def remove_existing_files(directory: Union[str, Path]) -> None:
        assert Path(directory).exists(), f"{directory} does not exist"
        for item in Path(directory).iterdir():
            # Do not touch the folders inside the directory
            if item.is_file():
                item.unlink()

    @staticmethod
    def read_zarr_patch(path: str, ix: int):
        d = zarr.open(path, mode="r")
        im_patch = d["imgs"][ix, ...]
        inst_patch = d["insts"][ix, ...]
        type_patch = d["types"][ix, ...]
        return im_patch, inst_patch, type_patch 

    @staticmethod
    def read_h5_patch(path: str, ix: int):
        with tb.open_file(path, "r") as h5:
            im_patch = h5.root.imgs[ix, ...]
            inst_patch = h5.root.insts[ix, ...]
            type_patch = h5.root.types[ix, ...]
            sem_patch = h5.root.areas[ix, ...]
        
        return im_patch, inst_patch, type_patch, sem_patch

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
    def get_model_checkpoint(
            experiment: str,
            version: str,
            which: str="last"
        ) -> Path:

        assert which in ("best", "last"), (
            f"param which: {which} not one of ('best', 'last')"
        )

        experiment_dir = Path(f"{RESULT_DIR}/{experiment}/version_{version}")  
        assert experiment_dir.exists(), (
            f"Experiment dir: {experiment_dir.as_posix()} does not exist.",
            "It might be that you have `resume_training` set to True",
            "but you are trying to run the experiment for the first time."
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
