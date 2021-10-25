import zipfile
import zarr
import cv2
import scipy.io as sio
import numpy as np
import tables as tb
from pathlib import Path
from typing import Tuple, Union, Dict

from src.settings import RESULT_DIR


class FileHandler:
    """
    Class for handling flie reading
    """
    @staticmethod
    def read_img(path: Union[str, Path]) -> np.ndarray:
        """
        Read an image & convert from bgr to rgb. (cv2 reads imgs in bgr)

        Args:
        ---------
            path (str | Path):
                Path to the image file.

        Returns:
        ---------
            np.ndarray: The image. Shape (H, W, 3)
        """
        path = Path(path)
        return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB)

    @staticmethod
    def write_img(path: Union[str, Path], img: np.ndarray) -> None:
        """
        Write an image.

        Args:
        ---------
            path (str | Path):
                Path to the image file.
            img (np.ndarray):
                The image to be written.

        """
        path = Path(path)
        cv2.imwrite(path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @staticmethod
    def read_mask(path: Union[str, Path], key: str="inst_map") -> np.ndarray:
        """
        Read a mask from a .mat file.

        Args:
        ---------
            path (str | Path):
                Path to the image file.
            key (str):
                name/key of the mask type that is being read from .mat
        
        Returns:
        ----------
            np.ndarray: The mask indice matrix. Shape (H, W) 
        """
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
    def read_zarr_patch(path: str, ix: int) -> Tuple[np.ndarray]:
        """
        Read img & mask patches at index `ix` from zarr db-arrays

        Args:
        --------
            path (str):
                Path to the hdf5 database
            ix (int):
                Index for the hdf5 db-arrays

        Returns:
        --------
            Tuple[np.ndarray]: Tuple of numpy matrices. Img is of shape
            (H, W, 3) & masks are of shape (H, W). the order of the
            returned matrices is img, inst, type, area.
        """
        d = zarr.open(path, mode="r")
        im_patch = d["imgs"][ix, ...]
        inst_patch = d["insts"][ix, ...]
        type_patch = d["types"][ix, ...]
        sem_patch = d["areas"][ix, ...]

        return im_patch, inst_patch, type_patch , sem_patch

    @staticmethod
    def read_h5_patch(path: str, ix: int) -> Tuple[np.ndarray]:
        """
        Read img & mask patches at index `ix` from hdf5 db-arrays

        Args:
        --------
            path (str):
                Path to the hdf5 database
            ix (int):
                Index for the hdf5 db-arrays

        Returns:
        --------
            Tuple[np.ndarray]: Tuple of numpy matrices. Img is of shape
            (H, W, 3) & masks are of shape (H, W). the order of the
            returned matrices is img, inst, type, area.
        """
        with tb.open_file(path, "r") as h5:
            im_patch = h5.root.imgs[ix, ...]
            inst_patch = h5.root.insts[ix, ...]
            type_patch = h5.root.types[ix, ...]
            sem_patch = h5.root.areas[ix, ...]
        
        return im_patch, inst_patch, type_patch, sem_patch

    @staticmethod
    def get_class_dicts(path: Union[str, Path]) -> Tuple[Dict[str, int]]:
        """
        Read the classes dict from a .h5 db

        Args:
        --------
            path (str):
                Path to the hdf5 database

        Returns:
        --------
            Tuple[Dict[str, int]]: A tuple of dict mappings to classes 
            e.g. ({"bg": 0, "immune": 1, "cancer": 2}, {"cancer": 1}
            The second return value can also be None.
        """
        path = Path(path)
        with tb.open_file(path.as_posix(), "r") as db:
            classes = db.root._v_attrs.classes
            sem_classes = db.root._v_attrs.sem_classes

        return classes, sem_classes

    @staticmethod
    def get_class_weights(
            path: Union[str, Path],
            binary: bool=False
        ) -> np.ndarray:
        """
        Read class weight arrays that are saved in h5 or zarr db.
        h5 or zarr dbs need to be written with the db-writers in this
        repo for this to work

        Args:
        --------
            path (str):
                Path to the hdf5 database
            binary (bool, default=False):
                If True, Computes only background vs. foregroung weights

        Returns:
        --------
            np.ndarray: class weights array. Shape (C, )
        """
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
        """
        Read dataset statistics arrays that are saved in h5 or zarr db.
        h5 or zarr dbs need to be written with the db-writers in this
        repo for this to work

        Args:
        --------
            path (str):
                Path to the hdf5 database

        Returns:
        --------
            Tuple: Tuple of dataset channel-wise mean and std arrays. 
                   Both have shape (C, ). Shape for RGB imgs (3, )
        """
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
    def get_dataset_size(path: Union[str, Path]) -> Tuple[np.ndarray]:
        pass

    @staticmethod
    def create_dir(path: Union[str, Path]) -> None:
        """
        Create a directory.

        Args:
        ---------
            path (str | Path):
                Path to the image file.
        """
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_zips(path: Union[str, Path], rm: bool = False) -> None:
        """
        Extract files from all the .zip files inside a folder.

        Args:
        ---------
            path (str | Path):
                Path to the image file.
            rm (bool, default=False):
                remove the .zip files after extraction.
        """
        for f in Path(path).iterdir():
            if f.is_file() and f.suffix == ".zip":
                with zipfile.ZipFile(f, 'r') as z:
                    z.extractall(path)
                if rm:
                    f.unlink()
    @staticmethod
    def remove_existing_files(path: Union[str, Path]) -> None:
        """
        Remove files inside a folder

        Args:
        ---------
            path (str | Path):
                Path to the image file.
        """
        assert Path(path).exists(), f"{path} does not exist"
        for item in Path(path).iterdir():
            # Do not touch the folders inside the directory
            if item.is_file():
                item.unlink()

    @staticmethod
    def get_experiment_dir(experiment: str, version: str) -> Path:
        """
        Get the path where all thr files related to a specific experiment
        are saved during training and inference.

        Args:
        ---------
            experiment (str):
                name of the experiment
            version (str):
                version of the experiment

        Returns:
        ---------
            Path: path to the experiment dir
        """
        experiment_dir = Path(f"{RESULT_DIR}/{experiment}/version_{version}")
        return experiment_dir

    @staticmethod
    def result_dir() -> Path:
        """
        Get the default results directory
        """
        return Path(RESULT_DIR)

    @staticmethod
    def get_model_checkpoint(
            experiment: str,
            version: str,
            which: str="last"
        ) -> Path:
        """
        Get the path to the model checkpoint/weights inside a specific
        experiment directory.

        Args:
        ---------
            experiment (str):
                name of the experiment
            version (str):
                version of the experiment

        Returns:
        ---------
            Path: path to the checkpoint file
        """
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
