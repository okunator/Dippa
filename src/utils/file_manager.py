import zipfile
import cv2
import scipy.io as sio
import numpy as np
import tables as tb
from pathlib import Path
from typing import Tuple, Union, Dict

from src.settings import RESULT_DIR
from .mask_utils import (
    get_inst_centroid, get_inst_types, bounding_box, fix_duplicates
)


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
    def write_mask(
            path: Union[str, Path],
            inst_map: np.ndarray,
            type_map: np.ndarray,
            sem_map: np.ndarray,
        ) -> None:
        """
        Write multiple masks to .mat file.

        Args:
        ---------
            path (str | Path):
                Path to the .mat file.
            inst_map (np.ndarray):
                The inst_map to be written.
            type_map (np.ndarray):
                The inst_map to be written.
            sem_map (np.ndarray, default=None):
                The inst_map to be written.
        """
        assert Path(path).suffix == ".mat"
        
        inst_map = fix_duplicates(inst_map)
        centroids = get_inst_centroid(inst_map)
        inst_types = get_inst_types(inst_map, type_map)
        inst_ids = list(np.unique(inst_map)[1:])
        bboxes = np.array(
            [bounding_box(np.array(inst_map == id_, np.uint8)) 
            for id_ in inst_ids]
        )
        
        sio.savemat(
            file_name=path, 
            mdict={
                "inst_map": inst_map,
                "type_map":type_map,
                "sem_map": sem_map,
                "inst_type":inst_types,
                "inst_centroid":centroids,
                "inst_bbox":bboxes
            }
        )
    
    @staticmethod
    def read_mask(
            path: Union[str, Path],
            key: str="inst_map"
        ) -> Union[np.ndarray, None]:
        """
        Read a mask from a .mat file. If a mask is not found with
        return None

        Args:
        ---------
            path (str | Path):
                Path to the image file.
            key (str):
                name/key of the mask type that is being read from .mat
        
        Returns:
        ----------
            np.ndarray or None: The mask indice matrix. Shape (H, W)
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

        assert path.exists(), f"{path} not found"

        try:
            mask = sio.loadmat(path.as_posix())[key].astype(dtypes[key])
        except:
            mask = None

        return mask
    
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

            # These masks are not necessarily in the dset
            try:
                type_patch = h5.root.types[ix, ...]
            except:
                type_patch = None

            try:
                sem_patch = h5.root.areas[ix, ...]
            except:
                sem_patch = None
        
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
            which: str="cells",
            binary: bool=False
        ) -> np.ndarray:
        """
        Read class weight arrays that are saved in h5 database.
        h5 db need to be written with the db-writers in this repo for
        this to work

        Args:
        --------
            path (str):
                Path to the hdf5 database
            which (str, default="cells"):
                Compute the weights for either area or cell type classes
            binary (bool, default=False):
                If True, Computes only background vs. foregroung weights

        Returns:
        --------
            np.ndarray: class weights array. Shape (C, )
        """
        assert which in ("cells", "areas")

        path = Path(path)

        if path.suffix == ".h5": 
            with tb.open_file(path.as_posix(), "r") as db:
                if which == "cells":
                    npixels = db.root.npixels_cells[:]
                else:
                    npixels = db.root.npixels_areas[:]

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
        Read dataset statistics arrays that are saved in h5 db.
        h5 db need to be written with the db-writers in this
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
            which: int=-1
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
            which (int, default=-1):
                The epoch number of the saved checkpoint. If -1, uses
                the last epoch 

        Returns:
        ---------
            Path: path to the checkpoint file
        """
        assert isinstance(which, int)

        experiment_dir = Path(f"{RESULT_DIR}/{experiment}/version_{version}")  
        assert experiment_dir.exists(), (
            f"Experiment dir: {experiment_dir.as_posix()} does not exist.",
            "It might be that you have `resume_training` set to True",
            "but you are trying to run the experiment for the first time."
        )

        cpt = None
        for it in experiment_dir.iterdir():
            if it.is_file() and it.suffix == ".ckpt":
                if which == -1 and "last" in str(it) and cpt is None:
                    cpt = it
                elif f"={which}-" in str(it) and cpt is None:
                    cpt = it

        assert cpt is not None, (
            f"ckpt: {cpt}. Checkpoint is None. Make sure that a .ckpt file",
            f"exists in {experiment_dir}. If you dont want to resume training,"
             "check that resume_training = False in experiment.yml"
        )
        return cpt
