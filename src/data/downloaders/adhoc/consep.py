import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
from distutils import dir_util, file_util
from tqdm import tqdm

from src.utils import (
    FileHandler, 
    get_inst_centroid, 
    bounding_box, 
    get_inst_types,
    fix_duplicates
)


def convert_consep_classes(type_map: np.ndarray) -> np.ndarray:
    """
    Convert CoNSeP classes 3 & 4 into class 3
    and classes 5 & 6 & 7 into class 4.  
    """
    type_map[(type_map == 3) | (type_map == 4)] = 3
    type_map[(type_map == 5) | (type_map == 6) | (type_map == 7)] = 4
    return type_map


def handle_consep(orig_dir: Path,
                  imgs_train_dir: Path,
                  anns_train_dir: Path,
                  imgs_test_dir: Path,
                  anns_test_dir: Path,
                  convert_classes: bool=True) -> None:
    """
    Moves the consep files to the "consep" directory and creates train and testing
    folders for the data.

    Args:
    -----------
        orig_dir (Path): 
            The path where the consep.zip file was extracted 
        imgs_train_dir (Path): 
            Path to the directory where the training images are saved
        anns_train_dir (Path): 
            Path to the directory where the training gt annotations are saved
        imgs_test_dir (Path): 
            Path to the directory where the testing images are saved
        anns_test_dir (Path): 
            Path to the directory where the testing gt annotations are saved
        convert_classes (bool, default=True):
            Convert CoNSeP dataset classes same way they did in the paper
    """
    FileHandler.create_dir(anns_train_dir)
    FileHandler.create_dir(imgs_train_dir)
    FileHandler.create_dir(anns_test_dir)
    FileHandler.create_dir(imgs_test_dir)

    # Copy the orig directory tree to the new folders
    for item in orig_dir.iterdir():
        if item.is_dir() and item.name == "CoNSeP":
            for d in item.iterdir():
                if d.is_dir() and d.name == "Test":
                    for g in d.iterdir():
                        if g.name == "Images":
                            dir_util.copy_tree(str(g), str(imgs_test_dir))
                        elif g.name == "Labels":
                            dir_util.copy_tree(str(g), str(anns_test_dir))
                elif d.is_dir() and d.name == "Train":
                    for g in d.iterdir():
                        if g.name == "Images":
                            dir_util.copy_tree(str(g), str(imgs_train_dir))
                        elif g.name == "Labels":
                            dir_util.copy_tree(str(g), str(anns_train_dir))

    # Add data and convert classes
    total = len(list(anns_train_dir.iterdir())) + len(list(anns_test_dir.iterdir()))
    with tqdm(total=total) as pbar:
        for f in anns_train_dir.iterdir():
            pbar.set_postfix(info= f"Processing mask: {f.name}")

            inst_map = FileHandler.read_mask(f, key="inst_map")
            type_map = FileHandler.read_mask(f, key="type_map")
            
            if convert_classes:
                type_map = convert_consep_classes(type_map)

            centroids = get_inst_centroid(inst_map)
            inst_ids = list(np.unique(inst_map)[1:])
            bboxes = np.array([bounding_box(np.array(inst_map == id_, np.uint8)) for id_ in inst_ids])
            inst_types = get_inst_types(inst_map, type_map)

            savemat(
                file_name=f.as_posix(), 
                mdict={
                    "inst_map": inst_map,
                    "type_map":type_map,
                    "inst_type":inst_types,
                    "inst_centroid":centroids,
                    "inst_bbox":bboxes
                }
            )
            pbar.update(1)

        # Add data and convert classes
        for f in anns_test_dir.iterdir():
            pbar.set_postfix(info= f"Processing mask: {f.name}")
            inst_map = FileHandler.read_mask(f, key="inst_map")
            type_map = FileHandler.read_mask(f, key="type_map")
            
            if convert_classes:
                type_map = convert_consep_classes(type_map)

            centroids = get_inst_centroid(inst_map)
            inst_ids = list(np.unique(inst_map)[1:])
            bboxes = np.array([bounding_box(np.array(inst_map == id_, np.uint8)) for id_ in inst_ids])
            inst_types = get_inst_types(inst_map, type_map)

            savemat(
                file_name=f.as_posix(), 
                mdict={
                    "inst_map": inst_map,
                    "type_map":type_map,
                    "inst_type":inst_types,
                    "inst_centroid":centroids,
                    "inst_bbox":bboxes
                }
            )
            pbar.update(1)

                    