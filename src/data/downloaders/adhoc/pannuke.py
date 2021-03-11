import cv2
import numpy as np
import scipy.io
from pathlib import Path 
from distutils import dir_util, file_util
from typing import Dict
from tqdm import tqdm

from src.utils import (
    FileHandler, 
    get_inst_centroid, 
    bounding_box, 
    get_inst_types,
    fix_duplicates
)


def get_type_map(pannuke_mask: np.ndarray) -> np.ndarray:
    """
    Convert the pannuke mask to type map

    Args:
    -----------
        pannuke_mask (np.ndarray):
            The pannuke mask of shape (H, W, 6), where each chl
            contains an instance map

    Returns:
    ---------- 
        np.ndarray label indices. Shape (H, W).
    """
    mask = pannuke_mask.copy()
    mask[mask > 0] = 1

    # init type_map and set the background channel
    # of the pannuke mask as the first channel
    type_map = np.zeros_like(mask)
    type_map[..., 0] = mask[..., -1]
    for i, j in enumerate(range(1, mask.shape[-1])):
        type_map[..., j] = mask[..., i]
    
    return np.argmax(type_map, axis=-1)


def get_inst_map(pannuke_mask: np.ndarray) -> np.ndarray:
    """
    Convert pannuke mask to inst_map of shape (H, W).

    Args:
    -----------
        pannuke_mask (np.ndarray):
            The pannuke mask of shape (H, W, 6), where each chl
            contains an instance map

    Returns:
    ---------- 
        np.ndarray labelled nuclear instances. Shape (H, W).
    """
    mask = pannuke_mask.copy()

    inst_map = np.zeros(mask.shape[:2], dtype="int32")
    for i in range(mask.shape[-1]):
        
        insts = mask[..., i]
        inst_ids = np.unique(insts)[1:]
        for inst_id in inst_ids:
            inst = np.array(insts == inst_id, np.uint8)
            inst_map[inst > 0] += inst_id
    
    # fix duplicated instances
    inst_map = fix_duplicates(inst_map)
    return inst_map


def handle_pannuke(orig_dir: Path,
                   imgs_train_dir: Path, 
                   anns_train_dir: Path, 
                   imgs_test_dir: Path, 
                   anns_test_dir: Path,
                   fold_num: int,
                   fold_phase: str) -> None:
    """
    Pannuke fold patches are saved in a numpy .npy file. This converts them to the same
    .mat (consep) format and saves each fold to a specific directory in the parent dir
    of the 'orig_dir' directory. 

    Args:
    ----------
        orig_dir (Path): 
            The path where the extracted .zip files are located
        imgs_train_dir (Path): 
            Path to the directory where the training images are saved
        anns_train_dir (Path): 
            Path to the directory where the training gt annotations are saved
        imgs_test_dir (Path): 
            Path to the directory where the testing images are saved
        anns_test_dir (Path): 
            Path to the directory where the testing gt annotations are saved
        fold_num (int): 
            the fold number
        fold_phase (str):
            One of ("train", "valid", "test")
    """
    
    # Create directories for the files
    FileHandler.create_dir(anns_train_dir)
    FileHandler.create_dir(imgs_train_dir)
    FileHandler.create_dir(anns_test_dir)
    FileHandler.create_dir(imgs_test_dir)
    
    # Dict["fold{i}_{images/masks/types}", file]
    fold_files = {
        f"{file.parts[-2]}_{file.name[:-4]}": file for dir1 in orig_dir.iterdir() if dir1.is_dir()
        for dir2 in dir1.iterdir() if dir2.is_dir()
        for dir3 in dir2.iterdir() if dir3.is_dir()
        for file in dir3.iterdir() if file.is_file() and file.suffix == ".npy"
    }
    
    # Dict["phase/fold", Dict["img/mask", Path("path/to/dir")]]
    fold_phases = {
        "train":{
            "img":imgs_train_dir,
            "mask":anns_train_dir
        },
        "valid":{
            "img":imgs_train_dir,
            "mask":anns_train_dir
        },
        "test":{
            "img":imgs_test_dir,
            "mask":anns_test_dir
        }
    }
     
    # load data
    masks = np.load(fold_files[f"fold{fold_num}_masks"]).astype("int32")
    imgs = np.load(fold_files[f"fold{fold_num}_images"]).astype("uint8")
    types = np.load(fold_files[f"fold{fold_num}_types"])
        
    with tqdm(total=len(types)) as pbar: # number of patches in the whole dataset 7901
        # loop by tissue
        for tissue_type in np.unique(types):

            # tqdm logs
            pbar.set_postfix(
                info=f"Converting {tissue_type} patches from fold_{fold_num} .npy file to .png/.mat files"
            ) 

            imgs_by_type = imgs[types == tissue_type]
            masks_by_type = masks[types == tissue_type]
            for j in range(imgs_by_type.shape[0]):
                name = f"{tissue_type}_fold{fold_num}_{j}"
                img_dir = fold_phases[fold_phase]["img"]
                mask_dir = fold_phases[fold_phase]["mask"]

                fn_im = Path(img_dir / name).with_suffix(".png")
                cv2.imwrite(str(fn_im), cv2.cvtColor(imgs_by_type[j, ...], cv2.COLOR_RGB2BGR))
                
                # Create inst and type maps
                temp_mask = masks_by_type[j, ...]
                type_map = get_type_map(temp_mask)
                inst_map = get_inst_map(temp_mask[..., 0:5])

                # add other data
                centroids = get_inst_centroid(inst_map)
                inst_ids = list(np.unique(inst_map)[1:])
                bboxes = np.array([bounding_box(np.array(inst_map == id_, np.uint8)) for id_ in inst_ids])
                inst_types = get_inst_types(inst_map, type_map)

                # save results
                fn_mask = Path(mask_dir / name).with_suffix(".mat")
                scipy.io.savemat(
                    file_name=fn_mask, 
                    mdict={
                        "inst_map": inst_map,
                        "type_map":type_map,
                        "inst_type":inst_types,
                        "inst_centroid":centroids,
                        "inst_bbox":bboxes
                    })

                pbar.update(1)