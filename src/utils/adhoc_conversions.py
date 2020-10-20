from pathlib import Path 
import numpy as np
import scipy.io
import cv2
import xml.etree.ElementTree as ET
from distutils import dir_util, file_util
from typing import Dict
from src.utils.file_manager import FileHandler


# Not optimized. Not a bottleneck
def kumar_xml2mat(x: Path, to: Path) -> None:
    """
    From https://github.com/vqdang/hover_net/blob/master/src/misc/proc_kumar_ann.py
    Convert the xml annotation files to .mat files with 'inst_map' and.

    Args:
        x (Path): path to the xml file
        to (Path): directory where the .mat file is written
    """
    xml = ET.parse(x)
    insts_list = []
    for idx, region_xml in enumerate(xml.findall('.//Region')):
        vertices = []
        for vertex_xml in region_xml.findall('.//Vertex'):
            attrib = vertex_xml.attrib
            vertices.append([float(attrib['X']),
                                float(attrib['Y'])])
        vertices = np.array(vertices) + 0.5
        vertices = vertices.astype('int32')
        contour_blb = np.zeros((1000, 1000), np.uint8)

        # fill both the inner area and contour with idx+1 color
        cv2.drawContours(contour_blb, [vertices], 0, idx+1, -1)
        insts_list.append(contour_blb)

    insts_size_list = np.array(insts_list)
    insts_size_list = np.sum(insts_size_list, axis=(1, 2))
    insts_size_list = list(insts_size_list)
    pair_insts_list = zip(insts_list, insts_size_list)

    # sort in z-axis basing on size, larger on top
    pair_insts_list = sorted(pair_insts_list, key=lambda x: x[1])
    insts_list, insts_size_list = zip(*pair_insts_list)
    ann = np.zeros((1000, 1000), np.int32)

    for idx, inst_map in enumerate(insts_list):
        ann[inst_map > 0] = idx + 1

    # binary mask as type_map
    type_map = np.copy(ann)
    type_map[ann > 0] = 1

    mask_fn = Path(to / x.with_suffix(".mat").name)
    scipy.io.savemat(
        mask_fn, mdict={"inst_map": ann, "type_map": type_map})


def handle_kumar(orig_dir: Path,
                 imgs_train_dir: Path,
                 anns_train_dir: Path,
                 imgs_test_dir: Path,
                 anns_test_dir: Path) -> None:
    """
    This converts the original files to the same HoVer-Net paper format and saves each fold 
    to a specific directory in the parent dir of the 'orig_dir' directory.

    Args:
        orig_dir (Path): the path where the .zip files are located
        imgs_train_dir (Path): path to the directory where the training images are saved
        anns_train_dir (Path): path to the directory where the training gt annotations are saved
        imgs_test_dir (Path): path to the directory where the testing images are saved
        anns_test_dir (Path): path to the directory where the testing gt annotations are saved
    """
    FileHandler.create_dir(anns_train_dir)
    FileHandler.create_dir(imgs_train_dir)
    FileHandler.create_dir(anns_test_dir)
    FileHandler.create_dir(imgs_test_dir)
    
    for f in orig_dir.iterdir():
        if f.is_dir() and "training" in f.name.lower():
            for item in f.iterdir():
                if item.name == "Tissue Images":
                    # item.rename(item.parents[2]/"train") #cut/paste
                    dir_util.copy_tree(str(item), str(imgs_train_dir)) #copy/paste
                elif item.name == "Annotations":
                    for ann in item.iterdir():
                        kumar_xml2mat(ann, anns_train_dir)
                        
        elif f.is_dir() and "test" in f.name.lower():
            for ann in f.glob("*.xml"):
                kumar_xml2mat(ann, anns_test_dir)
            for item in f.glob("*.tif"):
                file_util.copy_file(str(item), str(imgs_test_dir))


def handle_consep(orig_dir: Path,
                  imgs_train_dir: Path,
                  anns_train_dir: Path,
                  imgs_test_dir: Path,
                  anns_test_dir: Path) -> None:
    """
    Moves the consep files to the "consep" directory and creates train and testing
    folders for the data. Changes the classes to be the same as in the paper if 
    change_classes is set to True

    Args:
        orig_dir (Path): the path where the .zip files are located
        imgs_train_dir (Path): path to the directory where the training images are saved
        anns_train_dir (Path): path to the directory where the training gt annotations are saved
        imgs_test_dir (Path): path to the directory where the testing images are saved
        anns_test_dir (Path): path to the directory where the testing gt annotations are saved
    """
    FileHandler.create_dir(anns_train_dir)
    FileHandler.create_dir(imgs_train_dir)
    FileHandler.create_dir(anns_test_dir)
    FileHandler.create_dir(imgs_test_dir)

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
                    
    
def handle_pannuke(orig_dir: Path,
                   imgs_train_dir: Path, 
                   anns_train_dir: Path, 
                   imgs_test_dir: Path, 
                   anns_test_dir: Path,
                   pannuke_folds: Dict[str, str]) -> None:
    """
    Pannuke fold patches are saved in a numpy .npy file. This converts them to the same
    HoVer-Net paper format and saves each fold to a specific directory in the parent dir
    of the 'orig_dir' directory. Each one of the patches is written to a .mat file with
    the same keys as are in the consep dataset. 

    Args:
        orig_dir (Path): the path where the .zip files are located
        imgs_train_dir (Path): path to the directory where the training images are saved
        anns_train_dir (Path): path to the directory where the training gt annotations are saved
        imgs_test_dir (Path): path to the directory where the testing images are saved
        anns_test_dir (Path): path to the directory where the testing gt annotations are saved
        pannuke_folds (Dict[str, str]): the folds dict for setting pannuke folds e.g. 
                      {fold1: train, fold2:valid, fold3:test}. pannuke.yaml is to be modified
                      to change the folds.
    """
    
    FileHandler.create_dir(anns_train_dir)
    FileHandler.create_dir(imgs_train_dir)
    FileHandler.create_dir(anns_test_dir)
    FileHandler.create_dir(imgs_test_dir)
    
    fold_files = {
        f"{file.parts[-2]}_{file.name[:-4]}": file for dir1 in orig_dir.iterdir() if dir1.is_dir()
        for dir2 in dir1.iterdir() if dir2.is_dir()
        for dir3 in dir2.iterdir() if dir3.is_dir()
        for file in dir3.iterdir() if file.is_file() and file.suffix == ".npy"
    }
    
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
    
    for i in range(1, 4):
        masks = np.load(fold_files[f"fold{i}_masks"]).astype("int32")
        imgs = np.load(fold_files[f"fold{i}_images"]).astype("uint8")
        types = np.load(fold_files[f"fold{i}_types"])
        
        for ty in np.unique(types):
            imgs_by_type = imgs[types == ty]
            masks_by_type = masks[types == ty]
            for j in range(imgs_by_type.shape[0]):
                name = f"{ty}_fold{i}_{j}"
                dir_key = pannuke_folds[f"fold{i}"]
                img_dir = fold_phases[dir_key]["img"]
                mask_dir = fold_phases[dir_key]["mask"]

                fn_im = Path(img_dir / name).with_suffix(".png")
                cv2.imwrite(str(fn_im), cv2.cvtColor(imgs_by_type[j, ...], cv2.COLOR_RGB2BGR))
                
                # Create inst- and type maps
                temp_mask = masks_by_type[j, ...]
                inst_map = np.zeros(temp_mask.shape[:2], dtype=np.int32)
                type_map = np.zeros(temp_mask.shape[:2], dtype=np.int32)
                for t, l in enumerate(range(temp_mask.shape[-1]-1), 1):
                    inst_map += temp_mask[..., l]
                    temp_type = np.copy(temp_mask[..., l])
                    temp_type[temp_type > 0] = t
                    type_map += temp_type
                    
                # if two cells overlap, adding both classes to type_map will create
                # a sum of those classes for those pixels and things would break so
                # we'll just remove the overlaps here.
                type_map[type_map > temp_mask.shape[-1]-1] = 0
                fn_mask = Path(mask_dir / name).with_suffix(".mat")
                scipy.io.savemat(
                    fn_mask, 
                    mdict={
                        "inst_map": inst_map,
                        "type_map":type_map
                    })


def handle_dsb2018() -> None:
    pass


def handle_cpm() -> None:
    pass

