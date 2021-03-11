import cv2
import numpy as np
import scipy.io
import xml.etree.ElementTree as ET
from pathlib import Path
from distutils import dir_util, file_util

from src.utils import FileHandler


# Adapted from https://github.com/vqdang/hover_net/blob/master/src/misc/proc_kumar_ann.py
def kumar_xml2mat(x: Path, to: Path) -> None:
    """
    Convert the xml annotation files to .mat files with 'inst_map' and.

    Args:
    -----------
        x (Path): 
            Path to the xml file
        to (Path): 
            Directory where the .mat file is written
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
    scipy.io.savemat(mask_fn, mdict={"inst_map": ann, "type_map": type_map})


def handle_kumar(orig_dir: Path,
                 imgs_train_dir: Path,
                 anns_train_dir: Path,
                 imgs_test_dir: Path,
                 anns_test_dir: Path) -> None:
    """
    This traverses the files downloaded online and saves each fold to a 
    specific directory in the parent dir of the 'orig_dir' directory.

    Args:
    -----------
        orig_dir (Path): 
            The path where the .zip files are located
        imgs_train_dir (Path): 
            Path to the directory where the training images are saved
        anns_train_dir (Path):
            Path to the directory where the training gt annotations are saved
        imgs_test_dir (Path): 
            Path to the directory where the testing images are saved
        anns_test_dir (Path): 
            Path to the directory where the testing gt annotations are saved
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