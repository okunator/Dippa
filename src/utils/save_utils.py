import scipy.io
import shapely
import geojson
import cv2
import pandas as pd
import numpy as np
import geopandas as gpd

from pathlib import Path
from shapely.geometry import shape
from tqdm import tqdm
from skimage.draw import polygon2mask
from skimage.morphology import remove_small_objects, remove_small_holes
from typing import Dict, Tuple, Union

from .mask_utils import fix_duplicates, get_inst_centroid, get_inst_types, bounding_box


def poly2mask(contour: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    """
    Convert shapely Polygons to np.ndarray mask

    Args:
    ---------
        contour (Polygon.exterior.coords):
            Shapely xy coords for the polygon. Something like array(0 30, 5 50, ... , 90 500)
        shape (tuple):
            shape of the mask  
    """
    nuc = np.asarray(contour) # gdp contour = xy-coord. Need to flip
    inst = polygon2mask(shape, np.flip(nuc, axis=1))
    return inst


def mask2mat(inst_map: np.ndarray, 
             type_map: np.ndarray, 
             fname: Union[str, Path]=None,
             save_dir: Union[str, Path]=None) -> None:
    """
    Convert one set of NN output masks into a .mat file

    Args:
    ---------
        inst_map (np.ndarray):
            instance labelled instance segmentation mask from the segmentation model
        type_map (np.ndarray):
            cell type labelled semantic segmentation mask from the segmentation model
        fname (str, default=None):
            File name for the annotation json file. If None, no file is written.
        save_dir (str):
            directory where the .mat files are saved
    """
    centroids = get_inst_centroid(inst_map)
    inst_ids = list(np.unique(inst_map)[1:])
    bboxes = np.array([bounding_box(np.array(inst_map == id_, np.uint8)) for id_ in inst_ids])
    inst_types = get_inst_types(inst_map, type_map)

    save_dir = Path(save_dir)
    new_fname = Path(fname).with_suffix(".mat").name
    fn_mask = Path(save_dir / new_fname)
    scipy.io.savemat(
        file_name=fn_mask,
        mdict={
            "inst_map": inst_map,
            "type_map":type_map,
            "inst_type":inst_types,
            "inst_centroid":centroids,
            "inst_bbox":bboxes
        }
    )


def geojson2mat(fname: Union[str, Path], classes: Dict[str, int], save_dir: Union[str, Path]=None) -> None:
    """
    Convert geojson annotation file to numpy arrays and save them
    to .mat files

    Args:
    -----------
        fname (str):
            File path to the annotation file
        classes (Dict[str, int], default=None):
            class dict e.g. {"inflam":1, "epithelial":2, "connec":3}
        save_dir (str):
            directory where the .mat files are saved
    """
    # read files and init GeoDf
    anno = pd.read_json(fname)
    anno["geometry"] = anno["geometry"].apply(shape)
    annots = gpd.GeoDataFrame(anno).set_geometry('geometry')

    # inits
    cls_max = max([classes[t] for t in set([prop["classification"]["name"] for prop in annots["properties"]])])
    xmax, ymax = tuple(annots["geometry"].total_bounds[2:].astype("int"))
    target_shape = (ymax+1, xmax+1) # total_bounds -1 smaller than the image
    inst_map = np.zeros(target_shape, np.int32)
    type_map = np.zeros(target_shape, np.int32)

    # loop over the polygons of the GeoDataFrame
    with tqdm(total=len(annots)) as pbar:
        for i, (poly, props) in enumerate(zip(annots["geometry"], annots["properties"])):
            pbar.set_description("processing nuclear annotations")
            class_num = classes[props["classification"]["name"]]
            if isinstance(poly, shapely.geometry.multipolygon.MultiPolygon):
                for p in list(poly):
                    inst = poly2mask(p.exterior.coords, target_shape)
                    inst = remove_small_objects(inst, 10)
                    inst_map[inst > 0] += i
                    type_map[(inst > 0) & (type_map != class_num)] += class_num

            else:
                inst = poly2mask(poly.exterior.coords, target_shape)
                inst = remove_small_objects(inst, 10)
                inst_map[inst > 0] += i
                type_map[(inst > 0) & (type_map != class_num)] += class_num

            # fix overlaps
            inst_map[inst_map > i] = i
            type_map[type_map > cls_max] = cls_max
            pbar.update(1)

        pbar.set_postfix(saving=f"Save results to file: {fname}.mat")
        mask2mat(inst_map, type_map, fname, save_dir)


def mask2geojson(inst_map: np.ndarray, 
                 type_map: np.ndarray, 
                 fname: Union[str, Path]=None,
                 classes: Dict[str, int]=None,
                 save_dir: Union[str, Path]=None,
                 x_offset: int=0,
                 y_offset: int=0) -> Union[Dict, None]:
    """
    Convert one set of NN output masks into a single geoJSON obj

    Args:
    ---------
        inst_map (np.ndarray):
            instance labelled instance segmentation mask from the segmentation model
        type_map (np.ndarray):
            cell type labelled semantic segmentation mask from the segmentation model
        fname (Path or str, default=None):
            File name for the annotation json file. If None, no file is written.
        classes (Dict[str, int], default=None):
            class dict e.g. {"inflam":1, "epithelial":2, "connec":3}
        save_dir (Path or str, default=None):
            directory where the .mat/geojson files are saved
        x_offset (int, default=0):
            x-coordinate offset. (to set geojson to wsi coordinates)
        y_offset (int, default=0):
            y-coordinate offset. (to set geojson to wsi coordinates)

    Returns:
    ----------
        A dictionary with geojson fields or None
    """
    inst_list = list(np.unique(inst_map))[1:]
    geo_objs = []

    # use pannuke classes if no classes are given
    if classes is None:
        classes = {"background":0, "neoplastic":1, "inflammatory":2, "connective":3, "dead":4, "epithelial":5}

    for idx, inst_id in enumerate(inst_list):

        # set up the annotation geojson obj
        geo_obj = {}
        geo_obj.setdefault("type", "Feature")
        geo_obj.setdefault("id", "PathCellAnnotation")
        geo_obj.setdefault("geometry", {"type": "Polygon", "coordinates": None})
        geo_obj.setdefault("properties", {"isLocked": "false", "measurements": [], "classification": {"name": None}})

        # Get cell instance and cell type
        inst = np.array(inst_map == inst_id, np.uint8)
        inst_type = type_map[inst_map == inst_id].astype("uint8")
        inst_type = np.unique(inst_type)[0]
        inst_type = [key for key, val in classes.items() if classes[key] == inst_type][0]

        # get the cell contour coordinates
        contours, hierarchy = cv2.findContours(
            inst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours[0].shape[0] < 3:
            continue
        
        # shift coordinates based on the offsets
        if x_offset:
            contours[0][..., 0] += x_offset
            contours[0][..., 1] += y_offset

        # add the info to the annotation obj
        poly = contours[0].squeeze().tolist()
        poly.append(poly[0]) # close the polygon
        geo_obj["geometry"]["coordinates"] = [poly]
        geo_obj["properties"]["classification"]["name"] = inst_type
        geo_obj["properties"]["classification"]["name"] = inst_type
        geo_objs.append(geo_obj)

    if fname is not None:
        fname = Path(fname).with_suffix(".json").name
        save_dir = Path(save_dir)
        fn = Path(save_dir / fname)
        with open(fn, 'w') as out:
            geojson.dump(geo_objs, out)
        return
    
    return geo_objs


def merge_geojson_dir(in_dir: Union[Path, str], 
                      fname: Union[str, Path], 
                      save_dir: Union[str, Path]) -> None:
    """
    Merge a directory containing geojson files into one big geojson file.

    Args:
    ---------
        in_dir (Path or str):
            in directory
        fname (Path or str):
            File name for the annotation json file. If None, no file is written.
        save_dir (Path or str):
            directory where the .mat files are saved

    """
    gsons = []
    for f in Path(in_dir).iterdir():
        if f.suffix == ".json" and f.is_file():
            # print(f)
            with open(f.as_posix()) as f:
                gson = geojson.load(f)
            gsons.extend(gson)

    fname = Path(fname).with_suffix(".json").name
    save_dir = Path(save_dir)
    fn = Path(save_dir / fname)
    with open(fn, 'w') as out:
        geojson.dump(gsons, out)