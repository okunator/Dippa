import scipy.io
import shapely
import geojson
import cv2
import pandas as pd
import numpy as np
import geopandas as gpd

from pathlib import Path
from tqdm import tqdm
from skimage.draw import polygon2mask
from skimage.morphology import remove_small_objects
from typing import Dict, Tuple, Union

from .mask_utils import get_inst_centroid, get_inst_types, bounding_box
from .file_manager import FileHandler


def poly2mask(
        contour: np.ndarray, 
        shape: Tuple[int, int], 
        x_off: int=None, 
        y_off: int=None
    ) -> np.ndarray:
    """
    Convert shapely Polygons to np.ndarray mask

    Args:
    ---------
        contour (Polygon.exterior.coords):
            Shapely xy coords for the polygon. Something like 
            array(0 30, 5 50, ... , 90 500)
        shape (Tuple[int, int]):
            shape of the mask  
        x_off (int, default=None):
            the amount of translation/offset that is encoded into the 
            x-coord in the geojson
        y_off (int, default=None):
            the amount of translation/offset that is encoded into the
            y-coord in the geojson

    Returns:
    ---------
        np.ndarray: Shape (H, W)
    """
    nuc = np.asarray(contour) # gdp contour = xy-coord. Need to flip

    if x_off is not None:
        nuc[..., 0] -= x_off
    if y_off is not None:
        nuc[..., 1] -= y_off

    inst = polygon2mask(shape, np.flip(nuc, axis=1))
    return inst


def mask2mat(
        inst_map: np.ndarray,
        type_map: np.ndarray,
        fname: Union[str, Path]=None,
        save_dir: Union[str, Path]=None
    ) -> None:
    """
    Convert one set of NN output masks into a .mat file
    that contain (key, value) pairs. 
    
    Keys: "inst_map","type_map","inst_type","inst_centroid","inst_bbox" 

    Args:
    ---------
        inst_map (np.ndarray):
            instance labelled instance segmentation mask from the 
            segmentation model
        type_map (np.ndarray):
            cell type labelled semantic segmentation mask from the
            segmentation model
        fname (str, default=None):
            File name for the annotation .mat file. If None, no file is
            written.
        save_dir (str):
            directory where the .mat files are saved
    """
    save_dir = Path(save_dir)
    fname = Path(fname).with_suffix(".mat").name
    fn_mask = Path(save_dir / fname)
    
    if fname is not None:
        if not Path(save_dir).exists():
            FileHandler.create_dir(save_dir)
    
    centroids = get_inst_centroid(inst_map)
    inst_types = get_inst_types(inst_map, type_map)
    inst_ids = list(np.unique(inst_map)[1:])
    bboxes = np.array(
        [bounding_box(np.array(inst_map == id_, np.uint8)) 
        for id_ in inst_ids]
    )

    scipy.io.savemat(
        file_name=fn_mask.as_posix(),
        mdict={
            "inst_map": inst_map,
            "type_map": type_map,
            "inst_type": inst_types,
            "inst_centroid": centroids,
            "inst_bbox": bboxes
        }
    )


def geojson2mat(
        fname: Union[str, Path], 
        target_shape: Tuple[int, int], 
        classes: Dict[str, int]=None, 
        save_dir: Union[str, Path]=None,
        x_off: int=None, 
        y_off: int=None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts geojson annotation file to numpy arrays and saves them
    to .mat files if save_dir is specified.

    Args:
    -----------
        fname (str):
            File path to the annotation file
        target_shape (Tuple[int, int]):
            Height and width of the numpy array that the geojson is 
            converted into
        classes (Dict[str, int], default=None):
            class dict e.g. {"inflam":1, "epithelial":2, "connec":3}
        save_dir (str):
            directory where the .mat files are saved
        classes (Dict[str, int], default=None):
            class dict e.g. {"inflam":1, "epithelial":2, "connec":3}
        x_off (int):
            the amount of translation/offset that is encoded into the 
            x-coord in the geojson
        y_off (int):
            the amount of translation/offset that is encoded into the 
            y-coord in the geojson

    Retrurns:
    ------------
        Tuple: Tuple on np.ndarrays. The instance segmentation map and 
        the semantic segmentation map
    """
    # read files and init GeoDf
    anno = pd.read_json(fname)
    anno["geometry"] = anno["geometry"].apply(shapely.geometry.shape)
    annots = gpd.GeoDataFrame(anno).set_geometry('geometry')

    # Drop rectangle objects
    drop_ixs = [
        i for i, row in annots.iterrows() 
        if np.isclose(
            row.geometry.minimum_rotated_rectangle.area, 
            row.geometry.area
        )
    ]
    annots = annots.drop(drop_ixs)

    # inits
    # use pannuke classes if no classes are given
    if classes is None:
        classes = {
            "background":0, "neoplastic":1, "inflammatory":2, 
            "connective":3, "dead":4, "epithelial":5
        }

    cls_max = max(
        [
            classes[t] for t in set(
                [
                    prop["classification"]["name"] 
                    for prop in annots["properties"]
                ]
            )
        ]
    )
    target_shape = target_shape
    inst_map = np.zeros(target_shape, np.int32)
    type_map = np.zeros(target_shape, np.int32)

    # loop over the polygons of the GeoDataFrame
    with tqdm(total=len(annots)) as pbar:
        for i, (poly, props) in enumerate(zip(
            annots["geometry"], 
            annots["properties"])
        ):
            pbar.set_description("processing nuclear annotations")
            class_num = classes[props["classification"]["name"]]

            if isinstance(poly, shapely.geometry.MultiPolygon):
                # handle multipolygons exceptions
                for p in list(poly):
                    inst = poly2mask(
                        p.exterior.coords, target_shape, x_off, y_off
                    )
                    inst = remove_small_objects(inst, 10)
                    inst_map[inst > 0] += (i + 1)
                    type_map[(inst > 0) & (type_map != class_num)] = class_num
            elif isinstance(poly, shapely.geometry.LineString):
                # handle linestring exceptions
                coords = poly.coords[:]
                coords.append(coords[-1])
                p = shapely.geometry.polygon.Polygon(coords)
                inst = poly2mask(p.exterior.coords, target_shape, x_off, y_off)
                inst = remove_small_objects(inst, 10)
                inst_map[inst > 0] += (i + 1)
                type_map[(inst > 0) & (type_map != class_num)] = class_num

            else:
                inst = poly2mask(
                    poly.exterior.coords, target_shape, x_off, y_off
                )
                inst = remove_small_objects(inst, 10)
                inst_map[inst > 0] += (i + 1)
                type_map[(inst > 0) & (type_map != class_num)] = class_num

            # fix overlaps
            inst_map[inst_map > (i + 1)] = i + 1
            type_map[type_map > cls_max] = class_num
            pbar.update(1)

        pbar.set_postfix(saving=f"Save results to file: {fname[:-4]}.mat")

        if save_dir is not None:
            mask2mat(inst_map, type_map, fname, save_dir)

    return inst_map, type_map


def mask2geojson(
        inst_map: np.ndarray, 
        type_map: np.ndarray, 
        classes: Dict[str, int],
        fname: Union[str, Path]=None,
        save_dir: Union[str, Path]=None,
        x_offset: int=0,
        y_offset: int=0
    ) -> Union[Dict, None]:
    """
    Convert one set of NN output masks into a single geoJSON obj

    Args:
    ---------
        inst_map (np.ndarray):
            instance labelled instance segmentation mask from the 
            segmentation model
        type_map (np.ndarray):
            cell type labelled semantic segmentation mask from the 
            segmentation model
        classes (Dict[str, int]):
            class dict e.g. {"inflam":1, "epithelial":2, "connec":3}
        fname (Path or str, default=None):
            File name for the annotation json file. If None, no file is 
            written.
        save_dir (Path or str, default=None):
            directory where the .mat/geojson files are saved
        x_offset (int, default=0):
            x-coordinate offset. (to set geojson to wsi coordinates)
        y_offset (int, default=0):
            y-coordinate offset. (to set geojson to wsi coordinates)

    Returns:
    ----------
        Dict: A dictionary with geojson fields or None
    """

    if fname is not None:
        if not Path(save_dir).exists():
            FileHandler.create_dir(save_dir)

    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)
    
    geo_objs = []
    for inst_id in inst_list:
        # set up the annotation geojson obj
        geo_obj = {}
        geo_obj.setdefault("type", "Feature")

        # PathCellAnnotation, PathCellDetection, PathDetectionObject
        geo_obj.setdefault("id", "PathCellDetection") 
        geo_obj.setdefault(
            "geometry", {"type": "Polygon", "coordinates": None}
        )
        geo_obj.setdefault(
            "properties", {
                "isLocked": "false", "measurements": [], 
                "classification": {"name": None}
            }
        )

        # Get cell instance and cell type
        inst = np.array(inst_map == inst_id, np.uint8)
        inst_type = type_map[inst_map == inst_id].astype("uint8")
        inst_type = np.unique(inst_type)[0]
        
        inst_type = [
            key for key in classes.keys() if classes[key] == inst_type
        ][0]

        # get the cell contour coordinates
        contours, _ = cv2.findContours(
            inst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # got a line instead of a polygon
        if contours[0].shape[0] < 3:
            continue
        
        # shift coordinates based on the offsets
        if x_offset:
            contours[0][..., 0] += x_offset
        if y_offset:
            contours[0][..., 1] += y_offset

        # add the info to the annotation obj
        poly = contours[0].squeeze().tolist()
        poly.append(poly[0]) # close the polygon
        geo_obj["geometry"]["coordinates"] = [poly]
        geo_obj["properties"]["classification"]["name"] = inst_type
        geo_objs.append(geo_obj)

    if fname is not None:
        fname = Path(fname).with_suffix(".json").name
        save_dir = Path(save_dir)

        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True)

        fn = Path(save_dir / fname)
        with fn.open('w') as out:
            geojson.dump(geo_objs, out)
        return

    return geo_objs