import pandas as pd
import numpy as np
import geopandas as gpd
import scipy.io
import shapely


from pathlib import Path
from shapely.geometry import shape
from tqdm import tqdm
from skimage.draw import polygon2mask
from skimage.morphology import remove_small_objects, remove_small_holes
from typing import Dict, Tuple, Union

from .mask_utils import fix_duplicates, get_inst_centroid, get_inst_types, bounding_box


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


def geojson2mask(fname: Union[str, Path], classes: Dict[str, int], save_dir: Union[str, Path]=None) -> None:
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
    inst_mask = np.zeros(target_shape, np.int32)
    type_mask = np.zeros(target_shape, np.int32)

    # loop over the polygons of the GeoDataFrame
    with tqdm(total=len(annots)) as pbar:
        for i, (poly, props) in enumerate(zip(annots["geometry"], annots["properties"])):
            pbar.set_description("processing nuclear annotations")
            class_num = classes[props["classification"]["name"]]
            if isinstance(poly, shapely.geometry.multipolygon.MultiPolygon):
                for p in list(poly):
                    inst = poly2mask(p.exterior.coords, target_shape)
                    inst = remove_small_objects(inst, 10)
                    inst_mask[inst > 0] += i
                    type_mask[(inst > 0) & (type_mask != class_num)] += class_num

            else:
                inst = poly2mask(poly.exterior.coords, target_shape)
                inst = remove_small_objects(inst, 10)
                inst_mask[inst > 0] += i
                type_mask[(inst > 0) & (type_mask != class_num)] += class_num

            # fix overlaps
            inst_mask[inst_mask > i] = i
            type_mask[type_mask > cls_max] = cls_max
            pbar.update(1)

        # add other metrics
        pbar.set_postfix(centroids=f"Compute centroids...")
        centroids = get_inst_centroid(inst_mask)
        inst_ids = list(np.unique(inst_mask)[1:])
        pbar.set_postfix(bboxes=f"Compute bboxes...")
        bboxes = np.array([bounding_box(np.array(inst_mask == id_, np.uint8)) for id_ in inst_ids])
        pbar.set_postfix(types=f"Compute inst types...")
        inst_types = get_inst_types(inst_mask, type_mask)

        # save to .mat
        save_dir = Path(save_dir)
        new_fname = Path(fname).with_suffix(".mat").name
        fn_mask = Path(save_dir / new_fname)
        pbar.set_postfix(saving=f"Save results to file: {new_fname}")
        scipy.io.savemat(
            file_name=fn_mask,
            mdict={
                "inst_map": inst_mask,
                "type_map":type_mask,
                "inst_type":inst_types,
                "inst_centroid":centroids,
                "inst_bbox":bboxes
            }
        )