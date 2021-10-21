import re
import numpy as np
import geopandas as gpd

from typing import Union, Tuple, Dict
from pathlib import Path


class BaseGSONMerger:
    def __init__(
            self, 
            in_dir: Union[Path, str],
            xmax: int=None, 
            ymax: int=None, 
            xmin: int=None, 
            ymin: int=None,
            tile_size: Tuple[int, int]=(1000, 1000)
        ) -> None:
        """
        Base class for Geojson merger objects,

        Example files names:
        "x-45000_y-56000.json", "x-34000_y-16000.json" etc.

        Args:
        ---------
            in_dir (Path or str):
                Input directory of geojson files
            xmax (int, default=None):
                the max-x coordinate of the tiled images
            ymax (int, default=None):
                the max-y coordinate of the tiled images
            xmin (int, default=None):
                the min-x coordinate of the tiled images
            ymin (int, default=None):
                the min-y coordinate of the tiled images
            tile_size (Tuple[int, int], default=(1000, 1000)):
                size of the input tiles
        """
        self.xmax = xmax
        self.ymax = ymax
        self.xmin = xmin
        self.ymin = ymin
        self.tile_size = tile_size
        self.files = sorted(Path(in_dir).glob("*"))
        assert self.files, f"No files found in: {in_dir}"
        
    @property
    def _gsonobj(self) -> Dict:
        geo_obj = {}
        geo_obj.setdefault("type", "Feature")

        # PathDetectionObject, PathCellDetection, PathCellAnnotation
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
        return geo_obj

    @staticmethod
    def drop_rectangles(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Drop rectangular objects from a gpd.GeoDataFrame.

        Args:
        --------
            gdf (gpd.GeoDataFrame):
                The input GeoDataFrame.
        
        Returns:
        --------
            gpd.GeoDataFrame without rectangular objects
        """
        ixs = [
            i for i, row in gdf.iterrows() 
            if np.isclose(
                row.geometry.minimum_rotated_rectangle.area, 
                row.geometry.area
            )
        ]
        return gdf.drop(ixs)

    def _get_xy_coords(self, fname: str) -> Tuple[int, int]:
        """
        fname needs to contain x & y-coordinates in 
        "x-[coord1]_y-[coord2]"-format
        """
        if isinstance(fname, Path):
            fname = fname.as_posix()
            
        assert re.findall(r"(x-\d+_y-\d+)", fname), (
            "fname not in 'x-[coord1]_y-[coord2]'-format"
        )
        xy_str = re.findall(r"(x-\d+_y-\d+)", fname)
        x, y = (int(c) for c in re.findall(r"\d+", xy_str[0]))
        return x, y

    def _get_file_from_coords(self, x: int, y: int) -> Union[str, None]:
        """
        Get the file name from the given coords

        fname needs to contain x & y-coordinates in 
        "x-[coord1]_y-[coord2]"-format

        Args:
        ---------
            x (int):
                x-coord
            y (int):
                y-coord

        Returns:
        ---------
            str or None: returns the file name if it exists in the
                         given input dir, else None.
        """
        f = [f for f in self.files if f"x-{x}_y-{y}" in f.name]
        ret = f[0] if f else None
        
        return ret
    
    def _get_right_neighbor(self, fname: str) -> Path:
        x, y = self._get_xy_coords(fname)
        x += self.tile_size[0]
        return self._get_file_from_coords(x, y)

    def _get_left_neighbor(self, fname: str) -> Path:
        x, y = self._get_xy_coords(fname)
        x -= self.tile_size[0]
        return self._get_file_from_coords(x, y)

    def _get_bottom_neighbor(self, fname: str) -> Path:
        x, y = self._get_xy_coords(fname)
        y += self.tile_size[1]
        return self._get_file_from_coords(x, y)

    def _get_bottom_right_neighbor(self, fname: str) -> Path:
        x, y = self._get_xy_coords(fname)
        x += self.tile_size[0]
        y += self.tile_size[1]
        return self._get_file_from_coords(x, y)

    def _get_bottom_left_neighbor(self, fname: str) -> Path:
        x, y = self._get_xy_coords(fname)
        x -= self.tile_size[0]
        y += self.tile_size[1]
        return self._get_file_from_coords(x, y)
    
    def _get_top_neighbor(self, fname: str) -> Path:
        x, y = self._get_xy_coords(fname)
        y -= self.tile_size[1]
        return self._get_file_from_coords(x, y)

    def _get_top_right_neighbor(self, fname: str) -> Path:
        x, y = self._get_xy_coords(fname)
        x += self.tile_size[0]
        y -= self.tile_size[1]
        return self._get_file_from_coords(x, y)

    def _get_top_left_neighbor(self, fname: str) -> Path:
        x, y = self._get_xy_coords(fname)
        x -= self.tile_size[0]
        y -= self.tile_size[1]
        return self._get_file_from_coords(x, y)

    def _get_adjascent_tiles(self, fname: str) -> Dict[str, Path]:
        adj = {}
        adj["left"] = self._get_left_neighbor(fname)
        adj["right"] = self._get_right_neighbor(fname)
        adj["bottom"] = self._get_bottom_neighbor(fname)
        adj["top"] = self._get_top_neighbor(fname)
        
        # adj["bottom_right"] = self._get_bottom_right_neighbor(fname)
        # adj["bottom_left"] = self._get_bottom_left_neighbor(fname)
        # adj["top_right"] = self._get_top_right_neighbor(fname)
        # adj["top_left"] = self._get_top_left_neighbor(fname)

        return adj    