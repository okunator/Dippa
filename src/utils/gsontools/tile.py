import re
import warnings
import shapely
import pandas as pd
import geopandas as gpd

from typing import Union, Tuple
from pathlib import Path


class GSONTile:
    def __init__(
            self, 
            fname: Union[Path, str], 
            xmin: int=None, 
            ymin: int=None, 
            tile_size: int=(1000, 1000)
        ) -> None:
        """
        Class abstaraction for one tile of geojson annotations.

        Args:
        ----------
            fname: (Path or str):
                name of the gson file
            xmin (int, default=None):
                the min-x coordinate of the tile
            ymin (int, default=None):
                the min-y coordinate of the tile
            tile_size (Tuple[int, int], default=(1000, 1000)):
                size of the input tile
        """
        assert Path(fname).exists(), f"File {fname} not found."
        
        self.fname = Path(fname)

        if None in (xmin, ymin):
            xmin, ymin = self._get_xy_coords(self.fname)
            
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmin + tile_size[0]
        self.ymax = ymin + tile_size[1]


    def __len__(self):
        """
        Return the number of polygons in the json file
        """
        df = pd.read_json(self.fname)
        l = len(df)

        # empty mem
        del df

        return l

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

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """
        Read a geojson/json file and convert it to geopandas df
        Adds bounding box-coords for the polygons
        """

        # try:
        df = pd.read_json(self.fname)

        if df.empty:
            warnings.warn(
                f"No annotations detected in {self.fname}. Returning None.",                
                RuntimeWarning
            )

            return None

        df["geometry"] = df["geometry"].apply(shapely.geometry.shape)
        gdf = gpd.GeoDataFrame(df).set_geometry('geometry')
        gdf = gdf[~gdf.is_empty] # drop empty geometries
        gdf = gdf[gdf.is_valid] # drop invalid geometries

        try:
            # add bounding box coords of the polygons to the gdfs
            # and correct for the max coords
            gdf["xmin"] = gdf.bounds["minx"].astype(int)
            gdf["ymin"] = gdf.bounds["miny"].astype(int)
            gdf["ymax"] = gdf.bounds["maxy"].astype(int) + 1
            gdf["xmax"] = gdf.bounds["maxx"].astype(int) + 1
        except:
            warnings.warn(
                "Could not create bounds cols to gdf", RuntimeWarning
            )

        try:
            # add class name column
            gdf["class_name"] = gdf["properties"].apply(
                lambda x: x["classification"]["name"]
            )
        except:
            warnings.warn(
                "Could not create 'class_name' col to gdf.", RuntimeWarning
            )

        return gdf

    @property
    def non_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that do not touch any edges
        of the tile
        """
        not_r = self.gdf["xmax"] !=  self.xmax
        not_l = self.gdf["xmin"] !=  self.xmin
        not_b = self.gdf["ymax"] !=  self.ymax
        not_t = self.gdf["ymin"] !=  self.ymin
        non_border_annots = self.gdf[not_r & not_l & not_b & not_t].copy()
        non_border_annots.loc[:, "geometry"] = non_border_annots
        non_border_annots = non_border_annots.reset_index(drop=True)
        return non_border_annots
    
    @property
    def right_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the right edge
        of the tile
        """
        r_border_anns = self.gdf[self.gdf["xmax"] ==  self.xmax].copy()

        # translate one unit right
        r_border_anns.loc[:, "geometry"] = r_border_anns.translate(xoff=1.0)
        r_border_anns = r_border_anns.reset_index(drop=True)
        return r_border_anns

    @property
    def left_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the left edge
        of the tile
        """
        l_border_anns = self.gdf[self.gdf["xmin"] ==  self.xmin].copy()
        l_border_anns.loc[:, "geometry"] = l_border_anns
        l_border_anns = l_border_anns.reset_index(drop=True)
        return l_border_anns

    @property
    def bottom_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the bottom edge
        of the tile. (Origin in the top-left corner of the image/tile)
        """
        b_border_anns = self.gdf[self.gdf["ymax"] ==  self.ymax].copy()
        # translate 1-unit down
        b_border_anns.loc[:, "geometry"] = b_border_anns.translate(yoff=1.0) 
        b_border_anns = b_border_anns.reset_index(drop=True)
        return b_border_anns

    @property
    def bottom_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the bottom edge
        of the tile. (Origin in the top-left co            for r in gdf.iterrows():
                rrner of the image/tile)
        """
        b_border_anns = self.gdf[self.gdf["ymax"] ==  self.ymax].copy()
        # translate 1-unit down
        b_border_anns.loc[:, "geometry"] = b_border_anns.translate(yoff=1.0) 
        b_border_anns = b_border_anns.reset_index(drop=True)
        return b_border_anns

    @property
    def bottom_left_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the bottom edge
        of the tile. (Origin in the top-left corner of the image/tile)
        """
        b = (self.gdf["ymax"] ==  self.ymax) & (self.gdf["xmin"] ==  self.xmin)
        bl_border_anns = self.gdf[b].copy()
        # translate 1-unit down
        bl_border_anns.loc[:, "geometry"] = bl_border_anns.translate(yoff=1.0) 
        bl_border_anns = bl_border_anns.reset_index(drop=True)
        return bl_border_anns

    @property
    def bottom_right_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the bottom edge
        of the tile. (Origin in the top-left corner of the image/tile)
        """
        b = (self.gdf["ymax"] ==  self.ymax) & (self.gdf["xmax"] ==  self.xmax)
        br_border_anns = self.gdf[b].copy()
        # translate 1-unit down and right
        br_border_anns.loc[:, "geometry"] = br_border_anns.translate(yoff=1.0, xoff=1.0) 
        br_border_anns = br_border_anns.reset_index(drop=True)
        return br_border_anns

    @property
    def top_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the top edge
        of the tile. (Origin in the top-left corner of the image/tile)
        """
        t_border_anns = self.gdf[self.gdf["ymin"] ==  self.ymin].copy()
        t_border_anns.loc[:, "geometry"] = t_border_anns
        t_border_anns = t_border_anns.reset_index(drop=True)
        return t_border_anns

    @property
    def top_right_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the top edge
        of the tile. (Origin in the top-left corner of the image/tile)
        """
        b = (self.gdf["ymin"] ==  self.ymin) & (self.gdf["xmax"] == self.xmax)
        tr_border_anns = self.gdf[b].copy()
        tr_border_anns.loc[:, "geometry"] = tr_border_anns.translate(xoff=1.0)
        tr_border_anns = tr_border_anns.reset_index(drop=True)
        return tr_border_anns

    @property
    def top_left_border_annots(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the top edge
        of the tile. (Origin in the top-left corner of the image/tile)
        """
        b = (self.gdf["ymin"] ==  self.ymin) & (self.gdf["xmin"] == self.xmin)
        tl_border_anns = self.gdf[b].copy()
        tl_border_anns.loc[:, "geometry"] = tl_border_anns
        tl_border_anns = tl_border_anns.reset_index(drop=True)
        return tl_border_anns
