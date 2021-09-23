import re
import shapely
import geojson
import geopandas as gpd
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Dict, Tuple


class GSONTile:
    def __init__(
            self, 
            fname: Union[Path, str], 
            xmin: int, 
            ymin: int, 
            tile_size: int=(1000, 1000)
        ) -> None:
        """
        Class abstaraction for one tile of geojson annotations.

        Args:
        ----------
            fname: (Path or str):
                name of the gson file
            xmin (int):
                the min-x coordinate of the tile
            ymin (int):
                the min-y coordinate of the tile
            tile_size (Tuple[int, int], default=(1000, 1000)):
                size of the input tile
        """
        self.fname = Path(fname)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmin + tile_size[0]
        self.ymax = ymin + tile_size[1]
        self.gdf = self._gson2gdf

    @property
    def _gson2gdf(self) -> gpd.GeoDataFrame:
        """
        Read a geojson/json file and convert it to geopandas df
        Adds bounding box-coords for the polygons
        """
        try:
            df = pd.read_json(self.fname)
            df["geometry"] = df["geometry"].apply(shapely.geometry.shape)
            gdf = gpd.GeoDataFrame(df).set_geometry('geometry')

            # add bounding box coords of the polygons to the gdfs
            # and correct for the max coords
            gdf["xmin"] = gdf.bounds["minx"].astype(int)
            gdf["xmax"] = gdf.bounds["maxx"].astype(int) + 1
            gdf["ymin"] = gdf.bounds["miny"].astype(int)
            gdf["ymax"] = gdf.bounds["maxy"].astype(int) + 1

            return gdf
        except:
            pass

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
    def t_border_anns(self) -> gpd.GeoDataFrame:
        """
        Get all the annotations/polygons that touch the top edge
        of the tile. (Origin in the top-left corner of the image/tile)
        """
        t_border_anns = self.gdf[self.gdf["ymin"] ==  self.ymin].copy()
        t_border_anns.loc[:, "geometry"] = t_border_anns
        t_border_anns = t_border_anns.reset_index(drop=True)
        return t_border_anns


class GSONMerger:
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
        Helper class to merge geojson tile-files that are named with 
        histoprep convetion:
        
        Examples:
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
        
        # init helper lookups
        self.neighbor_relations = {
            "right": {"main":"right", "adj":"left"}, 
            "left": {"main":"left", "adj":"right"}, 
            "top": {"main":"top", "adj":"bottom"}, 
            "bottom": {"main":"bottom", "adj":"top"}, 
        }
        self.doned = {
            f.name: {
                "left": None, "right":None, "top": None, 
                "bottom": None, "non_border": None
            } for f in self.files
        }

    
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
        assert re.findall(r"(x-\d+_y-\d+)", fname), (
            "fname not in 'x-[coord1]_y-[coord2]'-format"
        )
        xy_str = re.findall(r"(x-\d+_y-\d+)", fname)
        x, y = (int(c) for c in re.findall(r"\d+", xy_str[0]))
        return x, y

    def _get_file_from_coords(self, x: int, y: int) -> str:
        """
        fname needs to contain x & y-coordinates in 
        "x-[coord1]_y-[coord2]"-format
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
    
    def _get_top_neighbor(self, fname: str) -> Path :
        x, y = self._get_xy_coords(fname)
        y -= self.tile_size[1]
        return self._get_file_from_coords(x, y)

    def _get_top_right_neighbor(self, fname: str) -> Path :
        x, y = self._get_xy_coords(fname)
        x += self.tile_size[0]
        y -= self.tile_size[1]
        return self._get_file_from_coords(x, y)

    def _get_top_left_neighbor(self, fname: str) -> Path :
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

        return adj

    def _get_non_border_polygons(self, gson: GSONTile) -> List[Dict]:
        """
        Get all the polygons of a tile that are not adjascent to 
        polygons in other tiles

        Args:
        ----------
            gson (GSONTile):
                geojson tile

        Returns:
        ----------
            List: A list of geojson formatted dictionaries of the 
            combined cell annotations
        """
        non_border_annots = gson.non_border_annots

        new_polys = []
        if not non_border_annots.empty:
            for poly, c in zip(
                non_border_annots["geometry"], 
                non_border_annots["properties"]
            ):  
                # workaround to handle invalid polygons
                if not poly.is_valid:
                    poly = poly.buffer(0)

                # do some simplifying
                poly = poly.buffer(
                    10.0, join_style=1
                ).buffer(-10.0, join_style=1).simplify(0.3)

                # init new geojson obj and set coords & cell type and append
                geo_obj = self._gsonobj
                ctype = c["classification"]["name"]
                geo_obj["properties"]["classification"]["name"] = ctype

                # Union for multipolygon polygons
                if poly.geom_type == 'MultiPolygon':
                    try:
                        poly = shapely.ops.cascaded_union([
                            shapely.geometry.Polygon(
                                c.exterior
                            ).buffer(0.01).buffer(-0.01) for p in poly
                        ])
                    except:
                        continue

                geo_obj["geometry"]["coordinates"] = [
                    [list(tup) for tup in poly.exterior.coords[:]]
                ]

                new_polys.append(geo_obj)
        
        return new_polys

    def _merge_adj_ploygons(
            self, 
            gson: GSONTile, 
            gson_adj: GSONTile, 
            adj_pos: str
        ) -> List[Dict]:
        """
        Merge adjascent geojsons. Combines the cells that are split at 
        the image borders


        Args:
        ----------
            gson (GSONTile):
                geojson of the main tile 
            gson_adj (GSONTile):
                geojson of the adjascent tile 
            adj_pos (str):
                the postition of the adjascent tile relative to the main
                tile. One of "left", "right", "bottom", "bottomleft", 
                "bottomright", "top", "topleft", "topright"

        Returns:
        ----------
            List: A list of geojson formatted dictionaries of the 
            combined cell annotations

        """
        # Get the polygons that end/start at the image border
        if adj_pos == "right":
            border_annots_main = gson.right_border_annots
            border_annots_adj = gson_adj.left_border_annots
        elif adj_pos == "left":
            border_annots_main = gson.left_border_annots
            border_annots_adj = gson_adj.right_border_annots
        elif adj_pos == "bottom":
            border_annots_main = gson.bottom_border_annots
            border_annots_adj = gson_adj.t_border_anns
        elif adj_pos == "top":
            border_annots_main = gson.t_border_anns
            border_annots_adj = gson_adj.bottom_border_annots
        
        # combine polygons that intersect/touch between two image tiles
        # (cells that are split in two between two image tiles)
        new_polys = []
        if not border_annots_main.empty and not border_annots_adj.empty:
            for main_poly, c in zip(
                border_annots_main["geometry"], 
                border_annots_main["properties"]
            ):
                for adj_poly in border_annots_adj["geometry"]:  
                    
                    # workaround to handle invalid polygons
                    if not main_poly.is_valid:
                        main_poly = main_poly.buffer(0)
                    if not adj_poly.is_valid:
                        adj_poly = adj_poly.buffer(0)

                    # combine the polygons if they intersect
                    if main_poly.intersects(adj_poly):
                        new_poly = shapely.ops.unary_union(
                            [main_poly, adj_poly]
                        )
                        # do some simplifying
                        new_poly = new_poly.buffer(
                            10.0, join_style=1
                        ).buffer(-10.0, join_style=1).simplify(0.3)

                        # init gson obj, set coords & type and append
                        geo_obj = self._gsonobj
                        ctype = c["classification"]["name"]
                        geo_obj["properties"]["classification"]["name"] = ctype

                        if new_poly.geom_type == 'MultiPolygon':
                            try:
                                new_poly = shapely.ops.cascaded_union(
                                    [
                                        shapely.geometry.Polygon(
                                            c.exterior
                                        ).buffer(0.01).buffer(-0.01) 
                                        for p in new_poly
                                    ]
                                )
                            except:
                                continue

                        geo_obj["geometry"]["coordinates"] = [
                            [list(tup) for tup in new_poly.exterior.coords[:]]
                        ]

                        new_polys.append(geo_obj)

        return new_polys

    def merge(self, fname: str) -> None:
        """
        Merge all geojson files to one file and handle the split cells
        at the image borders. (Not handling corners... yet)

        Args:
        ----------
            fname (str, default=None):
                name/filepath of the geojson file that is written
        """
        annotations = []
        pbar = tqdm(self.files)
        for f in pbar:
            pbar.set_description(f"Processing file: {f.name}")

            # get adjascent tiles
            adj = self._get_adjascent_tiles(f.name)
            
            # Init GSONTile obj
            x1, y1 = self._get_xy_coords(f.name)
            gson = GSONTile(f, x1, y1)

            if gson.gdf is not None:
                # add the non border polygons
                if self.doned[f.name]["non_border"] is None:
                    non_border_polygons = self._get_non_border_polygons(gson)
                    annotations.extend(non_border_polygons)
                    self.doned[f.name]["non_border"] = f.name

                # loop the adjascent tiles and add the border polygons
                for i, (pos, f_adj) in enumerate(adj.items()):
                    if f_adj is not None:
                        if self.doned[f.name][pos] is None:
                            x2, y2 = self._get_xy_coords(f_adj.name)
                            gson_adj = GSONTile(f_adj, x2, y2)
                            
                            if gson_adj.gdf is not None:
                                border_polygons = self._merge_adj_ploygons(
                                    gson, gson_adj, pos
                                )
                                annotations.extend(border_polygons)
                                
                                # update lookup
                                main_pos = self.neighbor_relations[pos]["main"]
                                adj_pos = self.neighbor_relations[pos]["adj"]
                                self.doned[f_adj.name][adj_pos] = f.name
                                self.doned[f.name][main_pos] = f_adj.name

        # write to file
        fname = Path(fname).with_suffix(".json")
        with fname.open('w') as out:
            geojson.dump(annotations, out)

