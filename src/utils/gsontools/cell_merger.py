import shapely
import geojson
import warnings

from shapely.strtree import STRtree
from typing import Union, List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

from .tile import GSONTile
from .base_merger import BaseGSONMerger


class CellMerger(BaseGSONMerger):
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
        Helper class to merge cell segmentation geojson tile-files
        File names need to contain x- and y- coordinates.

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
        super(CellMerger, self).__init__(
            in_dir, xmax, ymax, xmin, ymin, tile_size
        )

        # init helper lookups
        self.neighbor_relations = {
            "right": {"main":"right", "adj":"left"},
            "left": {"main":"left", "adj":"right"},
            "top": {"main":"top", "adj":"bottom"},
            "bottom": {"main":"bottom", "adj":"top"},
        }

        self.doned = {
            f.name: {
                "left": None, "right": None, "top": None, "bottom": None, 
                "non_border": None, "top_right": None, "top_left": None, 
                "bottom_left": None, "bottom_right" : None
            } for f in self.files
        }

    def _handle_multipoly(
            self,
            multipoly: shapely.geometry.MultiPolygon
        ) -> List[shapely.geometry.Polygon]:
        """
        Convert a shapely multipolygon into a list of polygons.

        If polygons intersect, unite them.

        Args:
        ---------
            multipoly (shapely.geometry.MultiPolygon):
                Shapely multipolygon object
        Returns:
        ---------
            List: A list of shapely polygons
        """

        ret = []

        try:
            polys = list(multipoly)
            tree = STRtree(polys)

            for poly in polys:
                poly = poly.buffer(1.0)
                inter = [p for p in tree.query(poly) if p.buffer(1.0).intersects(poly)]
                merged = shapely.ops.unary_union(inter)
                ret.append(merged)

        except:
            warnings.warn(
                "Failed to explode and merge a multipolygon. Continue",
                RuntimeWarning
            )

        return ret

    def _get_coords(
            self,
            poly: Union[List[shapely.geometry.Polygon], shapely.geometry.Polygon]
        ) -> List[List[Tuple[int, int]]]:
        """
        Set the shapely coordinates to correct geojson format
        """
        ret = []

        if isinstance(poly, list):
            for p in poly:
                ret.extend([list(tup) for tup in p.exterior.coords[:]])
        else:
            ret.extend([list(tup) for tup in poly.exterior.coords[:]])

        return ret

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
            border_annots_adj = gson_adj.top_border_annots
        elif adj_pos == "top":
            border_annots_main = gson.top_border_annots
            border_annots_adj = gson_adj.bottom_border_annots
        
        
        # combine polygons that intersect/touch between two image tiles
        # (cells that are split in two between two image tiles)
        new_polys = []
        if not border_annots_main.empty and not border_annots_adj.empty:
            for main_poly, c in zip(
                border_annots_main.geometry, 
                border_annots_main.class_name
            ):
                for adj_poly in border_annots_adj.geometry:  
                    
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
                        geo_obj["properties"]["classification"]["name"] = c

                        if new_poly.geom_type == 'MultiPolygon':
                            new_poly = self._handle_multipoly(new_poly)

                        coords = self._get_coords(new_poly)
                        geo_obj["geometry"]["coordinates"] = [coords]

                        new_polys.append(geo_obj)

        return new_polys

    def _get_non_border_polygons(self, gson: GSONTile) -> List[Dict]:
        """
        Get all the polygons of a tile that are not adjascent to 
        polygons in other tiles and make them valid.

        Args:
        ----------
            gson (GSONTile):
                geojson tile

        Returns:
        ----------
            List: A list of geojson formatted dictionaries of the 
                  combined cell annotations
        """
        nb_annots = gson.non_border_annots

        new_polys = []
        if not nb_annots.empty:
            for poly, c in zip(nb_annots.geometry, nb_annots.class_name):

                # workaround to handle invalid polygons
                if not poly.is_valid:
                    poly = poly.buffer(0)

                # do some simplifying
                poly = poly.buffer(
                    10.0, join_style=1
                ).buffer(-10.0, join_style=1).simplify(0.3)

                # init new geojson obj and set coords & cell type and append
                geo_obj = self._gsonobj
                geo_obj["properties"]["classification"]["name"] = c

                # Union for multipolygon polygons
                if poly.geom_type == 'MultiPolygon':
                    poly = self._handle_multipoly(poly)

                coords = self._get_coords(poly)
                geo_obj["geometry"]["coordinates"] = [coords]

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
                for pos, f_adj in adj.items():
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
