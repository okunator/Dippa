import geopandas as gpd
import shapely
import geojson

from typing import Union, Tuple
from tqdm import tqdm
from pathlib import Path
from shapely.strtree import STRtree

from .tile import GSONTile
from .base_merger import BaseGSONMerger


class AreaMerger(BaseGSONMerger):
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
        Helper class to merge area segmentation geojson tile-files
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

        super(AreaMerger, self).__init__(
            in_dir, xmax, ymax, xmin, ymin, tile_size
        )
  
        self.tiles = [
            GSONTile(f, *self._get_xy_coords(f.name)) for f in self.files
        ]

    @property
    def fullgdf(self) -> gpd.GeoDataFrame:

        cols = None
        rows = []
        for tile in self.tiles:
            if tile.gdf is not None:
                cols = tile.gdf.columns
                for _, row in tile.gdf.iterrows():
                    rows.append(row)

        gdf = gpd.GeoDataFrame(rows, columns=cols)
        return gdf.reset_index() 

    def merge(self, fname: Union[str, Path]) -> None:
        """
        Merge the area/semantic predictions if they overlap and have
        the same class and write them in one .json file.

        Args:
        ---------
            fname (str, default=None):
                name/filepath of the geojson file that is written
        """

        # get all the polygons in all of the geojson files
        gdfd = self.fullgdf.dissolve("class_name", as_index=False, sort=False)

        merged_polys = []
        pbar = tqdm(gdfd.iterrows())
        for _, row in pbar:

            cname = row.class_name
            geo = row.geometry

            pbar.set_description(f"Processing {cname}-annots")

            # merge the given polygons if they intersect and have same class    
            if isinstance(geo, shapely.geometry.MultiPolygon):
                
                new_coords = []
                tree = STRtree(
                    [poly.buffer(1.0) for poly in geo]
                )

                for poly in geo:
                    poly = poly.buffer(1.0)
                    inter = [p for p in tree.query(poly) if p.intersects(poly)]
                    merged = shapely.ops.unary_union(inter)
                    new_coords.append(merged)

                coords = shapely.ops.unary_union(new_coords)
                coords = coords.buffer(10.0).buffer(-10.0).simplify(0.3)
                merged_polys.append((cname, coords))
            else:
                merged_polys.append(
                    (cname, geo.buffer(10.0).buffer(-10.0).simplify(0.3))
                )

        # write to geojson
        annotations = []
        for c, polygons in merged_polys:
            if isinstance(polygons, shapely.geometry.MultiPolygon):
                
                for poly in polygons:
                    geo_obj = self._gsonobj
                    geo_obj["properties"]["classification"]["name"] = c
                    geo_obj["geometry"]["coordinates"] = [
                        [list(tup) for tup in poly.exterior.coords[:]]
                    ]
                    annotations.append(geo_obj)
            else:
                geo_obj = self._gsonobj
                geo_obj["properties"]["classification"]["name"] = c
                geo_obj["geometry"]["coordinates"] = [
                    [list(tup) for tup in polygons.exterior.coords[:]]
                ]
                annotations.append(geo_obj)

        # write to file
        fname = Path(fname).with_suffix(".json")
        with fname.open('w') as out:
            geojson.dump(annotations, out)