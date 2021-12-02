import re
import torch
from torch.utils.data import Dataset
from typing import Optional, Union, Tuple, List
from pathlib import Path

from src.utils import FileHandler


SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")


__all__ = ["FolderDataset"]


class FolderDataset(Dataset, FileHandler):
    def __init__(
        self, 
        folder_path: Union[str, Path],
        pattern: Optional[str]="*", 
        sort_by_y: Optional[bool]=False,
        xmax: Optional[int]=None,
        ymax: Optional[int]=None,
        auto_range: bool=False,
        tile_size: Optional[Tuple[int, int]]=(1000, 1000)
    ) -> None:
        """
        Simple pytorch folder dataset. Assumes that
        folder_path contains only image files which are readable
        by cv2.

        Args:
        ----------
            folder_path (Union[str, Path]):
                path to the folder containig tile/image files
            pattern (str, optional, default="*"):
                file pattern for filtering only the files that contain 
                the pattern.
            sort_by_y (bool, optional, default=False):
                sorts a folder (containing tiles extracted by histoprep 
                package) by the y-coord rather than the x-coord
            xmax (int, optional, default=None):
                filters all the tile-files that contain x-coord less 
                or equal to this param in their filename. Works with 
                tiles extracted with histoprep. 
                See https://github.com/jopo666/HistoPrep 
            ymax (int, optional, default=None):
                filters all the tile-files that contain y-coord less 
                or equal to this param in their filename. Works with 
                tiles extracted with histoprep. 
                See https://github.com/jopo666/HistoPrep 
            auto_range (bool, default=False):
                Automatically filter tiles that contain ONE tissue 
                section rather than every redundant tissue section in 
                the wsi.
            tile_size (Tuple[int, int], optional, default=(1000, 1000)):
                size of the input tiles in the folder. Optional.
        """
        super().__init__()
        self.tile_size = tile_size
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"folder: {folder_path} does not exist")
        
        if not folder_path.is_dir():
            raise ValueError(f"path: {folder_path} is not a folder")
        
        if not all([f.suffix in SUFFIXES for f in folder_path.iterdir()]):
            raise ValueError(
                f"files formats in given folder need to be in {SUFFIXES}"
            )

        #  sort files
        if sort_by_y:
            self.fnames = sorted(
                folder_path.glob(pattern), 
                key=lambda x: self._get_xy_coords(x.name)[1]
            )
        else:
            self.fnames = sorted(folder_path.glob(pattern))

        # filter by xy-cooridnates encoded in the filename
        if xmax is not None:
            self.fnames = [
                f for f in self.fnames 
                if self._get_xy_coords(f.name)[0] <= xmax
            ]
        if ymax is not None and not auto_range:
            self.fnames = [
                f for f in self.fnames 
                if self._get_xy_coords(f.name)[1] <= ymax
            ]
        
        if auto_range:
            ymin, ymax = self._get_auto_range(coord="y") # only y-axis for now
            self.fnames = [
                f for f in self.fnames 
                if ymin <= self._get_xy_coords(f.name)[1] <= ymax
            ]

    def _get_xy_coords(self, fname: str) -> List[int]:
        """
        Extract xy-coords from files named with x- and y- coordinates 
        in their file name.
        
        example filename: "sumthing_4955_x-47000_y-25000.png 
        """
        if not re.findall(r"(x-\d+_y-\d+)", fname):
            raise ValueError("""
                fname not in 'sumthing_x-[coord1]_y-[coord2]'-format
                Set auto_range to False if filenames are not in this format
                """
            )
        
        xy_str = re.findall(r"(x-\d+_y-\d+)", fname)
        xy = [int(c) for c in re.findall(r"\d+", xy_str[0])]

        return xy

    def _get_auto_range(
            self, 
            coord: str="y", 
            section_ix: int=0, 
            section_length: int=6000
        ) -> Tuple[int, int]:
        """
        Automatically extract a range of tiles that contain a section
        of tissue in a whole slide image. This is pretty ad hoc
        and requires histoprep extracted tiles and that the slides 
        contain many tissue sections. Use with care.

        Args:
        ---------
            coord (str, default="y"):
                specify the range in either x- or y direction
            section_ix (int, default=0):
                the nth tissue section in the wsi in the direction of 
                the `coord` param. Starts from 0th index. E.g. If
                `coord='y'` the 0th index is the upmost tissue section.
            section_length (int, default=6000):
                Threshold to concentrate only on tissue sections that
                are larger than 6000 pixels

        Returns:
        --------
            Tuple[int, int]: The start and end point of the tissue 
            section in the specified direction
        """
        ix = 1 if coord == "y" else 0
        coords = sorted(
            set([self._get_xy_coords(f.name)[ix] for f in self.fnames])
        )

        try:
            splits = []
            split = []
            for i in range(len(coords)-1):
                if coords[i + 1] - coords[i] == self.tile_size[ix]:
                    split.append(coords[i])
                else:
                    if i < len(coords) - 1:
                        split.append(coords[i]) 
                    splits.append(split)
                    split = []
            
            ret_splits = [
                split for split in splits 
                if len(split) >= section_length//self.tile_size[ix]
            ]
            ret_split = ret_splits[section_ix]
            return ret_split[0], ret_split[-1]
        except:
            # if there is only one tissue section, return min and max
            print(coords)
            start = min(coords, key=lambda x: x[ix])[ix]
            end = max(coords, key=lambda x: x[ix])[ix]
            return start, end

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> torch.Tensor:
        fn = self.fnames[index]
        im = FileHandler.read_img(fn.as_posix())
        im = torch.from_numpy(im.transpose(2, 0, 1))

        return {
            "im":im, 
            "file":fn.name[:-4]
        }
