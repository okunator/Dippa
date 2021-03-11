import pooch
from pathlib import Path
from typing import Unionn

from .adhoc import handle_pannuke

HASHES = {
    "fold_1":"6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b",
    "fold_2":"5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07",
    "fold_3":"c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d"
}


class PANNUKE(FileHandler):
    def __init__(self, save_dir: Union[str, Path], fold: int, phase: str) -> None:
        """
        Fetches a single fold of the pannuke dataset from https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke.
        Uses pooch package. Saves the data in dir called pannuke under the save_dir that is
        specified.

        Args:
        ----------
            save_dir (str, or Path obj):
                directory where the data is downloaded
            fold (int):
                the pannuke fold number
            phase (int):
                One of ("train", "valid", "test")

        """
        self.save_dir = Path(save_dir)
        assert self.save_dir.exists(), f"save_dir: {self.save_dir} does not exists"

        self.fold = fold
        assert 1 <= self.fold <= 3, f"fold {self.fold}. Only three folds in the data"

        self.phase = phase
        assert self.phase in ("train", "test", "valid")

        # Create pooch and set downloader
        self.downloader = pooch.HTTPDownloader(progressbar=True)
        self.POOCH = pooch.create(
            path=pooch.os_cache(f"{self.save_dir.as_posix()}/pannuke/original/"),
            base_url="https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/",
            version=None,
            version_dev="master",
            retry_if_failed=2,
            registry={
                f"fold_{self.fold}.zip": HASHES[f"fold_{self.fold}"],
            }
        )

    def processor(self, fname: Union[str, Path], action: str, pooch: pooch.Pooch) -> List[Path]:
        """
        Post-processing hook to unzip a file and convert format from .npy to .png/.mat
        The mask .mat files contain "type_map", "inst_map", "inst_centroid", "inst_type",
        and "inst_bbox" keys for every patch similarly to the the CoNSeP-dataset.

        Args:
        ----------
            fname (str):
                Full path of the zipped file in local storage
            action (str):
                One of "download" (file doesn't exist and will download),
                "update" (file is outdated and will download), and
                "fetch" (file exists and is updated so no download).
            pooch (pooch.Pooch)
                The instance of Pooch that called the processor function.

        Returns:
        ----------
            a list of Path objs to the train and test directories
        """
        fname = Path(fname)

        # file does not exist --> download.
        if action in ("update", "download") or not fname.exists():
            self.extract_zips(fname.parent, rm=False)

        # Create folders for the test & train data in the 'pannuke' folder
        # If dirs exists already, skips them
        imgs_test_dir = Path(f"{self.save_dir.as_posix()}/pannuke/test/images")
        anns_test_dir = Path(f"{self.save_dir.as_posix()}/pannuke/test/labels")
        imgs_train_dir = Path(f"{self.save_dir.as_posix()}/pannuke/train/images")
        anns_train_dir = Path(f"{self.save_dir.as_posix()}/pannuke/train/labels")
    
        # Convert .npy files to .mat files and add data to them
        handle_pannuke(
            fname.parent, 
            imgs_train_dir,
            anns_train_dir, 
            imgs_test_dir, 
            anns_test_dir,
            self.fold,
            self.phase
        )

        return [imgs_test_dir, anns_test_dir, imgs_train_dir, anns_train_dir]

    def __call__(self) -> None:
        self.fold_dirs = self.POOCH.fetch(
            f"fold_{self.fold}.zip", 
            processor=self.processor, 
            downloader=self.downloader
        )
