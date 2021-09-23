import pooch
from pathlib import Path
from typing import Union, Dict, Callable

from src.utils import FileHandler


class BaseDownloader(FileHandler):
    def __init__(
            self, 
            save_dir: Union[str, Path], 
            dataset_name: str,
            handler_func: Callable,
            **kwargs
        ) -> None:
        """
        Base downloader class.

        Args:
        ---------
            save_dir (Path or str):
                Directory where the db is saved
            dataset_name (str):
                A name for the db dataset
            handler_func (Callable):
                A function that executes the processing & converting & 
                moving of the downloaded data files. 
        """
        self.save_dir = Path(save_dir)
        self.dataset_name = dataset_name
        self.handler_func = handler_func
        self.kwargs = kwargs

    def processor(
            self, 
            fname: Union[str, Path], 
            action: str, 
            pooch: pooch.Pooch
        ) -> Dict[str, Path]:
        """
        Post-processing hook to unzip a file and convert file formats 
        after downloading 

        Args:
        ----------
            fname (str):
                Full path of the zipped file in local storage
            action (str):
                One of "download" (file doesn't exist and will download)
                "update" (file is outdated and will download), and
                "fetch" (file exists and is updated so no download).
            pooch (pooch.Pooch)
                The instance of Pooch that called the processor function

        Returns:
        ----------
            a list of Path objs to the train and test directories
        """
        fname = Path(fname)

        # Create folders for the test & train data in the 'pannuke' folder
        # If dirs exists already, skips them
        imgs_test_dir = Path(
            f"{self.save_dir.as_posix()}/{self.dataset_name}/test/images"
        )
        anns_test_dir = Path(
            f"{self.save_dir.as_posix()}/{self.dataset_name}/test/labels"
        )
        imgs_train_dir = Path(
            f"{self.save_dir.as_posix()}/{self.dataset_name}/train/images"
        )
        anns_train_dir = Path(
            f"{self.save_dir.as_posix()}/{self.dataset_name}/train/labels"
        )

        # Don't do anything train & test dir are already populated
        if imgs_test_dir.exists() and imgs_train_dir.exists():
            is_populated = [False]*4
            if any(imgs_test_dir.iterdir()):
                is_populated[0] = True
            if any(imgs_train_dir.iterdir()):
                is_populated[1] = True
            if any(anns_test_dir.iterdir()):
                is_populated[2] = True
            if any(anns_train_dir.iterdir()):
                is_populated[3] = True

            if all(is_populated):
                return {
                    "img_test": imgs_test_dir, 
                    "mask_test": anns_test_dir, 
                    "img_train": imgs_train_dir, 
                    "mask_train": anns_train_dir
                }


        # file does not exist --> download.
        if action in ("update", "download") or not fname.exists():
            self.extract_zips(fname.parent, rm=False)
        
            # Process
            self.handler_func(
                fname.parent, 
                imgs_train_dir,
                anns_train_dir, 
                imgs_test_dir, 
                anns_test_dir,
                **self.kwargs
            )

        return {
            "img_test": imgs_test_dir, 
            "mask_test": anns_test_dir, 
            "img_train": imgs_train_dir, 
            "mask_train": anns_train_dir
        }
