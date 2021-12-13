import torch
from pathlib import Path
from typing import Union, List, Dict, Tuple

from src.settings import DATA_DIR, PATCH_DIR
from src.dl.utils import to_device
from ._base._base_datamodule import BaseDataModule
from ..writers import HDF5Writer
from ..downloaders import PANNUKE


class PannukeDataModule(BaseDataModule):
    def __init__(
            self,
            seg_targets: List[str],
            img_transforms: List[str],
            inst_transforms: List[str],
            normalize: bool=False,
            return_weight_map: bool=False,
            batch_size: int=8,
            num_workers: int=8,
            download_dir: Union[str, Path]=None,
            database_dir: Union[str, Path]=None,
            **kwargs
        ) -> None:
        """
        Pannuke dataset lightning data module:

        Pannuke dataset papers:
        ---------------
        Gamper, J., Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, 
        N. (2019) PanNuke: an open pan-cancer histology dataset for 
        nuclei instance segmentation and classification. In European 
        Congress on Digital Pathology (pp. 11–19).

        Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, 
        S., Azam, A., Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset 
        Extension, Insights and Baselines. arXiv preprint 
        arXiv:2003.10778.

        License: https://creativecommons.org/licenses/by-nc-sa/4.0/

        Args:
        -----------
            seg_targets (List[str]):
                A list of target map names needed for the loss cfunc.
                All the segmentation targets that are not in this list
                are deleted to save memory
            img_transforms (albu.Compose): 
                Albumentations.Compose obj (a list of transformations).
                All the transformations that are applied to the input
                images and corresponding masks
            inst_transforms (ApplyEach):
                ApplyEach obj. (a list of augmentations). All the
                transformations that are applied to only the instance
                labelled masks.
            normalize (bool, default=False):
                If True, channel-wise min-max normalization is applied 
                to input imgs in the dataloading process
            return_weight_map (bool, default=False):
                Include a nuclear border weight map in the dataloading
                process
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading 
                process.
            download_dir (str, or Path obj, default=None):
                directory where the downloaded data is located. If None, 
                and downloading is required, will be downloaded in 
                Dippa/data/pannuke/ folders.
            database_dir (str or Path, default=None):
                The directory where the db is located. If None, and
                writing is required, will be downloaded in 
                Dippa/patches/pannuke/ folders
            
        """
        self.suffix = ".h5"
        self.database_dir = Path(PATCH_DIR  / "hdf5" / "pannuke")
        if database_dir is not None:
            self.database_dir = Path(database_dir)
            
        self.download_dir = Path(DATA_DIR)
        if download_dir is not None:
            self.download_dir = Path(download_dir)
            
        # Create the folders if it does not exist
        self.database_dir.mkdir(exist_ok=True)
        self.download_dir.mkdir(exist_ok=True)
        
        # set up generic db names
        db_train = Path(
            self.database_dir / f"train_pannuke"
        ).with_suffix(self.suffix)
        
        db_test = Path(
            self.database_dir / f"test_pannuke"
        ).with_suffix(self.suffix)
             
        super().__init__(
            db_train,
            db_test,
            seg_targets,
            img_transforms,
            inst_transforms,
            normalize,
            return_weight_map,
            batch_size,
            num_workers
        )

    @property
    def class_dicts(self) -> Dict[str, int]:
        classes = {
            "bg": 0, 
            "neoplastic": 1,
            "inflammatory": 2,
            "connective": 3, 
            "dead": 4,
            "epithelial": 5
        }

        return classes, None # None for convenience (sem classes). 
    
    @staticmethod
    def download(
            download_dir: Union[Path, str], 
            fold: int, 
            phase: str
        ) -> Dict[str, Path]:
        """
        Download pannuke folds from: 
        "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/"
        and convert the mask .npy files to .mat files containing "inst_map", 
        "type_map", "inst_centroid", "inst_type" and "inst_bbox" slots.

        Args:
        -----------
            download_dir (str or Path):
                The directory where the data is downloaded
            fold (int):
                Pannuke contains three data folds. One of (1, 2, 3).
            phase (str):
                Defines the phase that the fold will be used. One of 
                "test", "train". In practice, this arg sets the 
                directory name where the files are downloaded.
        
        Returns:
        -----------
            Dict: 
            
            example:

            {
                "img_train": Path, 
                "img_test":Path, 
                "mask_train":Path, 
                "mask_test":Path
            }
        """
        pannuke = PANNUKE(save_dir=download_dir, fold=fold, phase=phase)
        return pannuke.download()

    @staticmethod
    def write_db(
            img_dir: Union[Path, str],
            mask_dir: Union[Path, str],
            save_dir: Union[Path, str],
            phase: str,
            database_type: str, 
            classes: Dict[str, int],
            augment: bool=True,
            n_copies: int=3
        ) -> None:
        """
        Write (img [.png], mask [.mat]) pairs to either Zarr or HDF5 db

        Args:
        ---------
            img_dir (Path or str):
                Path to the dir containing the image .png files
            mask_dir (Path or str):
                Path to the dir containing the corresponding mask .mat 
                files
            save_dir (Path or str):
                Path to the dir where the DB is written
            phase (str):
                One of ("train", "test")
            database_type (str):
                One of ("zarr", "hdf5")
            classes (Dict[str, int]):
                classes dictionary e.g. {"bg":0, "neopl":1, "infl":2}
            augment (bool, default=True):
                If True, rigid augs are applied to the (img, mask) pairs
            n_copies (int, default=3):
                Number of copies taken for every (img, mask) pairs for 
                augs
        """
        assert database_type in ("zarr", "hdf5")
        writerobj = HDF5Writer # if database_type == "hdf5" else ZarrWriter 

        writer = writerobj(
            img_dir=img_dir,
            mask_dir=mask_dir,
            save_dir=save_dir,
            file_name=f"{phase}_pannuke",
            classes=classes,
            patch_shape=None,
            stride_size=None,
            rigid_augs_and_crop=augment,
            n_copies=n_copies,
        )

        writer.write2db()        
        
    @property
    def class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the proportion of pixels of diffenrent classes in the
        train dataset
        """
        weights = self.get_class_weights(self.db_fname_train.as_posix())
        weights_bin = self.get_class_weights(
            self.db_fname_train.as_posix(), binary=True
        )
        return to_device(weights), to_device(weights_bin)


    def prepare_data(self) -> None:
        """
        1. Download pannuke folds from: 
            "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/"
        2. Write images/masks to to zarr or hdf5 files

        Folds are saved to save_dir, unpacked, converted to singular
        .png/.mat files (CoNSeP-format) that are then moved to train
        and test folders. After conversion, the new filenames contain
        the fold and tissue type of the patch.
        """
        # Download folds and re-format. If data is downloaded and processed
        # already the downloading and processing is skipped
        fold1 = self.download(self.download_dir, fold=1, phase="train")
        fold2 = self.download(self.download_dir, fold=2, phase="train")
        fold3 = self.download(self.download_dir, fold=3, phase="test") 

        # Write dbs if they dont yet exist
        if not self.db_fname_train.exists():
            self.write_db(
                img_dir=fold1["img_train"],
                mask_dir=fold1["mask_train"],
                save_dir=self.database_dir,
                phase="train",
                database_type="hdf5",
                classes=self.class_dicts()[0],
                augment=True,
                n_copies=3
            )

        if not self.db_fname_test.exists():
            self.write_db(
                img_dir=fold1["img_test"],
                mask_dir=fold1["mask_test"],
                save_dir=self.database_dir,
                phase="test",
                database_type="hdf5",
                classes=self.class_dicts()[0],
                augment=False,
                n_copies=None,
            )
