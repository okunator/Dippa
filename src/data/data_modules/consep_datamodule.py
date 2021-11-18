import torch
from pathlib import Path
from typing import Union, List, Dict, Tuple

from src.settings import DATA_DIR, PATCH_DIR
from src.dl.utils import to_device
from ._base._base_datamodule import BaseDataModule
from ..writers import HDF5Writer
from ..downloaders import CONSEP


class ConsepDataModule(BaseDataModule):
    def __init__(
            self, 
            target_types: List[str],
            database_type: str="hdf5",
            dataset_type: str="hover",
            augs: List[str]=["hue_sat", "non_rigid", "blur"],
            normalize: bool=False,
            return_weight_map: bool=False,
            rm_touching_nuc_borders: bool=False,
            batch_size: int=8,
            num_workers: int=8,
            download_dir: Union[str, Path]=None, 
            database_dir: Union[str, Path]=None,
            convert_classes: bool=True,
            **kwargs
        ) -> None:
        """
        CoNSeP dataset lightning datamodule

        CoNSeP dataset paper:
        --------------
        S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, 
        J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation
        and Classification of Nuclei in Multi-Tissue Histology Images." 
        Medical Image Analysis, Sept. 2019. 

        Args:
        -----------
            target_types (List[str]):
                A list of the targets that are loaded during dataloading
                process. Allowed values: "inst", "type".
            database_type (str, default="hdf5"):
                One of ("zarr", "hdf5"). The files are written in either
                zarr or hdf5 files that is used by the torch dataloader 
                during training.
            dataset_type (str, default="hover"):
                The dataset type. One of: "hover", "dist", "contour",
                "basic", "unet"
            dataset_type (str, default="hover"):
                The dataset type. One of: "hover", "dist", "contour",
                "basic", "unet"
            augs (List, default=["hue_sat","non_rigid","blur"])
                List of augs. Allowed augs: "hue_sat", "rigid",
                "non_rigid", "blur", "non_spatial", "normalize"
            normalize (bool, default=False):
                If True, channel-wise min-max normalization is applied 
                to input imgs in the dataloading process
            return_weight_map (bool, default=False):
                Include a nuclear border weight map in the dataloading
                process
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading 
                process.
            download_dir (str, or Path obj, default=None):
                directory where the downloaded data is located. If None, 
                and downloading is required, will be downloaded in 
                Dippa/data/consep/ folders.
            database_dir (str or Path, default=None):
                The directory where the db is located. If None, and
                writing is required, will be downloaded in 
                Dippa/patches/consep/ folders
            convert_classes (bool, default=True):
                Convert the original classes to the reduced set of 
                classes. More info in their github and their paper.
            
        """
        self.database_type = database_type
        self.suffix = ".h5" if self.database_type == "hdf5" else ".zarr"

        self.database_dir = Path(PATCH_DIR  / f"{database_type}" / "consep")
        if database_dir is not None:
            self.database_dir = Path(database_dir)

        self.download_dir = Path(DATA_DIR)
        if download_dir is not None:
            self.download_dir = Path(download_dir)
        
        # Create the folders if it does not exist
        self.database_dir.mkdir(exist_ok=True)
        self.download_dir.mkdir(exist_ok=True)
        self.convert_classes = convert_classes
        
        # set up generic db names
        db_train = Path(
            self.database_dir / f"train_consep"
        ).with_suffix(self.suffix)

        db_test = Path(
            self.database_dir / f"test_consep"
        ).with_suffix(self.suffix)
        
        allowed_targets = ("type", "inst")
        if not all(t in allowed_targets for t in target_types):
            raise ValueError(f"""
                Allowed targets for CoNSeP dataset: {allowed_targets}. Got:
                {target_types}."""
            )
                    
        super().__init__(
            db_train,
            db_test,
            target_types,
            dataset_type,
            augs,
            normalize,
            return_weight_map,
            rm_touching_nuc_borders,
            batch_size,
            num_workers
        )

    @property
    def class_dicts(self) -> Tuple[Dict[str, int], None]:
        classes = {
            "bg":0, 
            "miscellanous":1, 
            "inflammatory":2, 
            "epithelial":3, 
            "spindle":4
        }

        return classes, None # None for convenience (sem classes). 

    @staticmethod
    def get_orig_classes() -> Dict[str, int]:
        return {
            "bg":0, 
            "miscellanous": 1, 
            "inflammatory": 2, 
            "healty_epithelial": 3,
            "malignant_epithelial": 4, 
            "fibroblast": 5,
            "muscle": 6, 
            "endothelial": 7
        }

    @staticmethod
    def download(
            download_dir: Union[Path, str], 
            convert_classes: bool=True
        ) -> Dict[str, Path]:
        """
        Download CoNSeP dataset from: 
        https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/.
        and optionally creduce the classes from 8 to 5. See their README

        Args:
        -----------
            download_dir (str or Path):
                The directory where the data is downloaded
            convert_classes (bool, default=True):
                Convert the original classes to the reduced set of 
                classes
        
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
        consep = CONSEP(save_dir=download_dir, convert_classes=convert_classes)
        return consep.download()

    @staticmethod
    def write_db(
            img_dir: Union[Path, str],
            mask_dir: Union[Path, str],
            save_dir: Union[Path, str],
            phase: str,
            database_type: str, 
            classes: Dict[str, int],
            augment: bool=True,
            stride_size: int=80,
            patch_shape: Tuple[int, int]=(512, 512),
            crop_shape: Tuple[int, int]=(256,256)
        ) -> None:
        """
        Write overlapping (img [.png], mask [.mat]) pairs to either 
        Zarr or HDF5 database

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
            patch_shape (Tuple[int, int], default=(512, 512)):
                Specifies the height and width of the patches. If this 
                is None, no patching is applied.
            stride_size (int, default=80):
                Stride size for the sliding window patcher. Needs to be 
                less or equal to patch_shape. If less than patch_shape, 
                patches are created with overlap. This arg is ignored if
                patch_shape is None.
            crop_shape (Tuple[int, int], default=(256, 256)):
                If augment is True, this is the crop shape for the 
                center crop.
        """
        assert database_type in ("zarr", "hdf5")
        writerobj = HDF5Writer # if database_type == "hdf5" else ZarrWriter 

        writer = writerobj(
            img_dir=img_dir,
            mask_dir=mask_dir,
            save_dir=save_dir,
            file_name=f"{phase}_consep",
            classes=classes,
            rigid_augs_and_crop=augment,
            patch_shape=patch_shape,
            stride_size=stride_size,
            crop_shape=crop_shape
        )

        writer.write2db()

    @property
    def class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the proportion of pixels of diffenrent classes in the train dataset
        """
        weights = self.get_class_weights(self.db_fname_train.as_posix())
        weights_bin = self.get_class_weights(
            self.db_fname_train.as_posix(), binary=True
        )
        return to_device(weights), to_device(weights_bin)

    # DataModule boilerplate
    def prepare_data(self) -> None:
        """
        1. Download CoNSeP dataset from: 
            https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
        2. Write images/masks to to zarr or hdf5 files

        Folds are saved to save_dir, unpacked, converted to singular
        .png/.mat files (CoNSeP-format) that are then moved to train
        and test folders. After conversion, the new filenames contain
        the fold and tissue type of the patch.
        """
        # Download folds and re-format. If data is downloaded and processed
        # already the downloading and processing is skipped
        fold1 = self.download(self.download_dir, convert_classes=self.convert_classes)

        # Write dbs if they dont yet exist
        if not self.db_fname_train.exists():
            self.write_db(
                img_dir=fold1["img_train"],
                mask_dir=fold1["mask_train"],
                save_dir=self.database_dir,
                phase="train",
                database_type=self.database_type,
                classes=self.get_classes(),
                augment=True,
            )

        if not self.db_fname_test.exists():
            self.write_db(
                img_dir=fold1["img_test"],
                mask_dir=fold1["mask_test"],
                save_dir=self.database_dir,
                phase="test",
                database_type=self.database_type,
                classes=self.get_classes(),
                augment=False,
                patch_shape=(256, 256),
                stride_size=256,
            )
