import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
from torch.utils.data import DataLoader

from src.settings import DATA_DIR, PATCH_DIR
from src.utils import FileHandler
from src.dl.utils import to_device
from ..writers import HDF5Writer, ZarrWriter
from ..downloaders import CONSEP
from ..datasets.dataset_builder import DatasetBuilder



class ConsepDataModule(pl.LightningDataModule, FileHandler):
    def __init__(self, 
                 database_type: str,
                 augmentations: List[str]=["hue_sat", "non_rigid", "blur"],
                 normalize: bool=False,
                 aux_branch: str="hover",
                 batch_size: int=8,
                 num_workers: int=8,
                 download_dir: Union[str, Path]=None, 
                 database_dir: Union[str, Path]=None,
                 convert_classes: bool=True) -> None:
        """
        CoNSeP dataset lightning datamodule

        CoNSeP dataset paper:
        --------------
        S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and
        N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of
        Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, 
        Sept. 2019. 

        Args:
        -----------
            database_type (str):
                One of ("zarr", "hdf5"). The files are written in either
                zarr or hdf5 files that is used by the torch dataloader 
                during training.
            augmentations (List[str], default=["hue_sat", "non_rigid", "blur"])
                List of augmentations. e.g. ["hue_sat", "non_rigid", "blur"]...
                allowed augs: ("hue_sat", "rigid", "non_rigid", "blur", 
                "non_spatial", "normalize")
            normalize (bool, default=False):
                If True, channel-wise min-max normalization is applied to input
                imgs in the dataloading process
            aux_branch (str, default="hover"):
                Signals that the dataset needs to prepare an input for an 
                auxiliary branch in the __getitem__ method. One of ("hover", 
                "dist", "contour", None). If None, assumes that the network 
                does not contain auxiliary branch and the unet style dataset 
                (edge weights and no overlapping cells) is used as the dataset. 
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading process.
            download_dir (str, or Path obj, default=None):
                directory where the downloaded data is located or saved to. If
                None, and downloading is required, will be downloaded in 
                Dippa/data/pannuke/ folders.
            database_dir (str or Path, default=None):
                The directory where the db is located or saved to. If None, and
                writing is required, will be downloaded in 
                Dippa/patches/pannuke/ folders
            convert_classes (bool, default=True):
                Convert the original classes to the reduced set of classes
                See the paper.
            
        """
        super(ConsepDataModule, self).__init__()

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
        
        # Variables for torch DataLoader
        self.database_type = database_type
        self.augs = augmentations
        self.norm = normalize
        self.aux_branch = aux_branch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.suffix = ".h5" if self.database_type == "hdf5" else ".zarr"

        # set up generic db names
        self.db_fname_train = Path(
            self.database_dir / f"train_consep"
        ).with_suffix(self.suffix)

        self.db_fname_test = Path(
            self.database_dir / f"test_consep"
        ).with_suffix(self.suffix)

    @staticmethod
    def get_classes() -> Dict[str, int]:
        return {
            "bg":0, 
            "miscellanous":1, 
            "inflammatory":2, 
            "epithelial":3, 
            "spindle":4
        } 

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
    def download(download_dir: Union[Path, str], convert_classes: bool=True) -> Dict[str, Path]:
        """
        Download CoNSeP dataset from: 
        https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/.
        and optionally creduce the classes from 8 to 5. See their README..

        Args:
        -----------
            download_dir (str or Path):
                The directory where the data is downloaded
            convert_classes (bool, default=True):
                Convert the original classes to the reduced set of classes
                See the paper.
        
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
    def write_db(img_dir: Union[Path, str],
                 mask_dir: Union[Path, str],
                 save_dir: Union[Path, str],
                 phase: str,
                 database_type: str, 
                 classes: Dict[str, int],
                 augment: bool=True,
                 stride_size: int=80,
                 patch_shape: Tuple[int, int]=(512, 512),
                 crop_shape: Tuple[int, int]=(256,256)) -> None:
        """
        Write overlapping (img [.png], mask [.mat]) pairs to either Zarr or 
        HDF5 db

        Args:
        ---------
            img_dir (Path or str):
                Path to the dir containing the image .png files
            mask_dir (Path or str):
                Path to the dir containing the corresponding mask .mat files
            save_dir (Path or str):
                Path to the dir where the DB is written
            phase (str):
                One of ("train", "test")
            database_type (str):
                One of ("zarr", "hdf5")
            classes (Dict[str, int]):
                classes dictionary e.g. {"bg":0, "neoplastic":1, "inflamma":2}
            augment (bool, default=True):
                If True, rigid augmentations are applied to the (img, mask) 
                pairs
            patch_shape (Tuple[int, int], default=(512, 512)):
                Specifies the height and width of the patches. If this is None,
                no patching is applied.
            stride_size (int, default=80):
                Stride size for the sliding window patcher. Needs to be less or
                equal to patch_shape. If less than patch_shape, patches are 
                created with overlap. Ignored if patch_shape is None.
            crop_shape (Tuple[int, int], default=(256, 256)):
                If augment is True, this is the crop shape for the center crop.
        """
        assert database_type in ("zarr", "hdf5")
        writerobj = HDF5Writer if database_type == "hdf5" else ZarrWriter 

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
        if self.db_fname_train.exists():
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
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = DatasetBuilder.set_train_dataset(
            decoder_aux_branch=self.aux_branch,
            augmentations=self.augs,
            fname=self.db_fname_train.as_posix(),
            normalize_input=self.norm
        )
        self.validset = DatasetBuilder.set_test_dataset(
            decoder_aux_branch=self.aux_branch,
            fname=self.db_fname_test.as_posix(),
            normalize_input=self.norm
        )
        self.testset = DatasetBuilder.set_test_dataset(
            decoder_aux_branch=self.aux_branch,
            fname=self.db_fname_test.as_posix(),
            normalize_input=self.norm
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validset, 
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset,
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=self.num_workers
        )
