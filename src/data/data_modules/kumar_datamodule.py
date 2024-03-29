import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
from torch.utils.data import DataLoader

from src.settings import DATA_DIR, PATCH_DIR
from src.dl.utils import to_device
from src.utils import FileHandler
from ..datasets.dataset_builder import DatasetBuilder
from ..writers import HDF5Writer, ZarrWriter
from ..downloaders import KUMAR


class KumarDataModule(pl.LightningDataModule, FileHandler):
    def __init__(
            self, 
            database_type: str,
            augmentations: List[str]=["hue_sat", "non_rigid", "blur"],
            normalize: bool=False,
            aux_branch: str=True,
            rm_touching_nuc_borders: bool=False,
            edge_weights: bool=False,
            batch_size: int=8,
            num_workers: int=8,
            download_dir: Union[str, Path]=None, 
            database_dir: Union[str, Path]=None,
            convert_classes: bool=True
        ) -> None:
        """
        Kumar dataset lightning datamodule

        Kumar dataset paper:
        --------------
        N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and 
        A. Sethi, "A Dataset and a Technique for Generalized Nuclear 
        Segmentation for Computational Pathology," in IEEE Transactions 
        on Medical Imaging, vol.36, no. 7, pp. 1550-1560, July 2017

        Args:
        -----------
            database_type (str):
                One of ("zarr", "hdf5"). The files are written in either
                zarr or hdf5 files that is used by the torch dataloader 
                during training.
            augmentations (List, default=["hue_sat","non_rigid","blur"])
                List of augmentations. Allowed augs: "hue_sat", "rigid",
                "non_rigid", "blur", "non_spatial", "normalize"
            normalize (bool, default=False):
                If True, channel-wise min-max normalization is applied 
                to input imgs in the dataloading process
            aux_branch (str, default="hover"):
                Signals that the dataset needs to prepare an input for 
                an auxiliary branch in the __getitem__ method. One of: 
                "hover", "dist", "contour", None. If None, assumes that
                the network does not contain auxiliary branch and the
                unet style dataset (edge weights) is used as the dataset
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
            edge_weights (bool, default=False):
                If True, each dataset iteration will create weight maps
                for the nuclear edges. This can be used to penalize
                nuclei edges in cross-entropy based loss functions.
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading 
                process.
            download_dir (str, or Path obj, default=None):
                directory where the downloaded data is located. If None, 
                and downloading is required, will be downloaded in 
                .../Dippa/data/kumar/ folders.
            database_dir (str or Path, default=None):
                The directory where the db is located. If None, and
                writing is required, will be downloaded in 
                .../Dippa/patches/kumar/ folders
            
        """
        super(KumarDataModule, self).__init__()

        self.database_dir = Path(PATCH_DIR  / f"{database_type}" / "kumar")
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
        self.edge_weights = edge_weights
        self.rm_touching_nuc_borders = rm_touching_nuc_borders
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.suffix = ".h5" if self.database_type == "hdf5" else ".zarr"

        # set up generic db names
        self.db_fname_train = Path(
            self.database_dir / f"train_kumar"
        ).with_suffix(self.suffix)

        self.db_fname_test = Path(
            self.database_dir / f"test_kumar"
        ).with_suffix(self.suffix)

    @property
    def class_dicts(self) -> Dict[str, int]:
        return {"bg":0, "fg":1}

    @staticmethod
    def download(download_dir: Union[Path, str]) -> Dict[str, Path]:
        """
        Download Kumar train & test folds from: 
        1. Train: 
        https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA/view

        2. Test: 
        https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view

        and convert the mask xml files to .mat files containing 
        "inst_map", "type_map", "inst_centroid","inst_type", "inst_bbox"
        keys

        Args:
        -----------
            download_dir (str or Path):
                The directory where the data is downloaded
        
        Returns:
        -----------
            Dict:

            example:
            
            {
                "img_train": Path,
                "img_test": Path, 
                "mask_train": Path, 
                "mask_test": Path
            }
        """
        return KUMAR()

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
        """
        assert database_type in ("zarr", "hdf5")
        writerobj = HDF5Writer if database_type == "hdf5" else ZarrWriter 

        writer = writerobj(
            img_dir=img_dir,
            mask_dir=mask_dir,
            save_dir=save_dir,
            file_name=f"{phase}_kumar",
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
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = DatasetBuilder.set_train_dataset(
            fname=self.db_fname_train.as_posix(),
            decoder_aux_branch=self.aux_branch,
            augmentations=self.augs,
            normalize_input=self.norm,
            rm_touching_nuc_borders=self.rm_touching_nuc_borders,
            edge_weights=self.edge_weights,
            type_branch=False,
            semantic_branch=False

        )
        self.validset = DatasetBuilder.set_test_dataset(
            fname=self.db_fname_test.as_posix(),
            decoder_aux_branch=self.aux_branch,
            augmentations=None,
            normalize_input=self.norm,
            rm_touching_nuc_borders=self.rm_touching_nuc_borders,
            edge_weights=self.edge_weights,
            type_branch=False,
            semantic_branch=False
        )
        self.testset = DatasetBuilder.set_test_dataset(
            fname=self.db_fname_test.as_posix(),
            decoder_aux_branch=self.aux_branch,
            augmentations=None,
            normalize_input=self.norm,
            rm_touching_nuc_borders=self.rm_touching_nuc_borders,
            edge_weights=self.edge_weights,
            type_branch=False,
            semantic_branch=False
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

