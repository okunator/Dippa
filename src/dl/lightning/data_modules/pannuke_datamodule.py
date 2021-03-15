import pytorch_lightning as pl
from pathlib import Path
from typing import Union, List, Optional
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.data import PANNUKE, HDF5Writer, ZarrWriter
from src.dl.datasets.dataset_builder import DatasetBuilder
from src.settings import DATA_DIR, PATCH_DIR


class PannukeDataModule(pl.LightningDataModule):
    def __init__(self, 
                 database_type: str,
                 augmentations: List[str],
                 normalize: bool,
                 aux_branch: str,
                 batch_size: int,
                 num_workers: int,
                 download_dir: Union[str, Path]=None, 
                 database_dir: Union[str, Path]=None) -> None:
        """
        Pannuke lightning data module:

        Pannuke papers:
        ---------------
        Gamper, J., Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N. (2019). PanNuke: an open pan-cancer histology dataset         
        for nuclei instance segmentation and classification. In European Congress on Digital Pathology (pp. 11–19).

        Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A., Hewitt, K., & Rajpoot, N. (2020). 
        PanNuke Dataset Extension, Insights and Baselines. arXiv preprint arXiv:2003.10778.

        Licensed under: https://creativecommons.org/licenses/by-nc-sa/4.0/

        Args:
        -----------
            database_type (str):
                One of ("zarr", "hdf5") .The files are written in either
                zarr or hdf5 files that is used by the torch dataloader 
                during training.
            augmentations (List[str]):
                List of augmentations. e.g. ["hue_sat", "non_rigid", "blur"]...
                allowed augs: ("hue_sat", "rigid", "non_rigid", "blur", "non_spatial", "normalize")
            normalize (bool):
                If True, channel-wise min-max normalization is applied to input imgs 
                in the dataloading process
            aux_branch (str):
                Signals that the dataset needs to prepare an input for an auxiliary branch in
                the __getitem__ method. One of ("hover", "dist", "contour", None). 
                If None, assumes that the network does not contain auxiliary branch and
                the unet style dataset (edge weights and no overlapping cells) is used as
                the dataset. 
            batch_size (int):
                Batch size for the dataloader
            num_workers (int):
                number of cpu cores/threads used in the dataloading process.
            download_dir (str, or Path obj):
                directory where the downloaded data is located or saved to. If None, and downloading
                is required, will be downloaded in Dippa/data/pannuke/ folders.
            database_dir (str or Path):
                The directory where the db is located or saved to. If None, and writing is required,
                will be downloaded in Dippa/patches/pannuke/ folders
            
        """
        assert database_type in ("zarr", "hdf5")
        super(PannukeDataModule, self).__init__()

        self.download_dir = Path(download_dir)
        self.database_dir = Path(database_dir)
        self.database_type = database_type
        self.augs = augmentations
        self.norm = normalize
        self.aux_branch = aux_branch
        self.batch_size = batch_size
        self.num_workers = num_workers


    @classmethod
    def from_conf(cls, conf: DictConfig, download_dir=None, database_dir=None):
        download_dir = download_dir
        database_dir = database_dir
        db_type = conf.runtime_args.db_type
        
        #  If no download dir give, download to /data/pannuke
        download_dir = database_dir if database_dir is not None else Path(DATA_DIR)
        # If no database dir given write to /patches
        database_dir = database_dir if database_dir is not None else Path(PATCH_DIR / f"{db_type}" / "pannuke")

        augs = conf.training_args.augs
        norm = conf.training_args.normalize_input
        aux_branch = conf.model_args.decoder_branches.aux_branch
        batch_size = conf.runtime_args.batch_size
        num_workers = conf.runtime_args.num_workers

        return cls(
            download_dir=download_dir,
            database_dir=database_dir,
            database_type=db_type,
            augmentations=augs,
            normalize=norm,
            aux_branch=aux_branch,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        
    def prepare_data(self, write_new_dbs: bool=False):
        """
        1. Download pannuke folds from: "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/" and re-format
        2. Write images/masks to to zarr or hdf5 files

        Zarr files can be thread/file-locked easily (if needed) but generally zarr files are slower
        than hdf5 in dataloading, especially in HPC environments: 
        https://sc19.supercomputing.org/proceedings/tech_poster/poster_files/rpost191s2-file3.pdf 

        Folds are saved to save_dir, unpacked, converted to singular
        .png/.mat files (CoNSeP-format) that are then moved to train
        and test folders. After conversion, the new filenames contain
        the fold and tissue type of the patch.

        Args:
        ----------
            write_new_dbs (bool, default=True):
                If True, a new database is written to database_dir.
        """
        # Download folds and re-format. If data is downloaded and processed
        # already the downloading and processing is skipped
        fold1 = PANNUKE(save_dir=self.download_dir, fold=1, phase="train")()
        fold2 = PANNUKE(save_dir=self.download_dir, fold=2, phase="train")()
        fold3 = PANNUKE(save_dir=self.download_dir, fold=3, phase="test")()

        # Get writers
        trainwriterobj = HDF5Writer if self.database_type == "hdf5" else ZarrWriter 
        testwriterobj = HDF5Writer if self.database_type == "hdf5" else ZarrWriter

        # Pannuke classes
        classes = {"bg":0, "neoplastic":1, "inflammatory":2, "connective":3, "dead":4, "epithelial":5}

        trainwriter = trainwriterobj(
            img_dir=fold1["img_train"],
            mask_dir=fold1["mask_train"],
            save_dir=self.database_dir,
            file_name="train_pannuke",
            classes = classes,
            patch_shape=None,
            stride_size=None,
            rigid_augs_and_crop=True,
            crop_shape=(256, 256),
            n_copies=3,
            chunk_size=1
        )

        testwriter = testwriterobj(
            img_dir=fold1["img_test"],
            mask_dir=fold1["mask_test"],
            save_dir=self.database_dir,
            file_name="test_pannuke",
            classes = classes,
            patch_shape=None,
            stride_size=None,
            rigid_augs_and_crop=False,
            crop_shape=(256, 256),
            n_copies=None,
            chunk_size=1
        )


        # write new dbs if no db files exist in db_dir or write_new_db flag set to True.
        skip = False
        if self.database_dir.exists():
            skip = True if not write_new_dbs else False

        self.train_pannuke = trainwriter.write2db(skip=skip)
        self.test_pannuke = testwriter.write2db(skip=skip)
        

    def setup(self, stage: Optional[str] = None):
        self.trainset = DatasetBuilder.set_train_dataset(
            aux_branch=self.aux_branch,
            augmentations=self.augs,
            fname=self.train_pannuke.as_posix(),
            norm=self.norm
        )
        self.validset = DatasetBuilder.set_test_dataset(
            aux_branch=self.aux_branch,
            fname=self.test_pannuke.as_posix(),
            norm=self.norm
        )
        self.testset = DatasetBuilder.set_test_dataset(
            aux_branch=self.aux_branch,
            fname=self.test_pannuke.as_posix(),
            norm=self.norm
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset, 
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=self.num_workers
        )
