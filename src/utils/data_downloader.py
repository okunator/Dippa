import cv2
import numpy as np
import pooch
import scipy.io
from pathlib import Path
from typing import Optional, Dict
from pooch import HTTPDownloader
from omegaconf import DictConfig
from src.utils.file_manager import ProjectFileManager
from google_drive_downloader import GoogleDriveDownloader as gdd
from src.img_processing.viz_utils import draw_contours
from src.img_processing.process_utils import get_inst_centroid, bounding_box, get_inst_types
from src.utils.adhoc_conversions import (
    handle_consep, handle_cpm, handle_dsb2018,
    handle_kumar, handle_pannuke, kumar_xml2mat
)
from src.settings import DATA_DIR


CONSEPPOOCH = pooch.create(
    path=pooch.os_cache(f"{DATA_DIR.as_posix()}/consep"),
    base_url="https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/",
    version=None,
    version_dev="master",
    registry={
        "consep.zip": "23eb8a717551827b4d42642b08bd64370810acf6e6e73a399182c1c915dfe82a"
    }
)


PANNUKEPOOCH = pooch.create(
    path=pooch.os_cache(f"{DATA_DIR.as_posix()}/pannuke"),
    base_url="https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/",
    version=None,
    version_dev="master",
    registry={
        "fold_1.zip": "6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b",
        "fold_2.zip": "5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07",
        "fold_3.zip": "c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d"
    }
)

class Downloader(ProjectFileManager):
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig) -> None:
        """
        A class that uses pooch and google_drive_downloader to download all the data
        files used in the project
        
        Args:
            dataset_args (DictConfig): omegaconfig DictConfig specifying arguments
                                        related to the dataset that is being used.
                                        Check config.py for more info
            experiment_args (DictConfig): omegaconfig DictConfig specifying arguments
                                            that are used for creating result folders
                                            files. Check config.py for more info           
        """

        super(Downloader, self).__init__(dataset_args, experiment_args)

    @classmethod
    def from_conf(cls, conf: DictConfig):
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args

        return cls(
            dataset_args,
            experiment_args,
        )

    @staticmethod
    def download_datasets() -> None:
        """
        This downloads all the datasets that are used in the project from online
        """
        download = HTTPDownloader(progressbar=True)
        # Kumar training data
        gdd.download_file_from_google_drive(
            file_id="1JZN9Jq9km0rZNiYNEukE_8f0CsSK3Pe4",
            dest_path=f"{DATA_DIR.as_posix()}/kumar/KumarTraining.zip",
            unzip=False
        )

        # kumar testing data
        gdd.download_file_from_google_drive(
            file_id="1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw",
            dest_path=f"{DATA_DIR.as_posix()}/kumar/KumarTesting.zip",
            unzip=False
        )

        # consep data
        CONSEPPOOCH.fetch("consep.zip", downloader=download)

        # pannuke data
        PANNUKEPOOCH.fetch("fold_1.zip", downloader=download)
        PANNUKEPOOCH.fetch("fold_2.zip", downloader=download)
        PANNUKEPOOCH.fetch("fold_3.zip", downloader=download)


    def handle_raw_data(self, 
                        rm_zips: bool = False, 
                        overlays: bool = True,
                        change_consep_classes: bool = True,
                        pannuke_folds: Optional[Dict[str, str]] = None) -> None:
        """
        Convert the raw data to the right format and move the files to the right folders for training and
        inference. Training and inference expects a specific folder and file structure and this automates
        that stuff. This should be run before any experiments.
        
        Args:
            rm_zips (bool): delete the .zp files after they are extracted
            overlays (bool): plot annotation contours on top of original images.
            pannuke_folds (Dict): If dataset == "pannuke", this dictionary specifies which pannuke folds
                                  are used as 'train', 'valid' and 'test' fold. If only 'train' and 'test'
                                  phases are used then the extra 'valid' fold is added to the 'train' fold.
                                  By default the folds are set as in the pannuke.yml. 
                                  If dataset is not "pannuke", this arg is ignored. 
            change_consep_classes (bool): Changes consep classes to be the same as in their HoVer-Net paper
                                          If dataset is not "consep" this arg is ignored
                                  
        """        
        root = Path(self.data_dirs["raw_data_dir"])
        assert any(root.iterdir()), (
            f"The directory: {root} is empty. download the data first"
        )
        assert not all((root.joinpath("train").exists(),  root.joinpath("test").exists())), (
            "test and training directories already exists. To run this again, remove the files ",
            f"and folders from: {root} and download the data again."
        )
        
        # extract .zips and move the files to 'orig' folder
        orig_dir = self.__mv_to_orig(root)
        self.extract_zips(orig_dir, rm_zips)
        imgs_test_dir = Path(root / "test/images")
        anns_test_dir = Path(root / "test/labels")
        imgs_train_dir = Path(root / "train/images")
        anns_train_dir = Path(root / "train/labels")
        
        # ad hoc handling of the different datasets to convert them to same format
        if self.dataset == "kumar":
            handle_kumar(
                orig_dir, 
                imgs_train_dir, 
                anns_train_dir, 
                imgs_test_dir, 
                anns_test_dir
            )
        elif self.dataset == "pannuke":
            handle_pannuke(
                orig_dir, 
                imgs_train_dir,
                anns_train_dir, 
                imgs_test_dir, 
                anns_test_dir,
                pannuke_folds=pannuke_folds
            )
        elif self.dataset == "consep":
            handle_consep(
                orig_dir,
                imgs_train_dir,
                anns_train_dir,
                imgs_test_dir,
                anns_test_dir
            )
            if change_consep_classes:
                self.__change_consep_classes(anns_train_dir)
                self.__change_consep_classes(anns_test_dir)            
        elif self.dataset == "dsb2018":
            pass # TODO
        
        # add bboxes, centroids, int types
        self.__add_slots2masks(anns_test_dir)
        self.__add_slots2masks(anns_train_dir)

        # draw overlays
        if overlays:
            self.__save_overlays(imgs_train_dir, anns_train_dir, type_overlays=True)
            self.__save_overlays(imgs_test_dir, anns_test_dir, type_overlays=True)

    def __add_slots2masks(self, ann_dir: Path) -> None:
        """
        Takes in a  file path of the mask and adds "inst_type", 
        "inst_centroid" and "bbox" keys to the file

        Args:
            ann_dir (Path): the path to the annotation mask directory
        """
        for ann_path in sorted(ann_dir.glob("*.mat")):
            m = scipy.io.loadmat(ann_path)
            inst_map = m["inst_map"].astype("int32")
            type_map = m["type_map"].astype("int32")
            inst_ids = list(np.unique(inst_map))
            inst_ids.remove(0)
            centroids = get_inst_centroid(inst_map)
            bboxes = np.array([bounding_box(np.array(inst_map == id_, np.uint8)) for id_ in inst_ids])
            inst_types = get_inst_types(inst_map, type_map)

            scipy.io.savemat(
                file_name=ann_path,
                mdict={
                    "inst_map": inst_map,
                    "type_map": type_map,
                    "inst_type": inst_types,
                    "inst_centroid": centroids,
                    "inst_bbox": bboxes
                }
            )

    def __change_consep_classes(self, ann_dir: Path) -> None:
        """
        this combines some of the classes as described in the hover-net paper.

        Args:
            ann_dir (Path): the path to the annotation mask directory
        """
        assert self.dataset == "consep", f"dataset: {self.dataset}. dataset not consep"
        for ann_path in sorted(ann_dir.glob("*.mat")):
            # Modify classes like in the paper
            mat = scipy.io.loadmat(ann_path)
            inst_map = mat["inst_map"]
            type_map = mat["type_map"]
            type_map[(type_map == 3) | (type_map == 4)] = 3
            type_map[(type_map == 5) | (type_map == 6) | (type_map == 7)] = 4
            scipy.io.savemat(
                file_name=ann_path,
                mdict={
                    "inst_map": inst_map,
                    "type_map": type_map,
                }
            )

    def __save_overlays(self, 
                      img_dir: Path,
                      ann_dir: Path,
                      type_overlays: bool = False) -> None:
        """
        Save contour overlays on the original image

        Args:
            img_dir (Path):  path to the image directory
            ann_dir (Path): path to the annotation mask directory
            type_overlays (bool): Save type_map overlays
        """
        inst_overlay_dir = Path(img_dir.parents[0] / "instance_overlays")
        self.create_dir(inst_overlay_dir)
        for im_path, ann_path in zip(sorted(img_dir.glob("*")), sorted(ann_dir.glob("*.mat"))):
            im = self.read_img(str(im_path))
            mask = self.read_mask(str(ann_path))
            if type_overlays:
                type_overlay_dir = Path(img_dir.parents[0] / "type_overlays")
                self.create_dir(type_overlay_dir)
                type_map = self.read_mask(str(ann_path), key="type_map")
                type_overlay = draw_contours(mask, im, type_map=type_map, classes=self.classes)
                fn = Path(type_overlay_dir / im_path.with_suffix(".png").name)
                cv2.imwrite(str(fn), cv2.cvtColor(type_overlay, cv2.COLOR_RGB2BGR))
                
            inst_overlay = draw_contours(mask, im)
            fn = Path(inst_overlay_dir / im_path.with_suffix(".png").name)
            cv2.imwrite(str(fn), cv2.cvtColor(inst_overlay, cv2.COLOR_RGB2BGR))

    def __mv_to_orig(self, folder: Path) -> Path:
        """
        Creates a folder called "orig" and moves the downloaded data there

        Args:
            folder (Path): the folder where the files are moved to 'orig'
        """
        p = folder.joinpath("orig")
        self.create_dir(p)
        for f in folder.iterdir():
            orig = f if f.is_dir() and f.name == "orig" else None
            if orig is None:
                f.rename(folder/"orig"/f.name)
        return p
