import zipfile
import cv2
import scipy.io
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
from distutils import dir_util, file_util
from sklearn.model_selection import train_test_split
from img_processing.viz_utils import draw_contours


class ProjectFileManager:
    def __init__(self, 
                 dataset: str, 
                 data_dirs: Dict, 
                 database_root: str,
                 experiment_root: str,
                 experiment_version: str,
                 model_name: str,
                 phases: List,
                 **kwargs: Dict) -> None:
        """
        This class is used for managing the files and folders needed in this project. 
        
            Args:
                dataset (str) : one of ("kumar", "consep", "pannuke", "other")
                data_dirs (dict) : dictionary of directories containing masks and images. Keys of this
                                   dict must be the one of ("kumar", "consep", "pannuke", "other").
                                   Check config.
                database_root (str) : directory where the databases are written
                experiment_root (str) : directory where results from a network training 
                                        and inference experiment are written
                experiment_version (str) : a name for the experiment you want to conduct. e.g. 
                                           'FPN_test_pannuke' or 'Unet_consep' etc. This name will be used
                                           for results folder of training and inference results
                model_name (str) : The name of the model used in the experiment. This name will be used
                                   for results folder of training and inference results
                phases (list) : list of the phases (["train", "valid", "test"] or ["train", "test"])
        """
        super().__init__(**kwargs)
        self.validate_data_args(dataset, data_dirs, phases, database_root, experiment_root)
        self.dataset = dataset
        self.data_dirs = data_dirs
        self.database_root = database_root
        self.experiment_root = experiment_root
        self.experiment_version = experiment_version
        self.model_name = model_name
        self.phases = phases
        
        # TODO:
        # self.patches_root = Path(patches_root)
        # self.patches_npy_dir = self.patches_root.joinpath(f"{self.dataset}")
        
    
    @classmethod
    def from_conf(cls, conf):
        dataset = conf["dataset"]["args"]["dataset"]
        data_dirs = conf["paths"]["data_dirs"]
        database_root = conf["paths"]["database_root_dir"]
        experiment_root = conf["paths"]["experiment_root_dir"]
        experiment_version = conf["experiment_args"]["experiment_version"]
        model_name = conf["experiment_args"]["model_name"]
        phases = conf["dataset"]["args"]["phases"]
        
        return cls(
            dataset,
            data_dirs,
            database_root,
            experiment_root,
            experiment_version,
            model_name,
            phases
        )
    
    
    @staticmethod
    def validate_data_args(dataset, data_dirs, phases, database_root, experiment_root):        
        assert dataset in ("kumar","consep","pannuke","dsb2018","cpm","other"), f"input dataset: {dataset}"
        assert set(data_dirs.keys()) == {"kumar","consep","dsb2018","cpm","pannuke","other"},(
                f"{data_dirs.keys()} unknown key added in the data_dirs keys."
        )
        assert phases in (["train", "valid", "test"], ["train", "test"]), f"{phases}"
        assert Path(database_root).exists(), f"database_root: {database_root} not found. Check config"
        assert Path(experiment_root).exists(), f"experiment_root: {experiment_root} not found. Check config"
        
        data_paths_bool = [Path(p).exists() for d in data_dirs.values() for p in d.values()]
        assert any(data_paths_bool), ("None of the config file 'data_dirs' paths exist. "
                                      "Make sure they are created, by runnning these... TODO")
        
        
        
    @staticmethod
    def read_img(path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    
    @staticmethod
    def read_mask(path: str, type_map: bool = False) -> np.ndarray:
        key = "type_map" if type_map else "inst_map"
        return scipy.io.loadmat(path)[key].astype("int32")
    
        
    @staticmethod
    def remove_existing_files(directory: str) -> None:
        assert Path(directory).is_dir(), f"{directory} is not a folder that can be iterated"
        for item in Path(directory).iterdir():
            # Do not touch the folders inside the directory
            if item.is_file():
                item.unlink()


    @staticmethod      
    def create_dir(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
                    
            
    @staticmethod
    def extract_zips(folder: str, rm: bool = False) -> None:
        for f in Path(folder).iterdir():
            if f.is_file() and f.suffix == ".zip":
                with zipfile.ZipFile(f, 'r') as z:
                    z.extractall(folder)
                if rm:
                    f.unlink()
        
        
    @property
    def database_dir(self):
        return Path(self.database_root).joinpath(f"{self.dataset}")
    
    
    @property
    def experiment_dir(self):
        return Path(self.experiment_root).joinpath(f"{self.model_name}/version_{self.experiment_version}")
        
        
    @property
    def data_folds(self):
        """
        Getter for the 'train', 'valid', 'test' fold paths. Also splits training set to train and valid
        if 'valid is in phases list'
        """
        train_imgs = self.__get_files(self.data_dirs[self.dataset]["train_im"])
        train_masks = self.__get_files(self.data_dirs[self.dataset]["train_gt"])
        test_imgs = self.__get_files(self.data_dirs[self.dataset]["test_im"])
        test_masks = self.__get_files(self.data_dirs[self.dataset]["test_gt"])
        
        # optionally create validation set from training set if dataset is not pan-Nuke
        if "valid" in self.phases and self.dataset == "pannuke":
            valid_imgs = self.__get_files(self.data_dirs[self.dataset]["valid_im"])
            valid_masks = self.__get_files(self.data_dirs[self.dataset]["valid_gt"])   
        elif "valid" in self.phases:
            d = self.__split_training_set(train_imgs, train_masks)
            train_imgs = d["train_imgs"]
            train_masks = d["train_masks"]
            valid_imgs = d["valid_imgs"]
            valid_masks = d["valid_masks"]
        else:
            valid_imgs = None
            valid_masks = None
            
        return {
            "train":{"img":train_imgs, "mask":train_masks},
            "valid":{"img":valid_imgs, "mask":valid_masks},
            "test":{"img":test_imgs, "mask":test_masks},
        }
    
    
    @property
    def databases(self):
        """
        Getter for the databases that were written by the PatchWriter
        """
        def get_input_sizes(paths):
            # Paths to dbs are named as 'patchsize_dataset'. e.g. 'patch256_kumar.pytable' 
            # This will parse the name and use patchsize as key for a dict
            path_dict = {}
            for path in paths:
                fn = path.as_posix().split("/")[-1]
                size = int("".join(filter(str.isdigit, fn)))
                path_dict[size] = path
            return path_dict
                    
        assert self.database_dir.exists(), ("Database directory not found, Create " 
                                            "the dbs first. Check instructions.")
            
        if "valid" in self.phases:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_valid_*"))
            test_dbs = list(self.database_dir.glob("*_test_*"))
        else:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_test_*")) # test set used for both valid and test
            test_dbs = list(self.database_dir.glob("*_test_*")) # remember to not use the best model w this
            
        assert train_dbs, (f"{train_dbs} HDF5 training db not found. Create the dbs first. " 
                            "Check instructions.")
        
        assert valid_dbs, (f"{valid_dbs} HDF5 validation db not found. Create the dbs first. "
                            "Check instructions")
        
        return {
            "train":get_input_sizes(train_dbs),
            "valid":get_input_sizes(valid_dbs),
            "test":get_input_sizes(test_dbs)
        }
    
                    
    def __get_img_suffix(self, directory):
        # Find out the suffix of the image and mask files and assert if images have mixed suffices
        d = Path(directory)
        assert all([file.suffix for file in d.iterdir()]), "All image files should be in same format"
        return [file.suffix for file in directory.iterdir()][0]
        
        
    def __get_files(self, directory):
        # Get the image and mask files from the directory that was provided 
        d = Path(directory)
        assert d.is_dir(), f"Provided directory: {d.as_posix()} for image files is not a directory."
        file_suffix = self.__get_img_suffix(d)
        return sorted([x.as_posix() for x in d.glob(f"*{file_suffix}")])
    
            
    def __split_training_set(self, train_imgs, train_masks, seed=42, size=0.2):
        """
        Split training set into training and validation set (correct way to train).
        This might affect training accuracy if training set is small.
        
        For pannuke
        """
        indices = np.arange(len(train_imgs))
        train_indices, valid_indices = train_test_split(
            indices, test_size=size, random_state=42, shuffle=True
        )
                      
        np_train_imgs = np.array(train_imgs)
        np_train_masks = np.array(train_masks)
        train_imgs = sorted(np_train_imgs[train_indices].tolist())
        train_masks = sorted(np_train_masks[train_indices].tolist())
        valid_imgs = sorted(np_train_imgs[valid_indices].tolist())
        valid_masks = sorted(np_train_masks[valid_indices].tolist())
        
        return {
            "train_imgs":train_imgs,
            "train_masks":train_masks,
            "valid_imgs":valid_imgs,
            "valid_masks":valid_masks
        }
    
                    
    def __kumar_xml2mat(self, x, to):
        """
        From https://github.com/vqdang/hover_net/blob/master/src/misc/proc_kumar_ann.py
        Convert the xml annotation files to .mat files with 'inst_map' key.
        
        Args:
            x (Path): path to the xml file
            to (Path): directory where the .mat file is written
        """
        xml = ET.parse(x)
        contour_dbg = np.zeros((1000, 1000), np.uint8)
        insts_list = []
        for idx, region_xml in enumerate(xml.findall('.//Region')):
            vertices = []
            for vertex_xml in region_xml.findall('.//Vertex'):
                attrib = vertex_xml.attrib
                vertices.append([float(attrib['X']), 
                                 float(attrib['Y'])])
            vertices = np.array(vertices) + 0.5
            vertices = vertices.astype('int32')
            contour_blb = np.zeros((1000, 1000), np.uint8)
            
            # fill both the inner area and contour with idx+1 color
            cv2.drawContours(contour_blb, [vertices], 0, idx+1, -1)
            insts_list.append(contour_blb)
            
        insts_size_list = np.array(insts_list)
        insts_size_list = np.sum(insts_size_list, axis=(1 , 2))
        insts_size_list = list(insts_size_list)
        pair_insts_list = zip(insts_list, insts_size_list)
        
        # sort in z-axis basing on size, larger on top
        pair_insts_list = sorted(pair_insts_list, key=lambda x: x[1])
        insts_list, insts_size_list = zip(*pair_insts_list)
        ann = np.zeros((1000, 1000), np.int32)
        
        for idx, inst_map in enumerate(insts_list):
            ann[inst_map > 0] = idx + 1
            
        mask_fn = Path(to / x.with_suffix(".mat").name)
        scipy.io.savemat(mask_fn, mdict={'inst_map': ann})
                
    
    def __mv_to_orig(self, folder):
        """
        Move contents of any folder to a new folder named "orig" created in the current folder
        """
        p = folder.joinpath("orig")
        self.create_dir(p)
        for f in folder.iterdir():
            orig = f if f.is_dir() and f.name == "orig" else None
            if orig is None:
                f.rename(folder/"orig"/f.name)
        return p
    
    
    def __save_overlays(self, img_path, ann_path, type_overlays=False):
        """
        Save original images with annotation contours overlayed on top
        """
        
        inst_overlay_dir = Path(img_path.parents[0] / "instance_overlays")
        self.create_dir(inst_overlay_dir)
        for im_path, ann_path in zip(sorted(img_path.glob("*")), sorted(ann_path.glob("*.mat"))):
            im = self.read_img(str(im_path))
            mask = self.read_mask(str(ann_path))
            if type_overlays:
                type_overlay_dir = Path(img_path.parents[0] / "type_overlays")
                self.create_dir(type_overlay_dir)
                type_map = self.read_mask(str(ann_path), type_map=True)
                _, type_overlay = draw_contours(mask, im, type_map=type_map)
                fn = Path(type_overlay_dir / im_path.with_suffix(".png").name)
                cv2.imwrite(str(fn), cv2.cvtColor(type_overlay, cv2.COLOR_RGB2BGR))
                
            _, inst_overlay = draw_contours(mask, im)
            fn = Path(inst_overlay_dir / im_path.with_suffix(".png").name)
            cv2.imwrite(str(fn), cv2.cvtColor(inst_overlay, cv2.COLOR_RGB2BGR))
                  

    def __check_raw_root(self, raw_root):
        root = Path(raw_root)
        assert not (root.joinpath("train").exists() and root.joinpath("test").exists()), (
                    "test and training directories already exists. To run this again, remove the files "
                    f"and folders from: {raw_root} and repeat the steps in the instructions."
        )
        
        
    def __handle_kumar(self, orig_dir, imgs_train_dir, anns_train_dir, imgs_test_dir, anns_test_dir):
        self.create_dir(anns_train_dir)
        self.create_dir(imgs_train_dir)
        self.create_dir(anns_test_dir)
        self.create_dir(imgs_test_dir)
        
        for f in orig_dir.iterdir():
            if f.is_dir() and "training" in f.name.lower():
                for item in f.iterdir():
                    if item.name == "Tissue Images":
                        # item.rename(item.parents[2]/"train") #cut/paste
                        dir_util.copy_tree(str(item), str(imgs_train_dir)) #copy/paste
                    elif item.name == "Annotations":
                        for ann in item.iterdir():
                            self.__kumar_xml2mat(ann, anns_train_dir)
                            
            elif f.is_dir() and "test" in f.name.lower():
                for ann in f.glob("*.xml"):
                    self.__kumar_xml2mat(ann, anns_test_dir)
                for item in f.glob("*.tif"):
                    file_util.copy_file(str(item), str(imgs_test_dir))
                    
    
    def __handle_consep(self, root, orig_dir):
        for item in orig_dir.iterdir():
            if item.is_dir() and item.name == "CoNSeP":
                dir_util.copy_tree(str(item), str(root))
        
        for d in root.iterdir():
            if d.is_dir() and d.name != "orig":
                for item in d.iterdir():
                    if item.name == "Overlay":
                        item.rename(Path(d / "type_overlays"))
                    else:
                        item.rename(Path(d / item.name.lower()))
                d.rename(d.as_posix().lower())
                
                
    def __handle_pannuke(self, pannuke_folds, orig_dir, imgs_train_dir, 
                         anns_train_dir, imgs_test_dir, anns_test_dir):
        
        self.create_dir(anns_train_dir)
        self.create_dir(imgs_train_dir)
        self.create_dir(anns_test_dir)
        self.create_dir(imgs_test_dir)
        
        fold_files = {
            f"{file.parts[-2]}_{file.name[:-4]}": file for dir1 in orig_dir.iterdir() if dir1.is_dir()
            for dir2 in dir1.iterdir() if dir2.is_dir()
            for dir3 in dir2.iterdir() if dir3.is_dir()
            for file in dir3.iterdir() if file.is_file() and file.suffix == ".npy"
        }
        
        fold_phases = {
            "train":{
                "img":imgs_train_dir,
                "mask":anns_train_dir
            },
            "valid":{
                "img":imgs_train_dir,
                "mask":anns_train_dir
            },
            "test":{
                "img":imgs_test_dir,
                "mask":anns_test_dir
            }
        }
        
        for i in range(1, 4):
            masks = np.load(fold_files[f"fold{i}_masks"]).astype("int32")
            imgs = np.load(fold_files[f"fold{i}_images"]).astype("uint8")
            types = np.load(fold_files[f"fold{i}_types"])
            
            for ty in np.unique(types):
                imgs_by_type = imgs[types == ty]
                masks_by_type = masks[types == ty]
                for j in range(imgs_by_type.shape[0]):
                    name = f"{ty}_fold{i}_{j}"
                    dir_key = pannuke_folds[f"fold{i}"]
                    img_dir = fold_phases[dir_key]["img"]
                    mask_dir = fold_phases[dir_key]["mask"]

                    fn_im = Path(img_dir / name).with_suffix(".png")
                    cv2.imwrite(str(fn_im), cv2.cvtColor(imgs_by_type[j, ...], cv2.COLOR_RGB2BGR))
                    
                    # Create inst map
                    temp_mask = masks_by_type[j, ...]
                    inst_map = np.zeros(temp_mask.shape[:2], dtype=np.int32)
                    type_map = np.zeros(temp_mask.shape[:2], dtype=np.int32)
                    for t, l in enumerate(range(temp_mask.shape[-1]-1), 1):
                        inst_map += temp_mask[..., l]
                        temp_type = np.copy(temp_mask[..., l])
                        temp_type[temp_type > 0] = t
                        type_map += temp_type
                        
                    # if two cells overlap, adding both classes to type_map will create
                    # a sum of those classes for those pixels and things would break so
                    # we'll just remove the overlaps here.
                    type_map[type_map > temp_mask.shape[-1]-1] = 0
                    fn_mask = Path(mask_dir / name).with_suffix(".mat")
                    scipy.io.savemat(fn_mask, mdict={"inst_map": inst_map, "type_map":type_map})
                    
                    
    def __handle_dsb2018(self):
        pass
    
    
    def __handle_cpm(self):
        pass
    
        
    def handle_raw_data(self, 
                        dataset: str, 
                        raw_root: str, 
                        rm_zips: bool = False, 
                        overlays: bool = True, 
                        pannuke_folds: Dict[str, str] = {"fold1":"train", "fold2":"valid", "fold3":"test"}
                       ) -> None:
        """
        Convert the raw data to the right format and move the files to the right folders for training and
        inference. Training and inference expects a specific folder and file structure and this automates
        that stuff.
        
        Args:
            dataset (str): one of ("kumar","consep","pannuke","dsb2018","cpm","other")
            raw_root (str): path to the raw data .zip files or extracted. See config for where to put the
                            .zip files after downloading them from the internet.
            rm_zips (bool): delete the .zp files after they are extracted
            overlays (bool): plot annotation contours on top of original images.
            pannuke_folds (Dict): If dataset == "pannuke", this dictionary specifies which pannuke folds
                                  are used as 'train', 'valid' and 'test' fold. If only 'train' and 'test'
                                  phases are used then the extra 'valid' fold is added to the 'train' fold.
                                  If dataset is not "pannuke", this arg is ignored.
                                  
        """
        assert all(key in ("fold1", "fold2", "fold3") for key in pannuke_folds.keys()), (
            f"keys of the pannuke_folds arg need to be in ('fold1', 'fold2', 'fold3')"
        )
        
        assert all(val in ("train", "valid", "test") for val in pannuke_folds.values()), (
            f"values of the pannuke_folds arg need to be in ('train', 'valid', 'test')"
        )
        
        # extract .zips and move the files to 'orig' folder
        root = Path(raw_root)
        self.__check_raw_root(root)
        orig_dir = self.__mv_to_orig(root)
        self.extract_zips(orig_dir, rm_zips)
        imgs_test_dir = Path(root / "test/images")
        anns_test_dir = Path(root / "test/labels")
        imgs_train_dir = Path(root / "train/images")
        anns_train_dir = Path(root / "train/labels")
        
        # ad hoc handling of the different datasets to convert them to same format
        if dataset == "kumar":
            self.__handle_kumar(orig_dir, imgs_train_dir, anns_train_dir, imgs_test_dir, anns_test_dir)
        elif dataset == "consep":
            self.__handle_consep(root, orig_dir)
        elif dataset == "pannuke":
            self.__handle_pannuke(pannuke_folds, orig_dir, imgs_train_dir,
                                  anns_train_dir, imgs_test_dir, anns_test_dir)
        elif dataset == "cpm":
            pass # TODO
        elif dataset == "dsb2018":
            pass # TODO
        
        # draw overlays
        if overlays:
            type_overlays = True if dataset == "pannuke" else False
            self.__save_overlays(imgs_test_dir, anns_test_dir, type_overlays=type_overlays)
            self.__save_overlays(imgs_train_dir, anns_train_dir, type_overlays=type_overlays)
    
    
    def model_checkpoint(self, which:str = "last") -> Path:
        """
        Get the best or last checkpoint of a trained network.
        Args:
            which (str): one of ("best", "last"). Specifies whether to use last epoch model or best model on
                         validation data.
        Returns:
            Path object to the pytorch checkpoint file.
        """
        assert self.experiment_dir.exists(), "Experiment dir does not exist. Train the model first."
        assert which in ("best", "last"), f"param which: {which} not one of ('best', 'last')"
        
        ckpt = None
        for item in self.experiment_dir.iterdir():
            if item.is_file() and item.suffix == ".ckpt":
                if which == "last" and "last" in str(item) and ckpt is None:
                    ckpt = item
                elif which == "best" and "last" not in str(item) and ckpt is None:
                    ckpt = item
        
        assert ckpt is not None, f"{ckpt} checkpoint is None."
        return ckpt
    
    
    def get_pannuke_fold(self, folds: List[str], types: List[str], data:str = "both") -> Dict[str, List[Path]]:
        """
        After converting the data to right format and moving it to right folders this can be used to get
        the file paths by pannuke fold.
        
        Args:
            folds (List): list of the folds you want to include. e.g. ["fold", "fold3"]
            types (List): list of the tissue types you want to include e.g. ["Breast", "Colon"]
            data (str): one of "img", "mask" or "both"
        Returns:
            A Dict of Path objects to the pannuke files specified by the options
            ex. {"img":[Path1, Path2], "mask":[]}
        """
        assert data in ("img", "mask", "both"), f"data arg: {data} needs to be one of ('img', 'mask', 'both')"
        assert all(fold in ("fold1", "fold2", "fold3") for fold in folds), f"incorrect folds: {folds}"
        assert all(Path(p).exists() for p in self.data_dirs["pannuke"].values()), ("Data folders do not exists"
                                                                                   "Convert the data first to"
                                                                                   "right format. See step 1."
                                                                                   "from the instructions.")
        
        # Use only unique values in given lists and sort
        folds = sorted(list(set(folds)))
        types = sorted(list(set(types)))
        
        # image type wildcard
        if data == "img":
            wc2 = ".png"
        elif data == "mask":
            wc2 = ".mat"
        else:
            wc2 = ""
        
        paths = []
        for d in dict(self.data_dirs["pannuke"]).values():
            for tissue in types:
                tf = list(map(lambda fold : f"{tissue}_{fold}", folds))
                for wc in tf:
                    for f in sorted(Path(d).glob(f"*{wc}*{wc2}")):
                        paths.append(f)
        
        path_dict = {}
        if data == "both":
            imgs = [path for path in sorted(paths) if path.suffix == ".png"]
            masks = [path for path in sorted(paths) if path.suffix == ".mat"]
            path_dict["img"] = imgs
            path_dict["mask"] = masks
        else:
            path_dict[data] = paths
        
        return path_dict
    
    
    # TODO: automatically download the files from the links in github
    def download_url(self):
        pass
        # import requests
        # 
        # def download_file_from_google_drive(id, destination):
        #     def get_confirm_token(response):
        #         for key, value in response.cookies.items():
        #             if key.startswith('download_warning'):
        #                 return value
        # 
        #         return None
        # 
        #     def save_response_content(response, destination):
        #         CHUNK_SIZE = 32768
        # 
        #         with open(destination, "wb") as f:
        #             for chunk in response.iter_content(CHUNK_SIZE):
        #                 if chunk: # filter out keep-alive new chunks
        #                     f.write(chunk)
        # 
        #     URL = "https://docs.google.com/uc?export=download"
        # 
        #     session = requests.Session()
        # 
        #     response = session.get(URL, params = { 'id' : id }, stream = True)
        #     token = get_confirm_token(response)
        # 
        #     if token:
        #         params = { 'id' : id, 'confirm' : token }
        #         response = session.get(URL, params = params, stream = True)
        # 
        #     save_response_content(response, destination)

    
