import zipfile
import cv2
import scipy.io
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf
from distutils import dir_util, file_util
from sklearn.model_selection import train_test_split
from src.img_processing.viz_utils import draw_contours
from src.settings import DATA_DIR, CONF_DIR


# This slipped to spaghetti...
class ProjectFileManager:
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig,
                 **kwargs) -> None:
        """
        A class for managing the files and folders needed in this project.
        
            Args:
                dataset_args (DictConfig): omegaconfig DictConfig specifying arguments
                                           related to the dataset that is being used.
                                           config.py for more info
                experiment_args (DictConfig): omegaconfig DictConfig specifying arguments
                                              that are used for creating result folders and
                                              files. Check config.py for more info
        """
        super().__init__(**kwargs)
        self.dsargs = dataset_args
        self.exargs = experiment_args
    
    
    @classmethod
    def from_conf(cls, conf: DictConfig):
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        
        return cls(
            dataset_args,
            experiment_args
        )
    
    
    @staticmethod
    def read_img(path: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    
    @staticmethod
    def read_mask(path: str, type_map: bool = False) -> np.ndarray:
        key = "type_map" if type_map else "inst_map"
        return scipy.io.loadmat(path)[key].astype("int32")
     
        
    @staticmethod
    def remove_existing_files(directory: str) -> None:
        assert Path(directory).exists(), f"{directory} does not exist"
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
    def experiment_dir(self) -> Path:
        ex_root = self.exargs.experiment_root_dir
        ex_version = self.exargs.experiment_version
        model_name = self.exargs.model_name
        return Path(ex_root).joinpath(f"{model_name}/version_{ex_version}")
    
    
    @property
    def dataset(self) -> str:
        assert self.dsargs.dataset in ("kumar","consep","pannuke","dsb2018","cpm")
        return self.dsargs.dataset
    
    
    @property
    def data_dirs(self) -> Dict:
        assert Path(DATA_DIR / self.dataset).exists(), (
            f"The folder {Path(DATA_DIR / self.dataset)} for the dataset {self.dataset} does not exist"
        )
        return {
            "raw_data_dir":Path(DATA_DIR / self.dataset),
            "train_im":Path(DATA_DIR / self.dataset / "train" / "images"),
            "train_gt":Path(DATA_DIR / self.dataset / "train" / "labels"),
            "test_im":Path(DATA_DIR / self.dataset / "test" / "images"),
            "test_gt":Path(DATA_DIR / self.dataset / "test" / "labels"),
        }
    
    
    @property
    def database_dir(self) -> Path:
        assert self.dsargs.patches_dtype in ("npy", "hdf5"), f"{self.dsargs.patches_dtype}"
        return Path(self.dsargs[f"{self.dsargs.patches_dtype}_patches_root_dir"]).joinpath(f"{self.dataset}")
    
    
    @property
    def phases(self) -> List:
        assert self.dsargs.phases in (["train", "valid", "test"], ["train", "test"]), f"{self.dsargs.phases}"
        return self.dsargs.phases
    
    
    @property
    def classes(self) -> Dict:
        assert self.dsargs.class_types in ("binary", "types")
        yml_path = [f for f in Path(CONF_DIR).iterdir() if self.dataset in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        return data_conf.class_types[self.dsargs.class_types]
        
    
    @property
    def pannuke_folds(self) -> Dict:
        assert self.dataset == "pannuke", f"dataset: {self.dataset}. dataset not pannuke"
        yml_path = [f for f in Path(CONF_DIR).iterdir() if self.dataset in f.name][0]
        data_conf = OmegaConf.load(yml_path)
        return data_conf.folds
    
    
    @property
    def pannuke_tissues(self) -> List:
        assert self.dataset == "pannuke", f"dataset: {self.dataset}. dataset not pannuke"
        return self.dsargs.tissues
       
        
    @property
    def data_folds(self) -> Dict:
        """
        Get the 'train', 'valid', 'test' fold paths. This also splits training set to train and valid
        if 'valid is in self.phases list'
        
        Returns:
            Dict[str, Dict[str, str]]
            Dictionary where keys (train, test, valid) point to dictionaries with keys img and mask.
            The innermost dictionary values where the img and mask -keys point to are sorted lists 
            containing paths to the image and mask files. 
        """
        assert all(Path(p).exists() for p in self.data_dirs.values()), (
            "Data folders do not exists Convert the data first to right format. See step 1."
        )
        
        train_imgs = self.__get_files(self.data_dirs["train_im"])
        train_masks = self.__get_files(self.data_dirs["train_gt"])
        test_imgs = self.__get_files(self.data_dirs["test_im"])
        test_masks = self.__get_files(self.data_dirs["test_gt"])
        
        if "valid" in self.phases:
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
    def databases(self) -> Dict:
        """
        Get the the databases that were written by the PatchWriter
        
        Returns:
            Dict[str, Dict[int, str]]
            A dictionary where the keys (train, test, valid) are pointing to dictionaries with
            values are the paths to the hdf5 databases and keys are the size of the square 
            patches saved in the databass
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
                    
        assert self.database_dir.exists(), (
            f"Database dir: {self.database_dir} not found, Create the dbs first. Check instructions."
        )
            
        if "valid" in self.phases:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_valid_*"))
            test_dbs = list(self.database_dir.glob("*_test_*"))
        else:
            train_dbs = list(self.database_dir.glob("*_train_*"))
            valid_dbs = list(self.database_dir.glob("*_test_*")) # test set used for both valid and test
            test_dbs = list(self.database_dir.glob("*_test_*")) # remember to not use the best model w this
            
        assert train_dbs, (
            f"{train_dbs} HDF5 training db not found. Create the dbs first. Check instructions."
        )
        
        assert valid_dbs, (
            f"{valid_dbs} HDF5 validation db not found. Create the dbs first. Check instructions."
        )
        
        return {
            "train":get_input_sizes(train_dbs),
            "valid":get_input_sizes(valid_dbs),
            "test":get_input_sizes(test_dbs)
        }
    
    
    def __get_img_suffix(self, directory: Path) -> str:
        assert all([f.suffix for f in directory.iterdir()]), "All image files should be in same format"
        return [f.suffix for f in directory.iterdir()][0]
        
        
    def __get_files(self, directory: str) -> List[str]:
        d = Path(directory)
        assert d.exists(), f"Provided directory: {d.as_posix()} does not exist."
        file_suffix = self.__get_img_suffix(d)
        return sorted([x.as_posix() for x in d.glob(f"*{file_suffix}")])
        
        
    def __split_training_set(self, 
                             train_imgs: List, 
                             train_masks: List,
                             seed: int = 42,
                             size: float = 0.2) -> None:
        """
        Split training set into training and validation set. This might affect training 
        accuracy if training set is small. Pannuke is split by the folds specified in the pannuke.yml
        """
        def split_pannuke(paths):
            phases = {}
            for fold, phase in self.pannuke_folds.items():
                phases[phase] = []
                for item in paths:
                    if fold in Path(item).name:
                        phases[phase].append(item)
            return phases
        
        if self.dataset == "pannuke":
            imgs = split_pannuke(train_imgs)
            masks = split_pannuke(train_masks)
            train_imgs = sorted(imgs["train"])
            train_masks = sorted(masks["train"])
            valid_imgs = sorted(imgs["valid"])
            valid_masks = sorted(masks["valid"])
        else:
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
    
    
    def __kumar_xml2mat(self, x: Path, to: Path) -> None:
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
    
    
    def __mv_to_orig(self, folder: Path) -> Path:
        p = folder.joinpath("orig")
        self.create_dir(p)
        for f in folder.iterdir():
            orig = f if f.is_dir() and f.name == "orig" else None
            if orig is None:
                f.rename(folder/"orig"/f.name)
        return p
    
    
    def __save_overlays(self, img_path: Path, ann_path: Path, type_overlays: bool = False) -> None:
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
    
    
    def __handle_kumar(self,
                       orig_dir: Path,
                       imgs_train_dir: Path,
                       anns_train_dir: Path,
                       imgs_test_dir: Path,
                       anns_test_dir: Path) -> None:
        
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
        
        
    def __handle_consep(self, root: Path, orig_dir: Path) -> None:
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

                d.rename(Path(d.parents[0] / d.name.lower()))

        
        
    def __handle_pannuke(self, 
                         orig_dir: Path,
                         imgs_train_dir: Path, 
                         anns_train_dir: Path, 
                         imgs_test_dir: Path, 
                         anns_test_dir: Path) -> None:
        
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
                    dir_key = self.pannuke_folds[f"fold{i}"]
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
        
        
    def __handle_dsb2018(self) -> None:
        pass
    
    
    def __handle_cpm(self) -> None:
        pass
    
    
    def handle_raw_data(self, rm_zips: bool = False, overlays: bool = True) -> None:
        """
        Convert the raw data to the right format and move the files to the right folders for training and
        inference. Training and inference expects a specific folder and file structure and this automates
        that stuff.
        
        Args:
            rm_zips (bool): delete the .zp files after they are extracted
            overlays (bool): plot annotation contours on top of original images.
            pannuke_folds (Dict): If dataset == "pannuke", this dictionary specifies which pannuke folds
                                  are used as 'train', 'valid' and 'test' fold. If only 'train' and 'test'
                                  phases are used then the extra 'valid' fold is added to the 'train' fold.
                                  If dataset is not "pannuke", this arg is ignored.
                                  
        """        
        root = Path(self.data_dirs["raw_data_dir"])
        assert any(p.iterdir()), f"The directory: {root} is empty. Move the raw data there first and repeat."
        assert not all((root.joinpath("train").exists(),  root.joinpath("test").exists())), (
            "test and training directories already exists. To run this again, remove the files "
            f"and folders from: {raw_root} and repeat the steps in the instructions."
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
            self.__handle_kumar(
                orig_dir, 
                imgs_train_dir, 
                anns_train_dir, 
                imgs_test_dir, 
                anns_test_dir
            )
        elif self.dataset == "pannuke":
            self.__handle_pannuke(
                orig_dir, 
                imgs_train_dir,
                anns_train_dir, 
                imgs_test_dir, 
                anns_test_dir
            )
        elif self.dataset == "consep":
            self.__handle_consep(root, orig_dir)
        elif dataset == "cpm":
            pass # TODO
        elif dataset == "dsb2018":
            pass # TODO
        
        # draw overlays
        if overlays:
            type_overlays = True if self.dataset == "pannuke" else False
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
        assert which in ("best", "last"), f"param which: {which} not one of ('best', 'last')"
        assert self.experiment_dir.exists(), (
            f"Experiment dir: {self.experiment_dir} does not exist. Train the model first."
        )
        
        ckpt = None
        for item in self.experiment_dir.iterdir():
            if item.is_file() and item.suffix == ".ckpt":
                if which == "last" and "last" in str(item) and ckpt is None:
                    ckpt = item
                elif which == "best" and "last" not in str(item) and ckpt is None:
                    ckpt = item
        
        assert ckpt is not None, (
            f"ckpt: {ckpt}. Checkpoint is None. Make sure that a .ckpt file exists in experiment dir"
        )
        return ckpt
    
    
    def get_pannuke_fold(self, folds: List, types: List, data:str = "both") -> Dict:
        """
        After converting the data to right format and moving it to right folders this can be used to get
        the file paths by pannuke fold.
        
        Args:
            folds (List[str]): list of the folds you want to include. e.g. ["fold", "fold3"]
            types (List[str]): list of the tissue types you want to include e.g. ["Breast", "Colon"].
            data (str): one of "img", "mask" or "both"
        Returns:
            Dict[str, List[Path]]
            A Dict of Path objects to the pannuke files specified by the options
            ex. {"img":[Path1, Path2], "mask":[]}
        """
        assert self.dataset == "pannuke"
        assert all(fold in list(self.pannuke_folds.keys()) for fold in folds), (
            f"incorrect folds: {folds}. Folds need to be in {list(self.pannuke_folds.keys())}"
        )
        assert all(tissue in self.pannuke_tissues for tissue in types), (
            f"types need to be in {self.pannuke_tissues}"
        )
        assert data in ("img", "mask", "both"), (
            f"data arg: {data} needs to be one of ('img', 'mask', 'both')"
        )
        assert all(Path(p).exists() for p in self.data_dirs.values()), (
            "Data folders do not exists Convert the data first to right format. See step 1."
        )
        
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
        for d in dict(self.data_dirs).values():
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

    