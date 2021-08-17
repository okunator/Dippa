import re
import torch
import itertools
import scipy.io
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Dict, Iterable, Union, Optional, Iterable
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from src.utils.file_manager import FileHandler
from src.utils.save_utils import mask2geojson, mask2mat
from src.patching import TilerStitcherTorch
from src.metrics.benchmarker import Benchmarker
from src.dl.torch_utils import tensor_to_ndarray, ndarray_to_tensor

from .post_processing.processor_builder import PostProcBuilder
from .predictor import Predictor


SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")


class FolderDataset(Dataset, FileHandler):
    def __init__(self, 
                 folder_path: Union[str, Path],
                 pattern: Optional[str]="*", 
                 sort_by_y: Optional[bool]=False,
                 xmax: Optional[int]=None,
                 ymax: Optional[int]=None,
                 auto_range: bool=False,
                 tile_size: Optional[Tuple[int, int]]=(1000, 1000)):
        """
        Simple pytorch folder dataset. Assumes that
        folder_path contains only image files which are readable
        by cv2.

        Args:
        ----------
            folder_path (Union[str, Path]):
                path to the folder containig tile/image files
            pattern (str, optional, default="*"):
                file pattern for filtering only the files that contain the pattern.
            sort_by_y (bool, optional, default=False):
                sorts a folder (containing tiles extracted by histoprep package)
                by the y-coordinate rather than the x-coordinate
            xmax (int, optional, default=None):
                filters all the tile-files that contain x-coordinate <= than this param
                in their filename. (works with tiles extracted with histoprep).
                See https://github.com/jopo666/HistoPrep 
            ymax (int, optional, default=None):
                filters all the tile-files that contain y-coordinate <= than this param
                in their filename. (works with tiles extracted with histoprep).
                See https://github.com/jopo666/HistoPrep.
            auto_range (bool, default=False):
                Automatically filter tiles that contain ONE tissue section rather than
                every redundant tissue section in the wsi. (Less redundant segmentation work).
            tile_size (Tuple[int, int], optional, default=(1000, 1000)):
                size of the input tiles in the folder. Optional.
        """
        super(FolderDataset, self).__init__()
        self.tile_size = tile_size
        folder_path = Path(folder_path)
        assert folder_path.exists(), f"folder: {folder_path} does not exist"
        assert folder_path.is_dir(), f"given path: {folder_path} is not a folder"
        assert all([f.suffix in SUFFIXES for f in folder_path.iterdir()]) ,(
            f"files formats in given folder need to be in {SUFFIXES}"
        )

        #  sort files
        if sort_by_y:
            self.fnames = sorted(folder_path.glob(pattern), key=lambda x: self._get_xy_coords(x.name)[1])
        else:
            self.fnames = sorted(folder_path.glob(pattern))

        # filer by xy-cooridnates encoded in the filename
        if xmax is not None:
            self.fnames = [f for f in self.fnames if self._get_xy_coords(f.name)[0] <= xmax]
        if ymax is not None and not auto_range:
            self.fnames = [f for f in self.fnames if self._get_xy_coords(f.name)[1] <= ymax]
        
        if auto_range:
            ymin, ymax = self._get_auto_range(coord="y") # auto ranging only in y-direction for now
            self.fnames = [f for f in self.fnames if ymin <= self._get_xy_coords(f.name)[1] <= ymax]
        # print(self.fnames)
        # print(len(self.fnames))

    def _get_xy_coords(self, fname: str) -> Tuple[int, int]:
        """
        Extract xy-coords from files named with x- and y- coordinates in their file name (see histoprep)
        https://github.com/jopo666/HistoPrep 
        
        example filename: x-47000_y-25000.png 
        """
        xy = [int(c) for c in re.findall(r"\d+", fname)]
        return xy

    def _get_auto_range(self, coord: str="y", section_ix: int=0, section_length: int=6000) -> Tuple[int, int]:
        """
        Automatically extract a range of tiles that contain a section
        of tissue in a whole slide image. This is pretty ad hoc
        and requires histoprep extracted tiles and that the slides 
        contain many tissue sections. Use with care.

        Args:
        ---------
            coord (str, default="y"):
                specify the range in either x- or y direction
            section_ix (int, default=0):
                the nth tissue section in the wsi in the direction of the
                coord param.
            section_length (int, default=6000):
                Threshold to concentrate only on tissue sections that are 
                larger than 6000 pixels

        Returns:
        --------
            The start and end point of the tissue section in the specified direction
        """
        ix = 1 if coord == "y" else 0
        coords = sorted(set([self._get_xy_coords(f.name)[ix] for f in self.fnames]))

        splits = []
        split = []
        for i in range(len(coords)-1):
            if coords[i+1] - coords[i] == self.tile_size[ix]:
                split.append(coords[i])
            else:
                if i < len(coords) - 1:
                    split.append(coords[i]) 
                splits.append(split)
                split = []
        
        ret_splits = [split for split in splits if len(split) >= section_length//self.tile_size[ix]]
        ret_split = ret_splits[section_ix]
        return ret_split[0], ret_split[-1]

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> torch.Tensor:
        fn = self.fnames[index]
        im = FileHandler.read_img(fn.as_posix())
        im = torch.from_numpy(im.transpose(2, 0, 1))

        return {
            "im":im, 
            "file":fn.name[:-4]
        }


class Inferer:
    def __init__(self,
                 model: pl.LightningModule,
                 in_data_dir: str=None,
                 dataset: str=None,
                 data_fold: str="test",
                 tta: bool=False,
                 model_weights: str="last",
                 loader_batch_size: int=8,
                 loader_num_workers: int=8,
                 patch_size: Tuple[int]=(256, 256),
                 stride_size: int=128,
                 model_batch_size: int=None,
                 thresh_method: str="naive",
                 thresh: float=0.5,
                 apply_weights: bool=False,
                 post_proc_method: str=None,
                 n_images: int=32,
                 fn_pattern: str="*",
                 xmax: Optional[int]=None,
                 ymax: Optional[int]=None,
                 auto_range: Optional[bool]=False,
                 **kwargs) -> None:
        """
        Class to perform inference and post-processing

        Args:
        -----------
            model (pl.LightningModule):
                Input SegModel (lightning model) specified in lightning_model.py.
            in_data_dir (str, default=None):
                If not None this directory will be used as the input data directory.
                Assumes that the directory contains only cv2 readable image files (.png, .tif, etc).
                This argument overrides all other dataset related arguments (dataset, data_fold, ).
            dataset (str, default=None):
                One of ("kumar","consep","pannuke","dsb2018", "monusac", None)
                If in_data_dir == None, images from this dataset will be used for inference. 
                If both dataset == None & in_data_dir == None. The inference is performed
                on the same dataset that the input model was trained with.
            data_fold (str, default="test"):
                Which fold of data to run inference. One of ("train", "test"). 
                If in_data_dir is set this arg will be ignored.
            tta (bool, default=False):
                If True, performs test time augmentation. Inference time goes up
                with often marginal performance improvements. (Not yet implemented)
            model_weights (str, default="last"):
                pytorch lightning saves the weights of the model for the last epoch
                and best epoch (based on validation data). One of ("best", "last").
            loader_batch_size (int, default=8):
                Number of images loaded from the input folder by the workers per dataloader
                iteration. This is the DataLoader batch size, NOT the batch size that is used 
                during the forward pass of the model.
            loader_num_workers (int, default=8):
                Number of threads/workers for torch dataloader
            patch_size (Tuple[int], default=(256, 256)):
                The size of the input patches that are fed to the segmentation model.
            stride_size (int, default=128):
                If input images are larger than the model input image size (patch_size), the images 
                are tiled with a sliding window into small patches with overlap. This param is the 
                stride size used in the sliding window operation. Small stride for the sliding window 
                results in less artefacts and checkerboard effect in the resulting prediction when the 
                patches are stitched back to the input image size. On the other hand small stride_size
                means more patches and larger number of patches leads to slower inference time and larger
                memory consumption. stride_size needs to be less or equal than the input patch_size.
            model_batch_size (int, default=None):
                The batch size that is used when the input is fed to the model (actual model batch size).
                If input images need patching, and the batch size for training batch size is too large
                (cuda out of memmory error). This argument overrides the model batch size, so you can reduce
                the memory footprint. 
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"
            apply_weights (bool, default=True):
                After a prediction, apply a weight matrix that assigns bigger weight on pixels
                in center and less weight to pixels on prediction boundaries. helps dealing with
                prediction artefacts on tile/patch boundaries. NOTE: This is only applied at the
                auxiliary branch prediction since there tiling effect has the most impact.
                (Especially, in HoVer-maps)
            post_proc_method (str, default=None):
                Defines the post-processing pipeline. If this is None, then the post-processing
                pipeline is defined by the aux_type of the model. If the aux_type of the model
                is None, then the basic watershed post-processing pipeline is used. The post-processing
                method is always specific to the auxiliary maps that the model outputs so if the
                aux_type == "hover", then the HoVer-Net and CellPose pipelines can be used.
                One of (None, "hover", "cellpose", "drfns", "dcan", "dran"). 
            n_images (int, default=32):
                Number of images inferred before clearing the memory. Useful if there is a large number of
                images in a folder. The segmentation results are saved after n_images are segmented and
                memory cleared for a new set of images to be segmented.
            fn_pattern (str, default="**):
                A pattern in file names in the in_data_dir. For example, for pannuke dataset you can run 
                inference for only images of specific tissue e.g. pattern = *_Adrenal_gland_*.
            xmax (int, optional, default=None):
                Filters all the file names in the input directory that contain x-coordinate less
                than this param in their filename. I.e. the tiles in the folder need to contain the x- 
                and y- coordinates (in xy- order) in the filename. Example tile filename: "x-45000_y-50000.png". 
                (Works with tiles extracted with histoprep package. See https://github.com/jopo666/HistoPrep.)
            ymax (int, optional, default=None):
                Filters all the file names in the input directory that contain y-coordinate less
                than this param in their filename. I.e. the tiles in the folder need to contain the x- 
                and y- coordinates (in xy- order) in the filename. Example tile filename: "x-45000_y-50000.png". 
                (Works with tiles extracted with histoprep package. See https://github.com/jopo666/HistoPrep.)
            auto_range (bool, optional, default=False):
                Automatically filter tiles from a folder that contain only ONE tissue section rather than
                every redundant tissue section in the wsi. The tiles in the folder need to contain the x- 
                and y- coordinates (in xy- order) in the filename. Example tile filename: "x-45000_y-50000.png". 
                (Works with tiles extracted with histoprep package. See https://github.com/jopo666/HistoPrep.)
        """
        assert isinstance(model, pl.LightningModule), "Input model needs to be a lightning model"
        assert dataset in ("kumar", "consep", "pannuke", "dsb2018", "monusac", None)
        assert model_weights in ("best", "last")
        assert data_fold in ("train", "test")
        assert stride_size <= patch_size[0], f"stride_size: {stride_size} > {patch_size[0]}"

        # set model to device and to inference mode
        self.model = model
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()
        torch.no_grad()

        # Load trained weights for the model 
        ckpt_path = self.model.fm.get_model_checkpoint(model_weights)
        checkpoint = torch.load(ckpt_path, map_location = lambda storage, loc : storage)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        self.patch_size = patch_size
        self.stride_size = stride_size
        self.n_images = n_images

        # Set input data folder and gt mask folder if there are gt masks
        self.dataset = dataset
        self.gt_mask_dir = None
        self.in_data_dir = in_data_dir
        if self.in_data_dir is None:
            if self.dataset is not None:
                dirs = self.model.fm.get_data_dirs(self.dataset)
                self.in_data_dir = dirs[f"{data_fold}_im"]
                self.gt_mask_dir = dirs[f"{data_fold}_gt"]
            else:
                self.dataset = self.model.fm.train_dataset
                dirs = self.model.fm.get_data_dirs(self.dataset)
                self.in_data_dir = dirs[f"{data_fold}_im"]
                self.gt_mask_dir = dirs[f"{data_fold}_gt"]

            self.gt_mask_paths = sorted(self.gt_mask_dir.glob(fn_pattern))
            
        # Set dataset dataloader
        self.loader_batch_size = loader_batch_size
        self.folderset = FolderDataset(self.in_data_dir, pattern=fn_pattern, xmax=xmax, ymax=ymax, auto_range=auto_range)
        self.dataloader = DataLoader(self.folderset, batch_size=loader_batch_size, shuffle=False, pin_memory=True, num_workers=loader_num_workers)
        self.model_batch_size = model_batch_size

        # set apply weights flag for aux branch and prdeictor helper class
        self.apply_weights = apply_weights
        self.predictor = Predictor(self.model, self.patch_size)

        # set the post-processing pipeline. Defaults to 
        # model.aux_type if model has an auxiliary branch
        self.post_proc_method = post_proc_method
        if self.post_proc_method is None:
            self.post_proc_method = self.model.aux_type if self.model.aux_branch else "basic"

        # Quick checks that a valid post-proc-method is used
        msg = f"post_proc_method set to: {self.post_proc_method}, while model.decoder_aux_branch: {self.model.decoder_aux_branch}"
        if self.model.decoder_aux_branch:
            if self.model.decoder_aux_branch == "hover":
                assert self.post_proc_method in ("hover", "cellpose", "basic"), msg
            elif self.model.decoder_aux_branch == "dist":
                assert self.post_proc_method in ("drfns", "basic"), msg
            elif self.model.decoder_aux_branch == "contour":
                assert self.post_proc_method in ("dcan", "dran", "basic"), msg
        
        # init the post-processor
        self.post_processor = PostProcBuilder.set_postprocessor(
            post_proc_method=self.post_proc_method,
            thresh_method=thresh_method,
            thresh=thresh
        )

        # input norm flag and train data stats
        self.norm = self.model.normalize_input
        self.stats = self.model.fm.get_dataset_stats(self.model.train_data.as_posix())

    def _get_batch(self, patches: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Divide a set of patches into batches of patches

        Args:
        ---------
            patches (torch.Tensor): 
                Batch of patches in. Shape (C, num_patches, pH, pW)
            batch_size (int): 
                size of the batch
            
        Yields:
        ---------
            torch.Tensor of shape (batch_size, C, H, W)
        """
        for i in range(0, patches.shape[1], batch_size):
            batch = patches[:, i:i+batch_size, ...].permute(1, 0, 2, 3) # to shape (B, C, pH, pW)
            yield batch 

    def _predict_batch(self, batch: torch.Tensor) -> Tuple[Union[torch.Tensor, None]]:
        """
        Forward pass + classify. Handles missing branches in the model.

        Args:
        ---------
            batch (torch.Tensor):
                A batch of patches. Shape (B, C, patch_size, patch_size)

        Returns:
        ---------
            A tuple of tensors containing the predictions. If network does not
            contain aux or type branch the predictions are None
        """
        # TODO: tta
        pred = self.predictor.forward_pass(batch, norm=self.norm, mean=self.stats[0], std=self.stats[1])
        insts = self.predictor.classify(pred["instances"], act="softmax")
        types = self.predictor.classify(pred["types"], act="softmax") if pred["types"] is not None else None
        aux = self.predictor.classify(pred["aux"], act=None, apply_weights=self.apply_weights) if pred["aux"] is not None else None
        return insts, types, aux

    def _infer_large_img_batch(self, 
                               batch: torch.Tensor, 
                               names: Tuple[str],
                               batch_loader: Iterable=None) -> Tuple[Iterable[Tuple[str, np.ndarray]]]:
        """
        Run inference on large images that require tiling and back stitching. I.e. For images 
        larger than the model input size.

        Args:
        --------
            batch (torch.Tensor):
                A batch of patches. Shape (B, C, patch_size, patch_size)
            names (Tuple[str]):
                filenames of the different images (without the suffices)
            batch_loader (Iterable, default=None):
                tqdm loader object

        Returns:
        --------
            Tuple of Zip (iterable) objects containing (name, np.ndarray) pairs
        """

        # Tile the image into patches
        tilertorch = TilerStitcherTorch(batch.shape, self.patch_size, self.stride_size, padding=True)
        patches = tilertorch.extract_patches_from_batched(batch)

        # (for tqdm logging)
        n_patches_total = patches.shape[2]*(len(batch_loader)*self.loader_batch_size)
        
        batch_instances = []
        batch_types = []
        batch_aux = []

        # model batch size
        batch_size = self.model_batch_size if self.model_batch_size is not None else self.model.batch_size

        # Loop through the B in batched patches (B, C, n_patches, patch_h, patch_w)
        for j in range(patches.shape[0]):

            pred_inst = []
            pred_type = []
            pred_aux = []

            # Divide patches into batches that can be used as input to model
            for batch in self._get_batch(patches[j, ...], batch_size):
                insts, types, aux = self._predict_batch(batch)
                pred_inst.append(insts)
                pred_type.append(types) if types is not None else None
                pred_aux.append(aux) if aux is not None else None

                self.n_batches_inferred += batch.shape[0]
                if batch_loader is not None:
                    batch_loader.set_postfix(patches=f"{self.n_batches_inferred}/{n_patches_total}")
                
            pred_inst = torch.cat(pred_inst, dim=0)
            pred_type = torch.cat(pred_type, dim=0) if pred_type else None
            pred_aux = torch.cat(pred_aux, dim=0) if pred_aux else None
            
            batch_instances.append(pred_inst)
            batch_types.append(pred_type)
            batch_aux.append(pred_aux)

        # Stitch the patches back to the orig img size
        insts = torch.stack(batch_instances, dim=0).permute(0, 2, 1, 3, 4)
        insts = tilertorch.stitch_batched_patches(insts)
        insts = zip(names, tensor_to_ndarray(insts))

        types = zip(names, [None]*len(names))
        if all(e is not None for e in batch_types):
            types = torch.stack(batch_types, dim=0).permute(0, 2, 1, 3, 4)
            types = tilertorch.stitch_batched_patches(types)
            types = zip(names, tensor_to_ndarray(types))

        aux = zip(names, [None]*len(names))
        if all(e is not None for e in batch_aux):
            aux = torch.stack(batch_aux, dim=0).permute(0, 2, 1, 3, 4)
            aux = tilertorch.stitch_batched_patches(aux)
            aux = zip(names, tensor_to_ndarray(aux))

        return insts, types, aux

    def _infer_img_batch(self, 
                         batch: Tuple[torch.Tensor], 
                         names: Tuple[str],
                         batch_loader: Iterable=None) -> Tuple[Iterable[Tuple[str, np.ndarray]]]:
        """
        Run inference on a batch of images that do not require tiling and stitching. I.e. For images 
        of the same size as the model input size (Pannuke).

        Args:
        --------
            batch (torch.Tensor):
                A batch of patches. Shape (B, C, patch_size, patch_size)
            names (Tuple[str]):
                filenames of the different images (without the suffices)
            batch_loader (Iterable, default=None):
                tqdm loader object

        Returns:
        --------
            Tuple of Zip (iterable) objects containing (name, np.ndarray) pairs
        """
        batch_instances, batch_types, batch_aux = self._predict_batch(batch)
        insts = zip(names, tensor_to_ndarray(batch_instances))

        types = zip(names, [None]*len(names))
        if batch_types is not None:
            types = zip(names, tensor_to_ndarray(batch_types))

        aux = zip(names, [None]*len(names))
        if batch_aux is not None:
            aux = zip(names, tensor_to_ndarray(batch_aux))
        
        self.n_batches_inferred += batch.shape[0]
        if batch_loader is not None:
            batch_loader.set_postfix(patches=f"{self.n_batches_inferred}/{len(self.folderset.fnames)}")

        return insts, types, aux

    def _post_proc(self, 
                   inst_probs: Dict[str, np.ndarray],
                   aux_maps: Dict[str, np.ndarray],
                   type_probs: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Takes in ordered dicts of name, mask pairs and runs post-processing on those

        Args:
        --------
            inst_probs (Dict[str, np.ndarray]):
                inst branch soft mask.
            aux_maps (Dict[str, np.ndarray]):
                aux branch mask
            type_probs (Dict[str, np.ndarray]):
                type branch soft mask 

        Returns:
        --------
            List[Tuple[str, np.ndarray, np.ndarray]]. A list of tuples containing 
            filename and masks per image e.g. ("filename1", inst_map: np.ndarray, type_map: np.ndarray) 
        """ 
        maps = self.post_processor.run_post_processing(
            inst_probs=inst_probs, 
            aux_maps=aux_maps, 
            type_probs=type_probs
        )

        return maps

    def _chunks(self, iterable: Iterable, size: int):
        """
        Generate adjacent chunks of an iterable 
        
        This is used to chunk the folder dataset for lower memory footprint
        
        Args:
        ---------
            iterable (Iterable):
                Input iterable (FolderDataset)
            size (int):
                size of one chunk. 
        """
        it = iter(iterable)
        return iter(lambda: tuple(itertools.islice(it, size)), ())

    def _infer(self, chunked_dataloader: Iterable) -> None:
        """
        Run inference on input images.

        Args:
        ---------
            chunked_dataloader (Iterable, default=None):
                If there is a lot of images in the folder, it's a good idea to chunk the folder dataloader
                to not overflow the Inferer instances memory. This argument is used in the 
        """
        # Start pipeline
        self.n_batches_inferred = 0
        soft_instances = []
        soft_types = []
        aux_maps = []

        with tqdm(chunked_dataloader, unit="batch") as batch_loader:
            for j, data in enumerate(batch_loader):
                
                # Get data
                batch = data["im"]
                names = data["file"]

                batch_loader.set_description(f"Running inference for {names}")

                # Set patching flag (most datasets require patching), Assumes square patches
                requires_patching = True if batch.shape[-1] > self.patch_size[0] else False

                if requires_patching:
                    inst_probs, type_probs, aux = self._infer_large_img_batch(batch, names, batch_loader)
                else:
                    inst_probs, type_probs, aux = self._infer_img_batch(batch, names, batch_loader)

                soft_instances.extend(inst_probs)
                soft_types.extend(type_probs)
                aux_maps.extend(aux)

        # save intermediate results to mem if save_dir not specified
        self.soft_insts = OrderedDict(soft_instances)
        self.soft_types = OrderedDict(soft_types)
        self.aux_maps = OrderedDict(aux_maps)


    def _post_process(self):
        """
        Run the post processing pipeline
        """
        assert "soft_insts" in self.__dict__.keys(), "No predictions found, run inference first."
        maps = self._post_proc(
            inst_probs=self.soft_insts, 
            aux_maps=self.aux_maps, 
            type_probs=self.soft_types
        )

        # save to containers
        self.inst_maps = OrderedDict()
        self.type_maps = OrderedDict()
        for res in maps:
            name = res[0]
            self.inst_maps[name] = res[1].astype("int32")
            self.type_maps[name] = res[2].astype("int32")

    def run_inference(self, 
                      save_dir: Union[Path, str]=None, 
                      fformat: str="geojson",
                      offsets: bool=False) -> None:
        """
        Run inference and post processing in chunks

        self.n_images is the size of one chunk of image files. After the chunk is finished
        the containers of the masks and intermediate arrays of the Inferer instance are cleared 
        for lower memory footprint. (Enables inference for larger sets of images).

        Args:
        ---------
            save_dir (Path or str, default=None):
                directory where the .mat/geojson files are saved
            fformat (str, default="geojson")
                file format for the masks. One of (".mat, "geojson")
            offsets (bool, default=False):
                If True, geojson coordnates are shifted by the offsets that are
                encoded in the file/samplenames (e.g. "x-1000_y-4000")
        """
        n_images_real = int(np.ceil(self.n_images / self.loader_batch_size))
        n_chunks = int(np.ceil(len(self.folderset.fnames) / self.n_images))
        loader = self._chunks(iterable=self.dataloader, size=n_images_real)

        with torch.no_grad():
            for i in range(n_chunks):
                self._infer(next(loader))
                self._post_process()

                # save results to files
                if save_dir is not None:
                    for name, inst_map in self.inst_maps.items():
                        if fformat == "geojson":
                            
                            # parse the offset coords from the name/key of the inst_map
                            x_off, y_off = (int(c) for c in re.findall(r"\d+", name)) if offsets else (0, 0)

                            mask2geojson(
                                inst_map=inst_map, 
                                type_map=self.type_maps[name], 
                                fname=name,
                                save_dir=save_dir,
                                x_offset=x_off,
                                y_offset=y_off
                            )

                        elif fformat == ".mat":
                            mask2mat(
                                inst_map=inst_map.astype("int32"),
                                type_map=self.type_maps[name].astype("int32"),
                                fname=name,
                                save_dir=save_dir
                            )

                    # clear memory
                    self.soft_insts.clear()
                    self.soft_types.clear()
                    self.aux_maps.clear()
                    self.inst_maps.clear()
                    self.type_maps.clear()
                    torch.cuda.empty_cache()

    def benchmark_insts(self, pattern_list: List[str]=None, file_prefix:str=""):
        """
        Run benchmarikng metrics for only instance maps 
        """
        assert "inst_maps" in self.__dict__.keys(), "No instance maps found, run inference and post proc first."
        assert self.gt_mask_dir is not None, f"gt_mask_dir is None. Benchmarking only with consep, kumar, pannuke"

        gt_masks = OrderedDict(
            [(f.name[:-4], FileHandler.read_mask(f, "inst_map")) for f in self.gt_mask_paths]
        )

        bm = Benchmarker()
        scores = bm.benchmark_insts(
            inst_maps=self.inst_maps,
            gt_masks=gt_masks,
            pattern_list=pattern_list,
            save_dir=self.model.fm.experiment_dir,
            prefix=file_prefix
        )
        return scores


    def benchmark_types(self, pattern_list: List[str]=None, file_prefix:str=""):
        """
        Run benchmarking for type maps
        """
        assert "inst_maps" in self.__dict__.keys(), "No instance maps found, run inference and post proc first."
        assert "type_maps" in self.__dict__.keys(), "No type maps found, run inference and post proc first."
        assert self.gt_mask_dir is not None, f"gt_mask_dir is None. Benchmarking only with consep, kumar, pannuke"
        assert self.model.decoder_type_branch, "the netowork model does not contain type branch"
        assert self.dataset == self.model.train_dataset, (
            "benchmarking per type can be done only for the same data set as the model training set",
            f"Given dataset for the inferer is not the training set: {self.dataset} != {self.model.train_dataset}"
        )

        gt_mask_insts = OrderedDict(
            [(f.name[:-4], FileHandler.read_mask(f, "inst_map")) for f in self.gt_mask_paths]
        )
        gt_mask_types = OrderedDict(
            [(f.name[:-4], FileHandler.read_mask(f, "type_map")) for f in self.gt_mask_paths]
        )

        bm = Benchmarker()
        scores = bm.benchmark_per_type(
            inst_maps=self.inst_maps, 
            type_maps=self.type_maps, 
            gt_mask_insts=gt_mask_insts, 
            gt_mask_types=gt_mask_types,
            pattern_list=pattern_list,
            classes=self.model.fm.get_classes(self.model.train_dataset), 
            save_dir=self.model.fm.experiment_dir,
            prefix=file_prefix
        )

        return scores
