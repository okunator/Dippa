import re
import torch
import itertools
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Iterable, Dict, Union, Optional, Iterable
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from src.utils import FileHandler, mask2geojson, mask2mat
from src.patching import TilerStitcherTorch
from src.metrics import Benchmarker
from src.dl.utils import tensor_to_ndarray
from .post_processing.processor_builder import PostProcBuilder
from .predictor import Predictor


SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")


class FolderDataset(Dataset, FileHandler):
    def __init__(
        self, 
        folder_path: Union[str, Path],
        pattern: Optional[str]="*", 
        sort_by_y: Optional[bool]=False,
        xmax: Optional[int]=None,
        ymax: Optional[int]=None,
        auto_range: bool=False,
        tile_size: Optional[Tuple[int, int]]=(1000, 1000)
    ) -> None:
        """
        Simple pytorch folder dataset. Assumes that
        folder_path contains only image files which are readable
        by cv2.

        Args:
        ----------
            folder_path (Union[str, Path]):
                path to the folder containig tile/image files
            pattern (str, optional, default="*"):
                file pattern for filtering only the files that contain 
                the pattern.
            sort_by_y (bool, optional, default=False):
                sorts a folder (containing tiles extracted by histoprep 
                package) by the y-coord rather than the x-coord
            xmax (int, optional, default=None):
                filters all the tile-files that contain x-coord less 
                or equal to this param in their filename. Works with 
                tiles extracted with histoprep. 
                See https://github.com/jopo666/HistoPrep 
            ymax (int, optional, default=None):
                filters all the tile-files that contain y-coord less 
                or equal to this param in their filename. Works with 
                tiles extracted with histoprep. 
                See https://github.com/jopo666/HistoPrep 
            auto_range (bool, default=False):
                Automatically filter tiles that contain ONE tissue 
                section rather than every redundant tissue section in 
                the wsi.
            tile_size (Tuple[int, int], optional, default=(1000, 1000)):
                size of the input tiles in the folder. Optional.
        """
        super(FolderDataset, self).__init__()
        self.tile_size = tile_size
        folder_path = Path(folder_path)
        assert folder_path.exists(), f"folder: {folder_path} does not exist"
        assert folder_path.is_dir(), f"path: {folder_path} is not a folder"
        assert all([f.suffix in SUFFIXES for f in folder_path.iterdir()]),(
            f"files formats in given folder need to be in {SUFFIXES}"
        )

        #  sort files
        if sort_by_y:
            self.fnames = sorted(
                folder_path.glob(pattern), 
                key=lambda x: self._get_xy_coords(x.name)[1]
            )
        else:
            self.fnames = sorted(folder_path.glob(pattern))

        # filter by xy-cooridnates encoded in the filename
        if xmax is not None:
            self.fnames = [
                f for f in self.fnames 
                if self._get_xy_coords(f.name)[0] <= xmax
            ]
        if ymax is not None and not auto_range:
            self.fnames = [
                f for f in self.fnames 
                if self._get_xy_coords(f.name)[1] <= ymax
            ]
        
        if auto_range:
            ymin, ymax = self._get_auto_range(coord="y") # only y-axis for now
            self.fnames = [
                f for f in self.fnames 
                if ymin <= self._get_xy_coords(f.name)[1] <= ymax
            ]

    def _get_xy_coords(self, fname: str) -> Tuple[int, int]:
        """
        Extract xy-coords from files named with x- and y- coordinates 
        in their file name.
        
        example filename: "sumthing_4955_x-47000_y-25000.png 
        """
        assert re.findall(r"(x-\d+_y-\d+)", fname), (
            "fname not in 'sumthing_x-[coord1]_y-[coord2]'-format",
            "Set auto_range to False if filenames are not in this format"
        )
        
        xy_str = re.findall(r"(x-\d+_y-\d+)", fname)
        xy = [int(c) for c in re.findall(r"\d+", xy_str[0])]

        return xy

    def _get_auto_range(
            self, 
            coord: str="y", 
            section_ix: int=0, 
            section_length: int=6000
        ) -> Tuple[int, int]:
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
                the nth tissue section in the wsi in the direction of 
                the coord param.
            section_length (int, default=6000):
                Threshold to concentrate only on tissue sections that
                are larger than 6000 pixels

        Returns:
        --------
            Tuple[int, int]: The start and end point of the tissue 
            section in the specified direction
        """
        ix = 1 if coord == "y" else 0
        coords = sorted(
            set([self._get_xy_coords(f.name)[ix] for f in self.fnames])
        )

        try:
            splits = []
            split = []
            for i in range(len(coords)-1):
                if coords[i + 1] - coords[i] == self.tile_size[ix]:
                    split.append(coords[i])
                else:
                    if i < len(coords) - 1:
                        split.append(coords[i]) 
                    splits.append(split)
                    split = []
            
            ret_splits = [
                split for split in splits 
                if len(split) >= section_length//self.tile_size[ix]
            ]
            ret_split = ret_splits[section_ix]
            return ret_split[0], ret_split[-1]
        except:
            # if there is only one tissue section, return min and max
            start = min(coords, key=lambda x: x[ix])[ix]
            end = max(coords, key=lambda x: x[ix])[ix]
            return start, end

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


class Inferer(FileHandler):
    def __init__(
        self,
        model: pl.LightningModule,
        in_data_dir: str,
        gt_mask_dir: str=None,
        tta: bool=False,
        model_weights: str="last",
        loader_batch_size: int=8,
        loader_num_workers: int=8,
        patch_size: Tuple[int, int]=(256, 256),
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
        **kwargs
    ) -> None:
        """
        Class to perform inference and post-processing

        Args:
        -----------
            model (pl.LightningModule):
                Input SegModel (lightning model)
            in_data_dir (str):
                This directory will be used as the input data directory. 
                Assumes that the directory contains only cv2 readable 
                image files: .png, .tif, etc
            gt_mask_dir (str, default=None):
                The directory of the test ground truth masks. Needed for 
                benchmarking only.
            tta (bool, default=False):
                If True, performs test time augmentation. Inference time
                goes up with often marginal performance improvements.
            model_weights (str, default="last"):
                pytorch lightning saves the weights of the model for the
                last epoch and best epoch (based on validation data). 
                One of ("best", "last").
            loader_batch_size (int, default=8):
                Number of images loaded from the input folder by the 
                workers per dataloader iteration. This is the DataLoader
                batch size, NOT the batch size that is used during the 
                forward pass of the model.
            loader_num_workers (int, default=8):
                Number of threads/workers for torch dataloader
            patch_size (Tuple[int, int], default=(256, 256)):
                The size of the input patches that are fed to the 
                segmentation model.
            stride_size (int, default=128):
                If input images are larger than the model input image 
                size (patch_size), the images are tiled with a sliding 
                window into small patches with overlap. This param is 
                the stride size used in the sliding window operation. 
                Small stride for the sliding window results in less 
                artefacts and checkerboard effect in the resulting 
                prediction when the patches are stitched back to the 
                input image size. On the other hand small stride_size 
                means more patches and larger number of patches leads to
                slower inference time and larger memory consumption. 
            model_batch_size (int, default=None):
                The batch size that is used when the input is fed to the
                model (actual model batch size). Use if the input images
                need patching and the batch size for training batch size
                is too large i.e. you get (cuda out of memmory error).
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance
                branch.One of ("naive", "argmax", "sauvola", "niblack").
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method="naive"
            apply_weights (bool, default=True):
                After a prediction, apply a weight matrix that assigns 
                bigger weight on pixels in center and less weight to 
                pixels on prediction boundaries. helps dealing with 
                prediction artefacts on tile/patch boundaries. NOTE: 
                This is only applied at the auxiliary branch prediction 
                since there tiling effect has the most impact. 
                (Especially, in HoVer-maps)
            post_proc_method (str, default=None):
                Defines the post-processing pipeline. If this is None, 
                then the post-processing pipeline is defined by the 
                aux_type of the model. If the aux_type of the model is 
                None, then the basic watershed post-processing pipeline
                is used. The post-processing method is always specific 
                to the auxiliary maps that the model outputs so if the 
                aux_type == "hover", then the HoVer-Net and CellPose 
                pipelines can be used. One of: None, "hover","cellpose",
                "drfns", "dcan", "dran". 
            n_images (int, default=32):
                Number of images inferred before clearing the memory. 
                Useful if there is a large number of images in a folder.
                The segmentation results are saved after n_images are 
                segmented and memory cleared for a new set of images.
            fn_pattern (str, default="**):
                A pattern in file names in the in_data_dir. For example,
                for pannuke dataset you can run inference for only 
                images of specific tissue e.g. pattern = *_Adrenal_*.
            xmax (int, optional, default=None):
                Filters all the file names in the input directory that 
                contain x-coordinate less than this param in their 
                filename. I.e. the tiles in the folder need to contain 
                the x- and y- coordinates (in xy- order) in the filename
                Example tile filename: "x-45000_y-50000.png".
            ymax (int, optional, default=None):
                Filters all the file names in the input directory that 
                contain y-coord less than this param in their filename. 
                I.e. the tiles in the folder need to contain the x- and 
                y- coords (in xy- order) in the filename. Example tile 
                filename: "x-45000_y-50000.png".
            auto_range (bool, optional, default=False):
                Automatically filter tiles from a folder that contain 
                only ONE tissue section rather than every redundant 
                tissue section in the wsi. The tiles in the folder need 
                to contain the x- and y-coords (in xy- order) in the 
                filename. Example tile filename: "x-45000_y-50000.png".
        """
        assert isinstance(model, pl.LightningModule), (
            "Input model needs to be a lightning model"
        )
        assert stride_size <= patch_size[0], (
            f"stride_size: {stride_size} > {patch_size[0]}"
        )
        assert model_weights in ("best", "last")

        # set model to device and to inference mode
        self.model = model
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()
        torch.no_grad()

        # Load trained weights for the model 
        self.exp_name = self.model.experiment_name
        self.exp_version = self.model.experiment_version
        ckpt_path = self.get_model_checkpoint(
            experiment=self.exp_name,
            version=self.exp_version,
            which=model_weights
        )
        checkpoint = torch.load(
            ckpt_path, map_location = lambda storage, loc: storage
        )
        self.model.load_state_dict(
            checkpoint['state_dict'], 
            strict=False
        )

        self.patch_size = patch_size
        self.stride_size = stride_size
        self.n_images = n_images

        # Set input data folder
        self.in_data_dir = in_data_dir
        
        # set gt mask folder
        self.gt_mask_dir = None
        if gt_mask_dir:
            self.gt_mask_dir = sorted(
                Path(gt_mask_dir).glob(fn_pattern)
            )

        # Batch sizes
        self.model_batch_size = model_batch_size
        self.loader_batch_size = loader_batch_size

        # Set dataset dataloader
        self.folderset = FolderDataset(
            self.in_data_dir, 
            pattern=fn_pattern, 
            xmax=xmax, 
            ymax=ymax, 
            auto_range=auto_range
        )

        self.dataloader = DataLoader(
            self.folderset, 
            batch_size=loader_batch_size, 
            shuffle=False, pin_memory=True, 
            num_workers=loader_num_workers
        )

        # set apply weights flag for aux branch and prdeictor class
        self.apply_weights = apply_weights
        self.predictor = Predictor(self.model, self.patch_size)

        # set the post-processing pipeline. Defaults to 
        # model.aux_type if model has an auxiliary branch
        self.post_proc_method = post_proc_method
        if self.post_proc_method is None:
            self.post_proc_method = "basic"
            if self.model.aux_branch:
                self.post_proc_method = self.model.aux_type

        # Quick checks that a valid post-proc-method is used
        msg = (
            "post_proc_method does not match to model config. ", 
            f"set to: {self.post_proc_method} while the model ",
            f"decoder_aux_branch is: {self.model.decoder_aux_branch}"
        )

        if self.model.decoder_aux_branch:
            if self.model.decoder_aux_branch == "hover":
                allowed = ("hover", "cellpose", "basic")
            elif self.model.decoder_aux_branch == "dist":
                allowed = ("drfns", "basic")
            elif self.model.decoder_aux_branch == "contour":
                allowed = ("dcan", "dran", "basic")

            assert self.post_proc_method in allowed, msg
        
        # init the post-processor
        self.post_processor = PostProcBuilder.set_postprocessor(
            post_proc_method=self.post_proc_method,
            thresh_method=thresh_method,
            thresh=thresh
        )

        # input norm flag and train data stats
        self.norm = self.model.normalize_input
        # self.stats = self.get_dataset_stats(
        #   self.model.train_data.as_posix()
        # )

    def _get_batch(
            self, 
            patches: torch.Tensor, 
            batch_size: int
        ) -> torch.Tensor:
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
            batch = patches[:, i:i+batch_size, ...].permute(1, 0, 2, 3)
            yield batch 

    def _predict_batch(
            self, 
            batch: torch.Tensor
        ) -> Tuple[Union[torch.Tensor, None]]:
        """
        Forward pass + classify. Handles missing branches in the model.

        Args:
        ---------
            batch (torch.Tensor):
                A batch of patches. Shape (B, C, patch_size, patch_size)

        Returns:
        ---------
            A tuple of tensors containing the predictions. If network 
            does no contain aux or type branch the predictions are None
        """
        # TODO: tta
        # pred = self.predictor.forward_pass(batch, norm=self.norm, mean=self.stats[0], std=self.stats[1])
        pred = self.predictor.forward_pass(batch, norm=self.norm)
        insts = self.predictor.classify(pred["instances"], act="softmax")

        types = None
        if pred["types"] is not None:
            types = self.predictor.classify(pred["types"], act="softmax")

        aux = None
        if pred["aux"] is not None:
            aux = self.predictor.classify(
                pred["aux"], act=None, apply_weights=self.apply_weights
            )

        return insts, types, aux

    def _infer_large_img_batch(
            self, 
            batch: torch.Tensor, 
            names: Tuple[str],
            batch_loader: Iterable=None,
        ) -> Tuple[Iterable[Tuple[str, np.ndarray]]]:
        """
        Run inference on large images that require tiling and back 
        stitching. I.e. For images larger than the model input size.

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
            Tuple: of Zip objects containing (name, np.ndarray) pairs
        """

        # Tile the image into patches
        tilertorch = TilerStitcherTorch(
            batch.shape,
            self.patch_size,
            self.stride_size,
            padding=True
        )
        patches = tilertorch.extract_patches_from_batched(batch)

        # (for tqdm logging)
        n_patches_total = patches.shape[2]*(len(batch_loader)*self.loader_batch_size)
        
        batch_instances = []
        batch_types = []
        batch_aux = []

        # model batch size
        batch_size = self.model.batch_size
        if self.model_batch_size is not None:
            batch_size = self.model_batch_size 

        # Loop the B in batched patches (B, C, n_patches, patch_h, patch_w)
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
                    batch_loader.set_postfix(
                        patches=f"{self.n_batches_inferred}/{n_patches_total}"
                    )
                
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

    def _infer_img_batch(
            self, 
            batch: Tuple[torch.Tensor], 
            names: Tuple[str],
            batch_loader: Iterable=None
        ) -> Tuple[Iterable[Tuple[str, np.ndarray]]]:
        """
        Run inference on a batch of images that do not require tiling 
        and stitching. I.e. For images of the same size as the model 
        input size.

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
            Tuple: of Zip objects containing (name, np.ndarray) pairs
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
            batch_loader.set_postfix(
                patches=f"{self.n_batches_inferred}/{len(self.folderset.fnames)}"
            )

        return insts, types, aux

    def _chunks(self, iterable: Iterable, size: int) -> Iterable:
        """
        Generate adjacent chunks of an iterable 
        
        This is used to chunk the folder dataset for lower mem footprint
        
        Args:
        ---------
            iterable (Iterable):
                Input iterable (FolderDataset)
            size (int):
                size of one chunk.

        Returns:
        ---------
            Iterable chunk of filenames
        """
        it = iter(iterable)
        return iter(lambda: tuple(itertools.islice(it, size)), ())

    def _infer(self, chunked_dataloader: Iterable) -> None:
        """
        Run inference on input images.

        Args:
        ---------
            chunked_dataloader (Iterable, default=None):
                A chunked dataloader object
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

                # Set patching flag (most datasets require patching), 
                # Assumes square patches
                requires_patching = False
                if batch.shape[-1] > self.patch_size[0]:
                    requires_patching = True

                if requires_patching:
                    inst_probs, type_probs, aux = self._infer_large_img_batch(
                        batch, names, batch_loader
                    )
                else:
                    inst_probs, type_probs, aux = self._infer_img_batch(
                        batch, names, batch_loader
                    )

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
        assert "soft_insts" in self.__dict__.keys(), (
            "No predictions found, run inference first."
        )

        maps = self.post_processor.run_post_processing(
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

    def run_inference(
            self, 
            save_dir: Union[Path, str]=None, 
            fformat: str="geojson",
            offsets: bool=False
        ) -> None:
        """
        Run inference and post processing in chunks

        Args:
        ---------
            save_dir (Path or str, default=None):
                directory where the .mat/geojson files are saved
            fformat (str, default="geojson")
                file format for the masks. One of ".mat, "geojson"
            offsets (bool, default=False):
                If True, geojson coords are shifted by the offsets that 
                are encoded in the filenames (e.g. "x-1000_y-4000.png")
        """
        n_images_real = int(np.ceil(self.n_images / self.loader_batch_size))
        n_chunks = int(np.ceil(len(self.folderset.fnames) / self.n_images))
        loader = self._chunks(iterable=self.dataloader, size=n_images_real)

        with torch.no_grad():
            for _ in range(n_chunks):
                self._infer(next(loader))
                self._post_process()

                # save results to files
                if save_dir is not None:
                    for name, inst_map in self.inst_maps.items():
                        if fformat == "geojson":
                            
                            # parse the offset coords from the inst key
                            x_off, y_off = (
                                int(c) for c in re.findall(r"\d+", name)
                            ) if offsets else (0, 0)

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

    def benchmark_insts(
            self, 
            pattern_list: Optional[List[str]]=None, 
            file_prefix: Optional[str]=""
        ) -> pd.DataFrame:
        """
        Run benchmarikng metrics for only instance maps and save them 
        into a csv file. The file is written into the "results"
        directory of the repositoy.

        Args:
        ---------
            pattern_list (List[str], optional, default=None):
                A list of string patterns used for filtering files in 
                the input data folder
            file_prefix (str, optional, default=""):
                prefix to give to the csv filename that contains the 
                benchmarking results

        Returns:
        ----------
            pd.DataFrame: a df containing the benchmarking results
        """
        assert "inst_maps" in self.__dict__.keys(), (
            "No instance maps found, run inference first."
        )

        gt_masks = OrderedDict(
            [
                (f.name[:-4], self.read_mask(f, "inst_map")) 
                for f in self.gt_mask_dir
            ]
        )

        exp_dir = self.get_experiment_dir(self.exp_name, self.exp_version)
        bm = Benchmarker()
        scores = bm.benchmark_insts(
            inst_maps=self.inst_maps,
            gt_masks=gt_masks,
            pattern_list=pattern_list,
            save_dir=exp_dir,
            prefix=file_prefix
        )
        return scores


    def benchmark_types(
            self,
            classes: Dict[str, int],
            pattern_list: Optional[List[str]]=None,
            file_prefix: Optional[str]=""
        ) -> pd.DataFrame:
        """
        Run benchmarking for inst_maps & type maps and save them into a 
        csv file. The file is written into the "results" directory of 
        the repositoy

        Args:
        ---------
            classes (Dict[str, int]):
                The class dict e.g. {bg: 0, immune: 1, epithel: 2}. 
                background must be the 0 class
            pattern_list (List[str], optional, default=None):
                A list of string patterns used for filtering files in 
                the input data folder
            file_prefix (str, optional, default=""):
                prefix to give to the csv filename that contains the 
                benchmarking results

        Returns:
        ----------
            pd.DataFrame: A df containing the benchmarking results
        """
        assert "inst_maps" in self.__dict__.keys(), (
            "No instance maps found, run inference first"
        )
        assert "type_maps" in self.__dict__.keys(), (
            "No type maps found, run inference first."
        )
        assert self.model.decoder_type_branch, (
            "the network model does not contain type branch"
        )

        gt_mask_insts = OrderedDict(
            [
                (f.name[:-4], FileHandler.read_mask(f, "inst_map")) 
                for f in self.gt_mask_dir
            ]
        )
        gt_mask_types = OrderedDict(
            [
                (f.name[:-4], FileHandler.read_mask(f, "type_map")) 
                for f in self.gt_mask_dir
            ]
        )

        exp_dir = self.get_experiment_dir(self.exp_name, self.exp_version)
        bm = Benchmarker()
        scores = bm.benchmark_per_type(
            inst_maps=self.inst_maps, 
            type_maps=self.type_maps, 
            gt_mask_insts=gt_mask_insts, 
            gt_mask_types=gt_mask_types,
            pattern_list=pattern_list,
            classes=classes, 
            save_dir=exp_dir,
            prefix=file_prefix
        )

        return scores
