import re
import torch
import itertools
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from copy import deepcopy
from torch.utils.data import DataLoader
from typing import Callable, Tuple, List, Iterable, Dict, Union, Optional, Iterable
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from src.utils import FileHandler, mask2geojson, mask2mat, label_sem_map
from src.patching import TilerStitcherTorch
from src.metrics import Benchmarker
from src.dl.utils import tensor_to_ndarray
from .post_processing.utils import post_processor
from .predictor import Predictor
from .folder_dataset import FolderDataset


__all__ = ["Inferer"]


class Inferer(FileHandler):
    def __init__(
        self,
        model: pl.LightningModule,
        in_data_dir: str,
        branch_weights: Dict[str, bool],
        branch_acts: Dict[str, str],
        post_proc_method: str,
        gt_mask_dir: str=None,
        tta: bool=False,
        model_weights: int=-1,
        loader_batch_size: int=1,
        loader_num_workers: int=1,
        patch_size: Tuple[int, int]=(256, 256),
        stride_size: int=128,
        model_batch_size: int=None,
        thresh_method: str="naive",
        thresh: float=0.5,
        n_images: int=None,
        fn_pattern: str="*",
        xmax: Optional[int]=None,
        ymax: Optional[int]=None,
        auto_range: Optional[bool]=False,
        device: str="cuda",
        test_mode: bool=False,
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
                benchmarking only. The GT-masks need to be in .mat files
            branch_weights (bool, default=Dict[str, bool]):
                Dictionary of branch names mapped to a boolean value.
                If the value is True, after a prediction, a weight
                matrix is applied that assigns bigger weight on pixels
                in center and less weight to pixels on prediction
                boundaries. helps dealing with prediction artefacts on
                tile/patch boundaries.
            branch_acts (bool, default=Dict[str, str]):
                Dictionary of branch names mapped to a str value that
                specifies the activation function applied for that
                branch. Allowed values: "softmax", "sigmoid", None
            post_proc_method (str, default=None):
                Defines the post-processing pipeline.The post-processing
                method is always specific to the auxiliary maps of the
                model. E.g. model.dec_branches == "hover", then the
                HoVer-Net or CellPose pipelines can be used. One of:
                "hover", "cellpose", "drfns", "dcan", "dran". 
            TODO: tta (bool, default=False):
                If True, performs test time augmentation. Inference time
                goes up with often marginal performance improvements.
            model_weights (int, default=-1):
                The epoch number of the saved checkpoint. If -1, uses
                the last epoch
            loader_batch_size (int, default=1):
                Number of images loaded from the input folder by the 
                workers per dataloader iteration. This is the DataLoader
                batch size, NOT the batch size that is used during the 
                forward pass of the model.
            loader_num_workers (int, default=1):
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
            n_images (int, default=None):
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
                Automatically filter tiles from a folder to contain 
                only ONE tissue section rather than every redundant 
                tissue section in the wsi. The tiles in the folder need 
                to contain the x- and y-coords (in xy- order) in the 
                filename. Example tile filename: "x-45000_y-50000.png".
            device (str, default="cuda"):
                The device of the input and model. One of: "cuda", "cpu"
            test_mode (bool, default=False):
                Flag for test runs
        """
        if not isinstance(model, pl.LightningModule): 
            raise ValueError(f"""
                Input model needs to be pl.LightningModule. Got: {type(model)}.
                Use the SegExperiment class to construct the model.
                """
            )
        if not stride_size <= patch_size[0]: 
            raise ValueError(f"""
                `stride_size` needs to be less or equal to `patch_size`.
                Got: stride_size: {stride_size} and patch_size {patch_size}.
                """
            )

        # set model
        self.model = model
        
        # device
        if device == "cpu":
            self.model.cpu()
        if torch.cuda.is_available() and device == "cuda":
            self.model.cuda()
        
        self.model.eval()
        self.device = self.model.device

        # Load trained weights for the model 
        self.exp_name = self.model.hparams["experiment_name"]
        self.exp_version = self.model.hparams["experiment_version"]
        
        if not test_mode:
            ckpt_path = self.get_model_checkpoint(
                experiment=self.exp_name,
                version=self.exp_version,
                which=model_weights
            )
            checkpoint = torch.load(
                ckpt_path, map_location = lambda storage, loc: storage
            )
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print(f"Using weights: {ckpt_path}")

        # Set input data folder
        self.in_data_dir = in_data_dir
        
        # Set num images inferred before clearing mem (chunk size)
        # By default there is no chunking.
        self.n_images = len(list(Path(self.in_data_dir).iterdir()))
        if n_images is not None:
            self.n_images = n_images

        # set gt mask folder
        self.gt_mask_dir = None
        if gt_mask_dir:
            gt_mask_dir = Path(gt_mask_dir)
            if not gt_mask_dir.exists():
                raise ValueError(f"Gt mask dir: {gt_mask_dir} not found")
            
            self.gt_mask_dir = sorted(gt_mask_dir.glob(fn_pattern))

        # Batch and tiling sizes
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.model_batch_size = model_batch_size
        self.loader_batch_size = loader_batch_size

        # Set dataset & dataloader.
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
        self.predictor = Predictor(self.model, self.patch_size)

        # set the post-processing pipeline. Defaults to 
        # model.aux_type if model has an auxiliary branch
        self.post_proc_method = post_proc_method
        self._validate_postproc_method()
        
        # init the post-processor
        self.post_processor = post_processor(
            self.post_proc_method,
            thresh_method=thresh_method,
            thresh=thresh
        )

        # input norm flag and train data stats
        self.norm = self.model.hparams["normalize_input"]

        # self.stats = self.get_dataset_stats(
        #   self.model.train_data.as_posix()
        # )
        
        # set up tha branch args
        self.branch_acts = branch_acts
        self.branch_weights = branch_weights
        self._validate_branch_args()
        
        self.branch_args = {
            f"{k}_map": {"act": a, "apply_weights": w}
            for (k, a), w in zip(branch_acts.items(), branch_weights.values())
        }
           
    def _get_map_dict(self, val: str=None):
        return {
            f"{k}_map": deepcopy(val)
            for k in self.model.hparams["dec_branches"].keys()
        }
        
    def _validate_branch_args(self) -> None:
        """
        Check that the branch args match in the model and inferer
        """
        mbranches = sorted(self.model.hparams["dec_branches"].keys())
        abranches = sorted(self.branch_acts.keys())
        wbranches = sorted(self.branch_weights.keys())
        
        if not mbranches == abranches == wbranches:
            raise ValueError(f"""
                Got mismatching keys for branch dict args. Model decoder
                branches: {mbranches}. `branch_weights`: {wbranches}.
                `branch_acts`: {abranches}."""
            )
        
        
    def _validate_postproc_method(self) -> None:
        """
        Check that post proc method can be used with current settings
        """
        allowed = ("basic")
        if "aux" in self.model.hparams["dec_branches"].keys():
            if self.model.hparams["dataset_type"] == "hover":
                allowed = ("hover", "cellpose", "basic")
            elif self.model.hparams["dataset_type"] == "dist":
                allowed = ("drfns", "basic")
            elif self.model.hparams["dataset_type"] == "contour":
                allowed = ("dcan", "dran", "basic")
            
        if not self.post_proc_method in allowed:
            raise ValueError(f"""
                Illegal arg: `post_proc_method`. Got: {self.post_proc_method}.
                Allowed methods: {allowed}. The allowed methods depend on
                the dataset type the model was trained with. Check the 
                `dataset_type` argument of the SegExperiment class.              
                """
            )

    def _apply(
            self,
            var: Union[torch.Tensor, None],
            op: Callable,
            **kwargs
        ) -> Union[torch.Tensor, None]:
        """
        Applies the given torch operation `op` to the given variable 
        `var`. This exists to catch memory errors
        
        Basically, if some cumulative torch operation overflows the GPU 
        memory, this catches the error, detaches the input tensor from 
        gpu and continues executing the operation on the cpu side. If 
        the `var` is None or an empty list/string etc. then this returns
        None for convenience.

        Args:
        --------
            var: (torch.Tensor or None):
                The torch.Tensor or list of tensors that should be 
                detached and moved to cpu before applying operations
            op (Callable):
                the torch function/callable that causes the mem error

        Returns:
        --------
            torch.Tensor or None: the ouput tensor or None
        """
        
        
        try:
            var = op(var, **kwargs)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if isinstance(var, list):
                    new_var = []
                    for elem in var:
                        elem = elem.detach()
                        if elem.is_cuda:
                            elem = elem.cpu()
                        new_var.append(elem)
                elif isinstance(var, torch.Tensor):
                    var = var.detach()
                    if var.is_cuda:
                        var = var.cpu()
                    new_var = var

                var = op(new_var, **kwargs)
        except BaseException as e:
            print(f"{e}")
            
        
        return var

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
        ) -> Dict[str, torch.Tensor]:
        """
        Forward pass + classify. Handles missing branches in the model.

        Args:
        ---------
            batch (torch.Tensor):
                A batch of patches. Shape (B, C, patch_size, patch_size)

        Returns:
        ---------
            A Dict of names mapped to tensors containing the predictions
        """
        # TODO: tta        
        # pred = self.predictor.forward_pass(batch, norm=self.norm, mean=self.stats[0], std=self.stats[1])
        pred = self.predictor.forward_pass(batch, norm=self.norm)
        
        maps = self._get_map_dict()
        for k, pred_map in pred.items():
            map = self.predictor.classify(pred_map, **self.branch_args[k]).cpu()
            maps[k] = map
    
        return maps

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
        n_batches_inferred = 0
        n_patches_total = patches.shape[2]*self.loader_batch_size
        
        batch_maps = self._get_map_dict([])
        
        # model batch size
        batch_size = self.model.hparams["batch_size"]
        if self.model_batch_size is not None:
            batch_size = self.model_batch_size 

        # Loop the B in batched patches (B, C, n_patches, patch_h, patch_w)
        for j in range(patches.shape[0]):
            
            # Divide patches into batches and predict
            pred_patches = self._get_map_dict([])
            for batch in self._get_batch(patches[j, ...], batch_size):
                soft_patches = self._predict_batch(batch)
                
                for k, map in soft_patches.items():
                    pred_patches[k].append(map)

                n_batches_inferred += batch.shape[0]
                if batch_loader is not None:
                    batch_loader.set_postfix(
                        patches=f"{n_batches_inferred}/{n_patches_total}"
                    )

            # catch runtime error if preds take too much GPU mem and 
            # move to cpu with the _apply method.
            for k, pred_list in pred_patches.items():
                preds = self._apply(var=pred_list, op=torch.cat, dim=0) # (n_patches, C, pH, pW)
                batch_maps[k].append(preds)
        
        for k, preds in batch_maps.items():
            out_map = self._apply(preds, torch.stack, dim=0).permute(0, 2, 1, 3, 4) # (B, C, n_patches, pH, pW)
            out_map = self._apply(out_map, tilertorch.stitch_batched_patches) # (B, C, H, W)
            batch_maps[k] = zip(names, tensor_to_ndarray(out_map))
            
        return batch_maps

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
        soft_patches = self._apply(batch, self._predict_batch)
        
        batch_maps = self._get_map_dict([])
        for k, soft_mask in soft_patches.items():
            out_map = tuple(zip(names, tensor_to_ndarray(soft_mask)))
            batch_maps[k].extend([*out_map])
        
        return batch_maps
            
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
        self.soft_masks = self._get_map_dict([])

        with tqdm(chunked_dataloader, unit="batch") as loader:
            with torch.no_grad():
                for data in loader:
                    
                    # Get data
                    batch = data["im"].to(self.device).float()
                    names = data["file"]

                    loader.set_description(f"Running inference for {names}")

                    # Set patching flag (most datasets require patching), 
                    # Assumes square patches
                    requires_patching = False
                    if batch.shape[-1] > self.patch_size[0]:
                        requires_patching = True

                    # predict soft maps
                    if requires_patching:
                        soft_mask = self._infer_large_img_batch(batch, names, loader)
                    else:
                        soft_mask = self._infer_img_batch(batch, names, loader)

                    for k in self.soft_masks.keys():
                        self.soft_masks[k].extend([*soft_mask[k]])

        # save intermediate results
        for k, mask in self.soft_masks.items():
            self.soft_masks[k] = OrderedDict(mask)

    def _post_process(self):
        """
        Run the post processing pipeline
        """
        if "soft_masks" not in self.__dict__.keys():
            raise RuntimeError("""
                No soft masks found. Inference need to be run before
                post-processing.
                """
            )

        maps = self.post_processor.run_post_processing(self.soft_masks)
        
        self.out_maps = {
            k: deepcopy({})
            for k in maps[0].keys() if k is not "fn"
        }

        for out_dict in maps:
            name = out_dict["fn"]
            for k in self.out_maps.keys():
                if k in out_dict.keys():
                    map = out_dict[k]
                    self.out_maps[k][name] = map

    def run_inference(
            self, 
            save_dir: Union[Path, str]=None, 
            fformat: str=None,
            offsets: bool=False,
            classes_type: Dict[str, int]=None,
            classes_sem: Dict[str, int]=None,
        ) -> None:
        """
        Run inference and post processing in chunks

        Args:
        ---------
            save_dir (Path or str, default=None):
                directory where the .mat/geojson files are saved
            fformat (str, default="geojson")
                file format for the masks. One of ".mat, "geojson", None
            offsets (bool, default=False):
                If True, geojson coords are shifted by the offsets that 
                are encoded in the filenames (e.g. "x-1000_y-4000.png")
            classes_type (Dict[str, int], default=None):
                class dict for the cell types.
                e.g. {"inflam":1, "epithelial":2, "connec":3}
                This is required if masks are saved to geojson.
            classes_sem (Dict[str, int], default=None):
                class dict for the area types.
                e.g. {"inflam":1, "epithelial":2, "connec":3}
                This is required if masks are saved to geojson.
        """

        # checks before lengthy processing
        if save_dir is not None:
            save_dir = Path(save_dir)
            
            if not save_dir.exists():
                FileHandler.create_dir(save_dir)
            
            allowed = ("geojson", ".mat", None)
            if fformat not in allowed:
                raise ValueError(f"""
                    Illegal `fformat`. Got {fformat}. Allowed: {allowed}
                    """
                )

            if fformat == "geojson":
                if classes_type is None:
                    raise ValueError(f"""
                        `classes_type` is None. Cell type classes Dict
                        is needed for the geojson format.
                        """
                    )
                if "sem" in self.model.hparams["dec_branches"].keys():
                    if classes_sem is None: 
                        raise ValueError("""
                            `area_classes` is None. Area classes Dict is
                            needed for geojson format.
                            """
                        )

        n_images_real = int(np.ceil(self.n_images / self.loader_batch_size))
        n_chunks = int(np.ceil(len(self.folderset.fnames) / self.n_images))
        loader = self._chunks(iterable=self.dataloader, size=n_images_real)

        for _ in range(n_chunks):
            self._infer(next(loader))
            self._post_process()

            # save results to files
            if save_dir is not None:
                for name, inst_map in tqdm(self.out_maps["inst_map"].items(), desc="saving..."):
                    if fformat == "geojson":
                        
                        # parse the offset coords from the inst key
                        x_off, y_off = (
                            int(c) for c in re.findall(r"\d+", name)
                        ) if offsets else (0, 0)

                        if "type" in self.model.hparams["dec_branches"].keys():
                            type_map = self.out_maps["type_map"][name]
                            mask2geojson(
                                inst_map=inst_map, 
                                type_map=type_map, 
                                fname=f"{name}_cells",
                                save_dir=Path(save_dir / "cells"),
                                x_offset=x_off,
                                y_offset=y_off,
                                classes=classes_type
                            )

                        if "sem" in self.model.hparams["dec_branches"].keys():
                            sem_type = self.out_maps["sem_map"][name]
                            sem_inst = label_sem_map(sem_type)
                            mask2geojson(
                                inst_map=sem_inst, 
                                type_map=sem_type, 
                                fname=f"{name}_areas",
                                save_dir=Path(save_dir / "areas"),
                                x_offset=x_off,
                                y_offset=y_off,
                                classes=classes_sem
                            )

                    elif fformat == ".mat":
                        if "type" in self.model.hparams["dec_branches"].keys():
                            type_map = self.out_maps["type_map"][name]
                            mask2mat(
                                inst_map=inst_map,
                                type_map=type_map,
                                fname=f"{name}_cells",
                                save_dir=Path(save_dir / "cells")
                            )
                            
                        if "sem" in self.model.hparams["dec_branches"].keys():
                            sem_type = self.out_maps["sem_map"][name]
                            sem_inst = label_sem_map(sem_type)
                            mask2mat(
                                inst_map=sem_inst,
                                type_map=sem_type,
                                fname=f"{name}_areas",
                                save_dir=Path(save_dir / "areas")
                            )

                # clear memory                
                self.soft_masks.clear()
                self.out_maps.clear()
                torch.cuda.empty_cache()

    def benchmark_insts(
            self, 
            pattern_list: Optional[List[str]]=None, 
            file_prefix: Optional[str]=""
        ) -> pd.DataFrame:
        """
        Run benchmarikng metrics for instance seg results and save them 
        into a csv file in the `results`-dir. 

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
        if self.gt_mask_dir is None:
            raise ValueError("""
                gt_mask_dir is None. Can't run benchmarking without
                ground truth annotations
                """
            )
                    
        if "out_maps" not in self.__dict__.keys():
            raise RuntimeError("""
                No instance maps found, run the `run_inference` method first.
                """
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
            inst_maps=self.out_maps["inst_map"],
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
        the repositoy. This requires that the `gt_mask_dir` arg is given

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
        if self.gt_mask_dir is None:
            raise ValueError("""
                gt_mask_dir is None. Can't run benchmarking without
                ground truth annotations
                """
            )
                    
        if "out_maps" not in self.__dict__.keys():
            raise RuntimeError("""
                No instance maps found, run the `run_inference` method first.
                """
            )
            
        if "type_map" not in self.out_maps.keys():
            raise RuntimeError("""
                No type maps found in `out_maps`. The model needs to contain
                a type seg branch to run benchmarking per type.
                """
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
            inst_maps=self.out_maps["inst_map"], 
            type_maps=self.out_maps["type_map"], 
            gt_mask_insts=gt_mask_insts, 
            gt_mask_types=gt_mask_types,
            pattern_list=pattern_list,
            classes=classes, 
            save_dir=exp_dir,
            prefix=file_prefix
        )

        return scores
