import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Union, Optional
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from src.utils.file_manager import FileHandler
from src.patching import TilerStitcherTorch
from src.metrics.benchmarker import Benchmarker
from src.dl.torch_utils import tensor_to_ndarray

from .post_processing.processor_builder import PostProcBuilder
from .predictor import Predictor


SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")


class FolderDataset(Dataset, FileHandler):
    def __init__(self, folder_path: Union[str, Path], pattern: str="*"):
        """
        Simple pytorch folder dataset. Assumes that
        folder_path contains only image files which are readable
        by cv2.

        Args:
        ----------
            folder_path (Union[str, Path]):
                path to the folder containig image files
            pattern (str, default="*"):
                file pattern for getting only files that contain the pattern.
        """
        super(FolderDataset, self).__init__()
        folder_path = Path(folder_path)
        assert folder_path.exists(), f"folder: {folder_path} does not exist"
        assert folder_path.is_dir(), f"given path: {folder_path} is not a folder"
        assert all([f.suffix in SUFFIXES for f in folder_path.iterdir()]) ,(
            f"files formats in given folder need to be in {SUFFIXES}"
        )
        self.fnames = sorted(folder_path.glob(pattern))

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
                 in_data_dir: Optional[str]=None,
                 dataset: Optional[str]=None,
                 data_fold: str="test",
                 tta: bool=False,
                 model_weights: str="last",
                 fn_pattern: str="*",
                 num_workers: int=8,
                 batch_size: int=8,
                 patch_size: Tuple[int]=(256, 256),
                 stride_size: int=128,
                 thresh_method: int="naive",
                 thresh: float=0.5,
                 apply_weights: bool=False,
                 post_proc_method: str=None,
                 **kwargs) -> None:
        """
        Class to perform inference and post-processing

        Args:
        -----------
            model (pl.LightningModule):
                Input SegModel (lightning model) specified in lightning_model.py.
            in_data_dir (str, optional, default=None):
                If not None this directory will be used as the input data directory.
                Assumes that the directory contains only cv2 readable image files (.png, .tif, etc).
                This argument overrides all other dataset related argument.
            dataset (str, optional, default=None):
                One of ("kumar","consep","pannuke","dsb2018", "monusac", None)
                If data_dir == None, images from this dataset will be used for inference. 
                If both dataset == None & data_dir == None. The inference is performed
                on the same dataset that the input model was trained with.
            data_fold (str, default="test"):
                Which fold of data to run inference. One of ("train", "test"). 
                If data_dir is set this arg will be ignored.
            tta (bool, default=False):
                If True, performs test time augmentation. Inference time goes up
                with often marginal performance improvements.
            model_weights (str, default="last"):
                pytorch lightning saves the weights of the model for the last epoch
                and best epoch (based on validation data). One of ("best", "last").
            fn_pattern (str, default="**):
                A pattern in file names. For example, in pannuke dataset you can run 
                inference for only images of specific tissue e.g. pattern = *_Adrenal_gland_*.
            num_workers (int, default=8):
                Number of thread workers for torch dataloader
            batch_size (int, default=8):
                Number of images loaded from the input folder by the workers per dataloader
                iteration. This is not the batch size that is used during the forward pass
                of the model.
            patch_size (Tuple[int], default=(256, 256)):
                The size of the input patches.
            stride_size (int, default=128):
                If input images are larger than the model input image size, the images are tiled
                with a sliding window into small patches with overlap. This param is the stride size 
                used in the sliding window operation. Small stride for the sliding window results in 
                less artefacts and checkerboard effect in the resulting prediction when the patches are 
                stitched back to the input image size. On the other hand small stride_size means more
                patches and larger number of patches -> slower inference time and larger memory 
                consumption. stride_size needs to be less or equal than the input image size.
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"
            apply_weights (bool, default=True):
                After a prediction, apply a weight matrix that assigns bigger weight on pixels
                in center and less weight to pixels on prediction boundaries. helps dealing with
                prediction artifacts on tile/patch boundaries. NOTE: This is only applied at the
                auxiliary branch prediction since there tiling effect has the most impact.
                (Especially, in HoVer-maps)
            post_proc_method (str, default=None):
                Defines the post-processing pipeline. If this is None, then the post-processing
                pipeline is defined by the aux_type of the model. If the aux_type of the model
                is None, then the basic watershed post-processing pipeline is used. If the
                aux_type == "hover", then the HoVer-Net and CellPose pipelines can be used.
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
        self.folderset = FolderDataset(self.in_data_dir, pattern=fn_pattern)
        self.dataloader = DataLoader(self.folderset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

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
        insts = self.predictor.classify(pred["instances"], act="softmax") # goes to cpu
        types = self.predictor.classify(pred["types"], act="softmax") if pred["types"] is not None else None
        aux = self.predictor.classify(pred["aux"], act=None, apply_weights=self.apply_weights) if pred["aux"] is not None else None
        return insts, types, aux

    # TODO: modularize
    def run_inference(self) -> None:
        """
        Run inference on the images in the input folder. 
        Results will be saved in OrderedDicts:

        - self.res_insts: instance branch predicitons
        - self.res_types: type branch predictions
        - self.res_aux: aux branch predictions
        """

        # Start pipeline
        soft_instances = []
        soft_types = []
        aux_maps = []
        
        # running int for tqdm logging
        running_int = 0

        with tqdm(self.dataloader, unit="batch") as batch_loader:
            for data in batch_loader:
                batch_loader.set_description(f"Inference: {self.in_data_dir}")

                # Get data
                batch = data["im"]
                fnames = data["file"]

                # Set patching flag (most datasets require patching), Assumes square patches
                requires_patching = True if batch.shape[-1] > self.patch_size[0] else False
                if requires_patching:

                    # Do patching if images bigger than model input size
                    tilertorch = TilerStitcherTorch(batch.shape, self.patch_size, self.stride_size, padding=True)
                    patches = tilertorch.extract_patches_from_batched(batch)
                    
                    # tqdm logging
                    n_patches_total = patches.shape[2]*len(self.folderset.fnames)

                    batch_instances = []
                    batch_types = []
                    batch_aux = []

                    # Loop through the B in batched patches (B, C, n_patches, patch_h, patch_w)
                    for j in range(patches.shape[0]):

                        pred_inst = []
                        pred_type = []
                        pred_aux = []

                        # Divide patches into batches that can be used as input to model
                        for batch in self._get_batch(patches[j, ...], self.model.batch_size):
                            insts, types, aux = self._predict_batch(batch)
                            pred_inst.append(insts)
                            pred_type.append(types) if types is not None else None
                            pred_aux.append(aux) if aux is not None else None
                            
                            running_int += batch.shape[0]
                            batch_loader.set_postfix(patches=f"{running_int}/{n_patches_total}")
                        
                        pred_inst = torch.cat(pred_inst, dim=0)
                        pred_type = torch.cat(pred_type, dim=0) if pred_type else None
                        pred_aux = torch.cat(pred_aux, dim=0) if pred_aux else None
                        
                        batch_instances.append(pred_inst)
                        batch_types.append(pred_type)
                        batch_aux.append(pred_aux)
                else:
                    # Ǐf no patching required (pannuke) -> straight forward inference
                    n_patches = batch.shape[0]
                    batch_instances, batch_types, batch_aux = self._predict_batch(batch)
                    insts = zip(fnames, tensor_to_ndarray(batch_instances))

                    types = zip(fnames, [None]*len(fnames))
                    if batch_types is not None:
                        types = zip(fnames, tensor_to_ndarray(batch_types))

                    aux = zip(fnames, [None]*len(fnames))
                    if batch_aux is not None:
                        aux = zip(fnames, tensor_to_ndarray(batch_aux))
                    
                    soft_instances.extend(insts)
                    soft_types.extend(types)
                    aux_maps.extend(aux)

                    running_int += n_patches
                    batch_loader.set_postfix(patches=f"{running_int}/{len(self.folderset.fnames)}")
                    
                # Stitch back to full size images if needed
                if requires_patching:
                    insts = torch.stack(batch_instances, dim=0).permute(0, 2, 1, 3, 4)
                    insts = tilertorch.stitch_batched_patches(insts)
                    insts = zip(fnames, tensor_to_ndarray(insts))

                    types = zip(fnames, [None]*len(fnames))
                    if all(e is not None for e in batch_types):
                        types = torch.stack(batch_types, dim=0).permute(0, 2, 1, 3, 4)
                        types = tilertorch.stitch_batched_patches(types)
                        types = zip(fnames, tensor_to_ndarray(types))

                    aux = zip(fnames, [None]*len(fnames))
                    if all(e is not None for e in batch_aux):
                        aux = torch.stack(batch_aux, dim=0).permute(0, 2, 1, 3, 4)
                        aux = tilertorch.stitch_batched_patches(aux)
                        aux = zip(fnames, tensor_to_ndarray(aux))

                    soft_instances.extend(insts)
                    soft_types.extend(types)
                    aux_maps.extend(aux)

        self.soft_insts = OrderedDict(soft_instances)
        self.soft_types = OrderedDict(soft_types)
        self.aux_maps = OrderedDict(aux_maps)


    def post_process(self):
        """
        Run post processing pipeline
        """
        assert "soft_insts" in self.__dict__.keys(), "No predictions found, run inference first."
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
