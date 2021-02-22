import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Union, Optional
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from src.utils.file_manager import FileHandler
from src.patching import TilerStitcherTorch
from src.dl.torch_utils import tensor_to_ndarray

from .post_processing.processor_builder import PostProcBuilder
from .predictor import Predictor



SUFFIXES = (".jpeg", ".jpg", "tif", ".tiff", ".png")


class FolderDataset(Dataset, FileHandler):
    def __init__(self, folder_path: Union[str, Path], pattern: str="*"):
        """
        Simple pytorch folder dataset. Assumes that
        folder_path contains only image files which are readable
        by cv2.

        Args:
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
                 stride_size: int=128,
                 thresh_method: int="naive",
                 thresh: float=0.5,
                 **kwargs) -> None:
        """
        Class to perform inference and post-processing

        Args:
            model (pl.LightningModule):
                Input SegModel (lightning model) specified in lightning_model.py.
            data_dir (str, optional, default=None):
                If not None this directory will be used as the input data directory.
                Assumes that the directory contains only cv2 readable image files (.png, .tif, etc).
                This argument overrides all other dataset related argument.
            dataset (str, optional, default=None):
                One of ("kumar","consep","pannuke","dsb2018", "monusac", None)
                If data_dir == None images from this dataset will be used for inference. 
                If both dataset == None & data_dir == None. The inference is performed
                on the same dataset that the input model was trained with.
            data_fold (str, default="test"):
                Which fold of data to run inference. One of ("train", "test"). If data_dir is set
                this arg will be ignored.
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
        """
        assert isinstance(model, pl.LightningModule), "Input model needs to be a lightning model"
        assert dataset in ("kumar", "consep", "pannuke", "dsb2018", "monusac", None)
        assert model_weights in ("best", "last")
        assert data_fold in ("train", "test")

        # set model to device and to inference mode
        self.model = model
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()
        torch.no_grad()
        
        # set stride size and check it is correct
        assert stride_size <= self.model.input_size, f"stride_size: {stride_size} > {self.model.input_size}"
        self.stride_size = stride_size

        # Load trained weights for the model 
        ckpt_path = self.model.fm.get_model_checkpoint(model_weights)
        checkpoint = torch.load(ckpt_path, map_location = lambda storage, loc : storage)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        # Set input data folder
        self.in_data_dir = in_data_dir
        if self.in_data_dir is None:
            if dataset is not None:
                dirs = self.model.fm.get_data_dirs(dataset)
                self.in_data_dir = dirs[f"{data_fold}_im"]
            else:
                dataset = self.model.fm.train_dataset
                dirs = self.model.fm.get_data_dirs(dataset)
                self.in_data_dir = dirs[f"{data_fold}_im"]
            
        # Set dataset dataloader
        self.folderset = FolderDataset(self.in_data_dir, pattern=fn_pattern)
        self.dataloader = DataLoader(self.folderset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

        # Some helper classes 
        self.predictor = Predictor(self.model)
        self.post_processor = PostProcBuilder.set_postprocessor(
            aux_branch=self.model.aux_branch, 
            aux_type=self.model.aux_type,
            thresh_method=thresh_method,
            thresh=thresh
        )

    def _get_batch(self, patches: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Divide a set of patches into batches of patches

        Args:
            patches (torch.Tensor): 
                Batch of patches in. Shape (C, num_patches, pH, pW)
            batch_size (int): 
                size of the batch
            
        Yields:
            torch.Tensor of shape (batch_size, C, H, W)
        """
        for i in range(0, patches.shape[1], batch_size):
            batch = patches[:, i:i+batch_size, ...].permute(1, 0, 2, 3) # to shape (B, C, pH, pW)
            yield batch 

    def _predict_batch(self, batch: torch.Tensor) -> Tuple[Union[torch.Tensor, None]]:
        """
        Forward pass + classify. Handles missing branches in the model.

        Args:
            batch (torch.Tensor):
                A batch of patches. Shape (B, C, patch_size, patch_size)

        Returns:
            A tuple of tensors containing the predictions. If network does not
            contain aux or type branch the predictions are None
        """
        # TODO: tta
        pred = self.predictor.forward_pass(batch)
        insts = self.predictor.classify(pred["instances"], act="sigmoid", return_type="torch") # goes to cpu
        types = self.predictor.classify(pred["types"], act="softmax", return_type="torch") if pred["types"] is not None else None
        aux = self.predictor.classify(pred["aux"], act=None, return_type="torch") if pred["aux"] is not None else None
        return insts, types, aux

    def run_inference(self) -> None:
        """
        Run inference on the images in the input folder. 
        Results will be saved in OrderedDicts:

        self.res_insts: instance branch predicitons
        self.res_types: type branch predictions
        self.res_aux: aux branch predictions
        """

        def unpack(l: List[Tuple[str, int]]):
            """
            Unpack the list of tuples of tuples that is generated by the pipeline
            into a list of tuples and convert the tensors in it to ndarrays
            """
            result = [] 
            for names, tensor in l:
                batches = tuple(zip(names, list(tensor))) 
                for name, im in batches:
                    result.append((name, tensor_to_ndarray(im)))
            return result

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

                # Set patching flag (most datasets require patching), Assumes square pathces here
                requires_patching = True if batch.shape[-1] > self.model.input_size else False
                patch_size = (self.model.input_size, self.model.input_size)

                if requires_patching:

                    # Do patching if images bigger than model input size
                    tilertorch = TilerStitcherTorch(batch.shape, patch_size, self.stride_size, padding=True)
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
                            
                            # increment tqdm logs
                            running_int += batch.shape[0]
                            batch_loader.set_postfix(patches=f"{running_int}/{n_patches_total}")
                        
                        # cat lists to the input tensor shape
                        pred_inst = torch.cat(pred_inst, dim=0)
                        pred_type = torch.cat(pred_type, dim=0) if pred_type else None
                        pred_aux = torch.cat(pred_aux, dim=0) if pred_aux else None
                        
                        batch_instances.append(pred_inst)
                        batch_types.append(pred_type)
                        batch_aux.append(pred_aux)
                else:
                    # Ǐf no patching required (pannuke) -> straight forward inference
                    n_patches = batch.shape[0]
                    insts, types, aux = self._predict_batch(batch)
                    soft_instances.append((fnames, insts))
                    soft_types.append((fnames, types)) if types is not None else None
                    aux_maps.append((fnames, aux)) if aux is not None else None

                    # increment tqdm
                    running_int += n_patches
                    batch_loader.set_postfix(patches=f"{running_int}/{len(self.folderset.fnames)}")
                    
                # Stitch back to full size images if needed
                if requires_patching:
                    # stack list of tensors into one big tensor and swap dims to right order for stiching
                    batch_instances = torch.stack(batch_instances, dim=0).permute(0, 2, 1, 3, 4) 
                    batch_types = torch.stack(batch_types, dim=0).permute(0, 2, 1, 3, 4) if batch_types is not None else None
                    batch_aux = torch.stack(batch_aux, dim=0).permute(0, 2, 1, 3, 4) if batch_aux is not None else None

                    # Stitch back to original size and save along with file names
                    soft_instances.append((fnames, tilertorch.stitch_batched_patches(batch_instances)))
                    soft_types.append((fnames, tilertorch.stitch_batched_patches(batch_types)))
                    aux_maps.append((fnames, tilertorch.stitch_batched_patches(batch_aux)))

        self.aux_maps = OrderedDict(unpack(aux_maps))
        self.soft_insts = OrderedDict(unpack(soft_instances))
        self.soft_types = OrderedDict(unpack(soft_types))


    def post_process(self):
        """
        Run post processing pipeline
        """
        assert "soft_insts" in self.__dict__.keys(), "No predictions found, run inference first."
        maps = self.post_processor.run_post_processing(self.soft_insts, self.aux_maps, self.soft_types)

        # save to containers
        self.inst_maps = OrderedDict()
        self.type_maps = OrderedDict()
        for res in maps:
            name = res[0]
            self.inst_maps[name] = res[1]
            self.type_maps[name] = res[2]


    def plot_results(self):
        pass

