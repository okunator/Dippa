import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from pathlib import Path

from src.utils.file_manager import FileHandler
from src.dl.inference.predictor import Predictor
from src.patching import TilerStitcherTorch
from src.dl.torch_utils import tensor_to_ndarray


class FolderDataset(Dataset, FileHandler):
    def __init__(self, folder_path: str, pattern: str="*"):
        """
        Simple pytorch folder dataset. Assumes that
        folder_path contains only image files which are readable
        by cv2. No error checking so use with care.

        Args:
            folder_path (str):
                path to the folder containig image files
            pattern (str, default="*"):
                file pattern for getting only files that contain the pattern.
        """
        super(FolderDataset, self).__init__()
        self.fnames = sorted(Path(folder_path).glob(pattern))

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
                 tta: bool=False,
                 data_fold: str="test",
                 model_weights: str="last",
                 fn_pattern: str="*",
                 num_workers: int=8,
                 **kwargs) -> None:
        """
        Class to perform inference and post-processing

        Args:
            model (pl.LightningModule):
                Input SegModel (lightning model) specified in lightning_model.py.
            tta (bool, default=False):
                If True, performs test time augmentation. Inference time goes up
                with often marginal performance improvements.
            data_fold (str, default="test"):
                Which fold of data to run inference. One of ("train", "test").
            model_weights (str, default="last"):
                pytorch lightning saves the weights of the model for the last epoch
                and best epoch (based on validation data). One of ("best", "last").
            fn_pattern (str, default="**):
                A pattern in file names. For example, in pannuke dataset you can run 
                inference for only images of specific tissue e.g. pattern = *_Adrenal_gland_*.
            num_workers (int, default=8):
                Number of thread workers for torch dataloader
        """
        assert isinstance(model, pl.LightningModule), "Input model needs to be a lightning model"
        assert model_weights in ("best", "last")
        assert data_fold in ("train", "test")

        # set model to device and to inference mode
        self.model = model
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.eval()
        torch.no_grad()
        
        # Load trained weights for the model 
        ckpt_path = self.model.fm.get_model_checkpoint(data_fold)
        checkpoint = torch.load(ckpt_path, map_location = lambda storage, loc : storage)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        # Set dataset dataloader
        dataset = FolderDataset("/home/leos/Dippa/datasets/data/consep/train/images", pattern=fn_pattern)
        self.dataloader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=num_workers)

        # Some helper classes 
        self.predictor = Predictor(self.model)


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

        Returns
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

        for i, data in enumerate(self.dataloader):
            
            # Get data
            batch = data["im"]
            fnames = data["file"]

            # Set patching flag (most datasets require patching), Assumes square pathces here
            requires_patching = True if batch.shape[-1] > self.model.input_size else False
            patch_size = (self.model.input_size, self.model.input_size)
            stride_size = self.model.input_size//2

            if requires_patching:

                # Do patching if images bigger than model input size
                tilertorch = TilerStitcherTorch(batch.shape, patch_size, stride_size, padding=True)
                patches = tilertorch.extract_patches_from_batched(batch)

                batch_instances = []
                batch_types = []
                batch_aux = []

                # Loop through the B in batched patches (B, C, n_patches, patch_h, patch_w)
                for i in range(patches.shape[0]):

                    pred_inst = []
                    pred_type = []
                    pred_aux = []

                    # Divide patches into batches that can be used as input to model
                    for batch in self._get_batch(patches[i, ...], patches.shape[0]):
                        insts, types, aux = self._predict_batch(batch)
                        pred_inst.append(insts)
                        pred_type.append(types) if types is not None else None
                        pred_aux.append(aux) if aux is not None else None
                    
                    # cat lists to the input tensor shape
                    pred_inst = torch.cat(pred_inst, dim=0)
                    pred_type = torch.cat(pred_type, dim=0) if pred_type else None
                    pred_aux = torch.cat(pred_aux, dim=0) if pred_aux else None
                    
                    batch_instances.append(pred_inst)
                    batch_types.append(pred_type)
                    batch_aux.append(pred_aux)
            else:
                # Ǐf no patching required (pannuke) -> straight forward inference
                insts, types, aux = self._predict_batch(batch)
                soft_instances.append((fnames, insts))
                soft_types.append((fnames, types)) if types is not None else None
                aux_maps.append((fnames, aux)) if aux is not None else None
                
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

        self.res_aux = OrderedDict(unpack(aux_maps))
        self.res_insts = OrderedDict(unpack(soft_instances))
        self.res_types = OrderedDict(unpack(soft_types))

