
import torch

from collections import OrderedDict
from typing import List, Dict
from omegaconf import DictConfig
from src.utils.file_manager import FileManager


class Inferer(FileManager):
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig,
                 inference_args: DictConfig,
                 runtime_args: DictConfig,
                 **kwargs) -> None:
        """
        Inferer class for the lightning model

        Args: 
            dataset_args (DictConfig): 
                omegaconfig DictConfig specifying arguments related to the 
                dataset that is being used. config.py for more info
            experiment_args (DictConfig): 
                Omegaconfig DictConfig specifying arguments that are used for
                creating result folders and files. Check config.py for more info
            inference_args (DictConfig): 
                Omegaconfig DictConfig specifying arguments that are used for 
                inference and post processing. Check config.py for more info
            runtime_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying batch size and 
                input image size for the model.
        """
        super(Inferer, self).__init__(dataset_args, experiment_args)

        self.batch_size: int = runtime_args.batch_size
        self.input_size: int = runtime_args.model_input_size
        self.verbose: bool = inference_args.verbose
        self.fold: str = inference_args.data_fold
        self.test_time_augs: bool = inference_args.tta
        self.thresh_method: str = inference_args.thresh_method
        self.thresh: float = inference_args.threshold
        self.post_proc: bool = inference_args.post_processing
        self.post_proc_method: str = inference_args.post_proc_method

        # init containers for results
        self.soft_insts = OrderedDict()
        self.soft_types = OrderedDict()
        self.inst_maps = OrderedDict()
        self.type_maps = OrderedDict()
        self.aux_maps = OrderedDict()
        self.result_maps = OrderedDict()

        # Put SegModel to gpu|cpu and eval mode
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.model.eval()
        torch.no_grad()


class HoverNetPostProcessor:
    def __init__(self):
        """
        blah
        """

    def post_process(self, 
                     inst_map: np.ndarray,
                     prob_map: np.ndarray,
                     aux_map: np.ndarray = None,
                     method1: str = "regular", 
                     method2: str = "default") -> np.ndarray:
        """
        Apply the post-process pipeline. Uses the appropriate method depending on the network architecture.
        For example, if aux branch regresses horizontal and vertical maps, then "cellpose" and "hover" 
        methods are considered. If there is no auxiliary branch in the network, then the "regular" one is used.
        The method architecture is specified in the config.py file. Methods are found in post_processing/...

        Args:
            inst_map (np.ndarray, np.int32):
                The instance map. (Output from thresholding). Shape (H, W)
            prob_map (np.ndarray, np.float64):
                The probability map from the network, (Before thresholding). Shape (H, W)
            aux_map (np.ndarray, np.float64): 
                Output from the auxiliary regression branch of the network. Shape (H, W, C)
                For "hover" and "cellpose", C = 2.
            method1 (str, "regular"):
                The post processing method. One of ["hover", "micro", "cellpose", "regular"] 
            method2 (str, "default")
                One of ["default", "experimental"]. Use default.

        Returns:
            The post-processed intance map (np.ndarray)
        """


    
