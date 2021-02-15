import segmentation_models_pytorch as smp
import torch
import src.dl.models as mdls
from typing import Optional
from torch import nn
from omegaconf import DictConfig
from src.utils.file_manager import ProjectFileManager
from src.dl.activations.utils import convert_relu_to_mish, convert_relu_to_swish


class ModelBuilder:
    def __init__(self,
                 dataset_args: DictConfig,
                 model_args: DictConfig,
                 **kwargs) -> None:
        """
        Class which builds the model from the architectural desing choices 
        which are specified in experiment.yml
        
        Args:
            dataset_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the dataset that is being used.
            model_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the model architecture that is being used. 
        """
        self.activation: str = model_args.architecture_design.activation
        self.normalization: str = model_args.architecture_design.normalization
        self.weight_standardize: bool = model_args.architecture_design.weight_standardize
        self.weight_init: str = model_args.architecture_design.weight_init
        self.encoder_name: str = model_args.architecture_design.encoder
        self.pretrained: bool = model_args.architecture_design.pretrain
        self.short_skips: str = model_args.architecture_design.short_skips
        self.long_skips: str = model_args.architecture_design.long_skips
        self.merge_policy: str = model_args.architecture_design.merge_policy
        self.upsampling: str = model_args.architecture_design.upsampling

    @classmethod
    def set_model(cls,
                  dataset_args: DictConfig,
                  model_args: DictConfig,
                  n_classes: int,
                  **kwargs) -> nn.Module:
        """
        Initializes smp or other pytorch model specifications

        Args:
            dataset_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the dataset that is being used.
            model_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the model architecture that is being used.
            n_classes (int):
                Number of classses in the training dataset

        Returns:
            nn.Module initialized pytorch model specification
        """
        c = cls(dataset_args, model_args)
        

