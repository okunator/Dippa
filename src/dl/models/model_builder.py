import segmentation_models_pytorch as smp
import torch
import src.dl.models as mdls
from typing import Optional
from torch import nn
from omegaconf import DictConfig
from src.utils.file_manager import ProjectFileManager
from src.dl.activations.utils import convert_relu_to_mish, convert_relu_to_swish


class ModelBuilder(ProjectFileManager):
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig) -> None:
        """
        Sets a pytorch model specification with an encoder branch and
        a decoder branch. Then converts the model for panoptic segmentation task
        if specified in the config.py file. If objective is to do only instance
        segmentation then this just model = s a wrapper for the model without
        semantic segmentation decoder branch. This can take in your own model specs,
        smp models, toolbelt models or any models with distinct encoder and decoder
        specifications

        Args:
            dataset_args (DictConfig): omegaconfig DictConfig specifying arguments
                related to the dataset that is being used. config.py for more info
            experiment_args (DictConfig): omegaconfig DictConfig specifying arguments
                that are used for creating result folders and files. Check config.py
                for more info
        """
        super(ModelBuilder, self).__init__(dataset_args, experiment_args)

    @property
    def nclasses(self) -> int:
        return len(self.classes)

    @classmethod
    def set_model(cls,
                  conf: DictConfig,
                  encoder_name: str = "resnext50_32x4d",
                  encoder_weights: str = "imagenet",
                  relu_to_mish: bool = False,
                  relu_to_swish: bool = False,
                  **kwargs) -> nn.Module:
        """
        Initializes smp or other pytorch model specifications

        Args:
            conf (DictConfig): the config.py file
            encoder_name (str): name of the encoder to be used. 
            encoder_weights (str): One of ("imagenet", "instagram", None)
            relu_to_mish: bool = converts ReLU units Mish units in the model
            relu_to_swish: bool = converts ReLU units Swish units in the model

        Returns:
            nn.Module initialized pytorch model specification

        Example:
            >>> from src.conf.config import CONFIG
            >>> ModelBuilder.get_model(CONFIG)
        """
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        c = cls(dataset_args, experiment_args)

        kwargs = kwargs.copy()
        kwargs.setdefault("classes", c.nclasses)
        kwargs.setdefault("encoder_name", encoder_name)
        kwargs.setdefault("encoder_weights", encoder_weights)
        kwargs.setdefault("aux_branch_name", c.aux_branch)

        if c.class_types == "instance":
            mn = mdls.MODEL_NAIVE_LOOKUP[c.model_name]
            model = mdls.__dict__[mn](**kwargs)
        elif c.class_types == "panoptic":
            mn = mdls.MODEL_LOOKUP[c.model_name]
            model = mdls.__dict__[mn](**kwargs)

        if relu_to_mish:
            convert_relu_to_mish(model)

        if relu_to_swish:
            convert_relu_to_swish(model)
        
        return model

