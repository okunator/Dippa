import segmentation_models_pytorch as smp
import torch
from typing import Optional
from torch import nn
from omegaconf import DictConfig
from src.utils.file_manager import ProjectFileManager
from src.dl.models.utils import convert_relu_to_mish
from src.dl.models.linknet.model import LinknetSmp, LinknetSmpWithClsBranch
from src.dl.models.unet.model import UnetSmp, UnetSmpWithClsBranch
from src.dl.models.unet3plus.model import Unet3pInst, Unet3pWithClsBranch
from src.dl.models.unetplusplus.model import UnetPlusPlusSmp, UnetPlusPlusSmpWithClsBranch
from src.dl.models.pspnet.model import PSPNetSmp, PSPNetSmpWithClsBranch 
from src.dl.models.fpn.model import FpnSmp, FpnSmpWithClsBranch 
from src.dl.models.pan.model import PanSmp, PanSmpWithClsBranch 
from src.dl.models.deeplabv3.model import DeepLabV3Smp, DeepLabV3SmpWithClsBranch 
from src.dl.models.deeplabv3plus.model import DeepLabV3PlusSmp, DeepLabV3PlusSmpWithClsBranch 


class ModelBuilder(ProjectFileManager):
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig) -> None:
        """
        Takes in a pytorch model specification that has an encoder branch and
        a decoder branch. Then converts the model for panoptic segmentation task
        if specified in the config.py file. If objective is to do only instance
        segmentation then this just model = s a wrapper for the model without
        semantic segmentation decoder branch. This can take in your own model specs,
        smp models, toolbelt models or any models with distinct encoder and decoder
        specifications

        Args:
            dataset_args (DictConfig): omegaconfig DictConfig specifying arguments
                            related to the dataset that is being used.
                            config.py for more info
            experiment_args (DictConfig): omegaconfig DictConfig specifying arguments
                                          that are used for creating result folders and
                                          files. Check config.py for more info
        """
        super(ModelBuilder, self).__init__(dataset_args, experiment_args)

    @property
    def nclasses(self) -> int:
        return len(self.classes)

    @classmethod
    def set_model(cls,
                  model_name: str,
                  conf: DictConfig,
                  encoder_name: str = "resnext50_32x4d",
                  encoder_weights: str = "imagenet",
                  relu_to_mish: bool = False,
                  **kwargs) -> nn.Module:
        """
        Initializes smp or other pytorch model specifications

        Args:
            model_name (str): one of ("FPN", "DeepLabV3", "LinkNet", "Unet", "PAN", "PSPNet")
            conf (DictConfig): the config.py file
            encoder_name (str): name of the encoder to be used. This is needed for especially 
                           all the smp models.
            encoder_weights (str): One of ("imagenet", "instagram", None)

        Returns:
            nn.Module initialized pytorch model specification

        Example:
            >>> from src.conf.config import CONFIG
            >>> ModelBuilder.get_model("FPN", CONFIG)
        """

        models = (
            "FPN", "DeepLabV3", "DeepLabV3+", "LinkNet", "Unet", "PAN", 
            "PSPNet", "Unet3+", "Attention-Unet", "Unet++", "HoverNet"
        )
        assert model_name in models, (
            f"model name: {model_name} not recognized. Available models: {models}"
        )

        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        c = cls(dataset_args, experiment_args)

        kwargs = kwargs.copy()
        kwargs.setdefault("classes", c.nclasses)
        kwargs.setdefault("encoder_name", encoder_name)
        kwargs.setdefault("encoder_weights", encoder_weights)
        kwargs.setdefault("aux_branch_name", c.aux_branch)

        if c.class_types == "instance":
            if model_name == "Unet":
                model = UnetSmp(**kwargs)
            elif model_name == "Unet3+":
                model = Unet3pInst(**kwargs)
            elif model_name == "Unet++":
                model = UnetPlusPlusSmp(**kwargs)
            elif model_name == "Attention-Unet":
                kwargs.setdefault("decoder_attention_type", "scse")
                model = UnetSmp(**kwargs)
            elif model_name == "FPN":
                kwargs.setdefault("decoder_merge_policy", "cat")
                model = FpnSmp(**kwargs)
            elif model_name == "DeepLabV3":
                model = DeepLabV3Smp(**kwargs)
            elif model_name == "DeepLabV3+":
                model = DeepLabV3PlusSmp(**kwargs)
            elif model_name == "LinkNet":
                model = LinknetSmp(**kwargs)
            elif model_name == "PAN":
                model = PanSmp(**kwargs)
            elif model_name == "PSPNet":
                model = PSPNetSmp(**kwargs)

        elif c.class_types == "panoptic":
            if model_name == "Unet":
                model = UnetSmpWithClsBranch(**kwargs)
            elif model_name == "Unet3+":
                model = Unet3pWithClsBranch(**kwargs)
            elif model_name == "Unet++":
                model = UnetPlusPlusSmpWithClsBranch(**kwargs)
            elif model_name == "Attention-Unet":
                kwargs.setdefault("decoder_attention_type", "scse")
                model = UnetSmpWithClsBranch(**kwargs)
            elif model_name == "FPN":
                kwargs.setdefault("decoder_merge_policy", "cat")
                model = FpnSmpWithClsBranch(**kwargs)
            elif model_name == "DeepLabV3":
                model = DeepLabV3SmpWithClsBranch(**kwargs)
            elif model_name == "DeepLabV3+":
                model = DeepLabV3PlusSmpWithClsBranch(**kwargs)
            elif model_name == "LinkNet":
                model = LinknetSmpWithClsBranch(**kwargs)
            elif model_name == "PAN":
                model = PanSmpWithClsBranch(**kwargs)
            elif model_name == "PSPNet":
                model = PSPNetSmpWithClsBranch(**kwargs)

            if relu_to_mish:
                convert_relu_to_mish(model)
            
            return model

