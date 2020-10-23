import segmentation_models_pytorch as smp
import torch
from torch import nn
from omegaconf import DictConfig
from src.utils.file_manager import ProjectFileManager


class SmpModelWithClsBranch(nn.Module):
   def __init__(self, inst_model: nn.Module, type_model: nn.Module):
       """
       This class adds a semantic segmentation decoder branch to any smp model
       that is specified to do binary segmentation.
       smp = segmentation_models_pytorch. More at:
       https://github.com/qubvel/segmentation_models.pytorch
       
       Args:
            inst_model (nn.Module): smp model used for binary segmentation. n_classes needs to be 2
            type_model (nn.Module): smp model used for semantic segmentation with n_classes 
       """
       super().__init__()
       self.encoder = inst_model.encoder
       self.inst_decoder = inst_model.decoder
       self.type_decoder = type_model.decoder
       self.inst_seg_head = inst_model.segmentation_head
       self.type_seg_head = type_model.segmentation_head

   def forward(self, x):
       features = self.encoder(x)
       insts = self.inst_decoder(*features)
       types = self.type_decoder(*features)
       return {
           "instances": self.inst_seg_head(insts), 
           "types": self.type_seg_head(types)
       }


class SmpGeneralModel(nn.Module):
   def __init__(self, inst_model):
       """
        Wrapper for smp model for binary or semantic segmentation.
        Args:
            inst_model (nn.Module): smp mode for binary segmentation
        """
       super().__init__()
       self.encoder = inst_model.encoder
       self.inst_decoder = inst_model.decoder
       self.inst_seg_head = inst_model.segmentation_head

   def forward(self, x):
       features = self.encoder(x)
       insts = self.inst_decoder(*features)
       return {
           "instances": self.inst_seg_head(insts)
       }


class ModelBuilder(ProjectFileManager):
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig) -> None:
        """
        Takes in a pytorch model specification that has an encoder branch and
        a decoder branch. Then converts the model for panoptic segmentation task
        if specified in the config.py file. If objective is to do only instance
        segmentation then this just returns a wrapper for the model without
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
    def set_model(cls, model_name: str, conf: DictConfig, **kwargs) -> nn.Module:
        """
        Use this method to get the model you want. This uses the config.py file to
        deduce how many classes for a specific dataset are needed and if objective
        is to do panoptic or instance segmentation. For now these models can be used
        ("FPN", "DeepLabV3", "LinkNet", "Unet", "PAN", "PSPNet")
        Example:
            >>> from src.conf.config import CONFIG
            >>> ModelBuilder.get_model("FPN", CONFIG)
        """
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args

        c = cls(dataset_args, experiment_args)

        if c.class_types == "panoptic":
            model_inst = c.init_model(model_name, 2, **kwargs)
            model_type = c.init_model(model_name, c.nclasses, **kwargs)
            return SmpModelWithClsBranch(model_inst, model_type)
        elif c.class_types == "instance":
            model_inst = c.init_model(model_name, 2, **kwargs)
            return SmpGeneralModel(model_inst)

    @staticmethod
    def init_model(model_name: str,
                   classes: int,
                   encoder: str = "resnext50_32x4d",
                   **kwargs) -> nn.Module:
        """
        Initializes smp or other pytorch model specifications

        Args:
            model_name (str): one of ("FPN", "DeepLabV3", "LinkNet", "Unet", "PAN", "PSPNet")
            classes (int): number of classes for the segmentation head
            encoder (str): name of the encoder to be used. This is needed for especially 
                           all the smp models.
        Returns:
            nn.Module initialized pytorch model specification
        """

        assert model_name in ("FPN", "DeepLabV3", "LinkNet", "Unet", "PAN", "PSPNet"), (
            f"model name: {model_name} not recognized. Check docstring"
        )
        kwargs = kwargs.copy()
        kwargs.setdefault("classes", classes)
        kwargs.setdefault("encoder_name", encoder)
        if model_name in ("FPN", "DeepLabV3", "LinkNet", "Unet", "PAN", "PSPNet"):
            if model_name == "FPN":
                kwargs.setdefault("decoder_merge_policy", "cat")
            return smp.__dict__[model_name](**kwargs)
