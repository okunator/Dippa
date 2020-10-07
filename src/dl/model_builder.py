import torch
from torch import nn


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
       return self.inst_seg_head(insts), self.type_seg_head(types)


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
       return self.inst_seg_head(insts)
