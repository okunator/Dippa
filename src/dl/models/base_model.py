import torch
import torch.nn as nn
from typing import Dict

import src.dl.models.initialization as init
from src.dl.models.modules import Mish, Swish


# Adapted from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/model.py
class MultiTaskSegModel(nn.Module):
    """
    Base class for models instance seg models that have also cell type cls branch
    and an optional aux branch
    """
    def initialize(self) -> None:
        init.initialize_decoder(self.inst_decoder)
        init.initialize_head(self.inst_seg_head)

        if self.decoder_type_branch:
            init.initialize_decoder(self.type_decoder)
            init.initialize_head(self.type_seg_head)

        if self.decoder_aux_branch:
            init.initialize_decoder(self.aux_decoder)
            init.initialize_head(self.aux_seg_head)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features = self.encoder(x)
        insts = self.inst_decoder(*features)
        insts = self.inst_seg_head(insts)

        types = None
        if self.decoder_type_branch:
            types = self.type_decoder(*features)
            types = self.type_seg_head(types)

        aux = None
        if self.decoder_aux_branch:
            aux = self.aux_decoder(*features)
            aux = self.aux_seg_head(aux)

        return {
            "instances": insts,
            "types": types,
            "aux": aux
        }

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def convert_activation(self, model: nn.Module=None, act: str="relu") -> None:
        assert act in ("swish", "mish", "leaky-relu")
            
        if act == "swish":
            Act = Swish
        elif act == "mish":
            Act = Mish
        elif act == "leaky-relu":
            Act = nn.LeakyReLU

        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, Act(inplace=False))
            else:
                self.convert_activation(child, act)
