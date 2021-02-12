import torch.nn as nn
import src.dl.models.initialization as init


# Adapted from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/model.py
class MultiTaskSegModel(nn.Module):
    """
    Base class for models instance seg models that have also cell type cls branch
    and an optional aux branch
    """
    def initialize(self):
        init.initialize_decoder(self.inst_decoder)
        init.initialize_head(self.inst_seg_head)

        if self.type_branch:
            init.initialize_decoder(self.type_decoder)
            init.initialize_head(self.type_seg_head)

        if self.aux_branch:
            init.initialize_decoder(self.aux_decoder)
            init.initialize_head(self.aux_seg_head)

    def forward(self, x):
        features = self.encoder(x)
        insts = self.inst_decoder(*features)
        insts = self.inst_seg_head(insts)

        types = None
        if self.type_branch:
            types = self.type_decoder(*features)
            types = self.type_seg_head(types)

        aux = None
        if self.aux_branch:
            aux = self.aux_decoder(*features)
            aux = self.aux_seg_head(aux)

        return {
            "instances": insts,
            "types": types,
            "aux": aux
        }
