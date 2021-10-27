import torch
import torch.nn as nn
from typing import Dict

from .modules import (
    Mish, Swish, BCNorm, GroupNorm,
    WSConv2d, WSConv2dStaticSamePadding
)

from .initialization import initialize_decoder, initialize_head


class MultiTaskSegModel(nn.Module):
    """
    Base class for instance seg models that have also cell type cls 
    branch and an optional aux branch
    """
    def initialize(self) -> None:
        """
        Init the decoder branches and their classification/regression
        heads.
        """
        initialize_decoder(self.inst_decoder)
        initialize_head(self.inst_seg_head)

        if self.decoder_type_branch:
            initialize_decoder(self.type_decoder)
            initialize_head(self.type_seg_head)

        if self.decoder_aux_branch:
            initialize_decoder(self.aux_decoder)
            initialize_head(self.aux_seg_head)

        if self.decoder_sem_branch:
            initialize_decoder(self.sem_decoder)
            initialize_head(self.sem_seg_head)

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

        sem = None
        if self.decoder_sem_branch:
            sem = self.sem_decoder(*features)
            sem = self.sem_seg_head(sem)

        return {
            "instances": insts,
            "types": types,
            "aux": aux,
            "sem": sem
        }

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
    