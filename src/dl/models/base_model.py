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
    
    ###########################
    # Bunch of HACKS
    def convert_activation(
            self,
            model: nn.Module=None,
            act: str="relu"
        ) -> None:
        """
        Change the pre-trained encoder act function
        """
        assert act in ("swish", "mish", "leaky-relu")
            
        if act == "swish":
            Act = Swish
            inplace = False
        elif act == "mish":
            Act = Mish
            inplace = False
        elif act == "leaky-relu":
            Act = nn.LeakyReLU
            inplace = True

        for child_name, child in model.named_children():
            if (isinstance(child, nn.ReLU) or 
                type(child).__name__ == "MemoryEfficientSwish"):
                 # the latter clause for efficientnets
                setattr(model, child_name, Act(inplace=inplace))
            else:
                self.convert_activation(child, act)

    def convert_norm(self, model: nn.Module, norm: str) -> None:
        assert norm in ("bcn", "gn")
            
        if norm == "bcn":
            Norm = BCNorm
        elif norm == "gn":
            Norm = GroupNorm

        for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                norm_fn = Norm(num_features=child.num_features, num_groups=32)
                setattr(model, child_name, norm_fn)
            else:
                self.convert_norm(child, norm)


    def convert_conv(self, model: nn.Module, **kwargs) -> None:
        WSConv = WSConv2d 
        for child_name, child in model.named_children():
            if isinstance(child, nn.Conv2d):  
                if "image_size" in kwargs.keys(): # efficient-net hack
                    WSConv = WSConv2dStaticSamePadding

                wsconv = WSConv(
                    in_channels=child.in_channels, 
                    out_channels=child.out_channels, 
                    kernel_size=child.kernel_size,
                    bias=child.bias is not None,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    **kwargs
                )
                setattr(model, child_name, wsconv)
            else:
                self.convert_conv(child, **kwargs)
    

