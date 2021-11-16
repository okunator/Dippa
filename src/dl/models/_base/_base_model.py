import torch
import torch.nn as nn
from typing import Dict

from ..initialization import initialize_decoder, initialize_head


class BaseMultiTaskSegModel(nn.ModuleDict):
    """
    Base class for multi-task segmentation model
    """
    def initialize(self) -> None:
        """
        Init the decoder branches and their classification/regression
        heads.
        """
        for name, module in self.items():
            if "deocder" in name:
                initialize_decoder(module)
            if "head" in name:
                initialize_head(module)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        features = self.encoder(x)
        
        results = {}    
        decoders = [k for k in self.keys() if "decoder" in k]
        heads = [k for k in self.keys() if "head" in k]
        
        for dec, head in zip(decoders, heads):
            x = self[dec](*features)
            x = self[head](x)
            
            branch = dec.split("_")[0]
            results[f"{branch}_map"] = x
        
        return results

    def freeze_encoder(self) -> None:
        """
        Freeze the parameters of the encoeder
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
    