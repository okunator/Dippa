import torch.nn as nn

from ..activations.utils import act_func


class SqueezeAndExcite(nn.Module):
    def __init__(
            self,
            in_channels: int,
            squeeze_ratio: int=0.25,
            activation: str="relu",
            gate_activation: str="sigmoid",
        ) -> None:
        """ 
        Squeeze-and-Excitation block

        Paper: https://arxiv.org/abs/1709.01507

        Args:
        ---------
            in_channels (int): 
                Number of input channels
            squeeze_ratio (float): 
                Ratio of squeeze
            activation (str): 
                Activation layer after squeeze
            gate_activation (str, default="sigmoid"): 
                Attention gate function
        """

        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = round(in_channels*squeeze_ratio)

        # squeeze channel pooling
        self.conv_squeeze = nn.Conv2d(
            in_channels, squeeze_channels, 1, bias=True
        )
        self.act = act_func(activation)

        # excite channel pooling
        self.conv_excite = nn.Conv2d(
            squeeze_channels, in_channels, 1, bias=True
        )
        self.gate = act_func(gate_activation)

    def forward(self, x):

        # squeeze
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_squeeze(x_se)
        x_se = self.act(x_se)

        # excite
        x_se = self.conv_excite(x_se)
        
        return x * self.gate(x_se)