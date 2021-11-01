import torch
import torch.nn as nn

from ._base.basic import BasicConvBlockPreact, BasicConvBlock


class DepthwiseSeparableConvBlock(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            n_blocks: int=2,
            preactivate: bool=False,
            attention: bool=False
        ) -> None:
        """ 
        DepthwiseSeparable convolution 

        Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
        (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
        """

        super(DepthwiseSeparableConvBlock, self).__init__()

        # use either preact or normal conv block
        ConvBlock = BasicConvBlockPreact if preactivate else BasicConvBlock

        self.conv1 = ConvBlock(
            in_channels=in_channels, 
            out_channels=in_channels,
            same_padding=same_padding, 
            normalization=normalization, 
            activation=activation, 
            weight_standardize=weight_standardize,
        )

        blocks = list(range(1, n_blocks))
        for i in blocks:
            use_attention = i == blocks[-1]
            conv_block = ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                same_padding=same_padding,
                normalization=normalization,
                activation=activation,
                weight_standardize=weight_standardize,
                attention=use_attention if attention else False
            )
            self.add_module('conv%d' % (i + 1), conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
        
