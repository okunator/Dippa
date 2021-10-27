import torch
import torch.nn as nn

from ...modules import ResidualConvBlockPreact, ResidualConvBlock


class MultiBlockResidual(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            n_blocks: int=2,
            preactivate: bool=False
        ) -> None:
        """
        Stack residual conv blocks in a ModuleDict. These are used in 
        the full sized decoderblocks. The number of basic conv blocks 
        can be adjusted. Default is 2. The residual connection is 
        applied at the final conv block, before the last activation.

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                If True, performs same-covolution
            normalization (str): 
                Normalization method to be used.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            activation (str):
                Activation method. One of: "mish", "swish", "relu",
                "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh",
                "sigmoid", "silu", "prelu", "leaky-relu", "elu",
                "hardshrink", "tanhshrink", "hardsigmoid"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            n_blocks (int, default=2):
                Number of BasicConvBlocks used in this block
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
        """
        super(MultiBlockResidual, self).__init__()

        # use either preact or normal (original) resblock
        ResBlock = ResidualConvBlock
        if preactivate:
            ResBlock = ResidualConvBlockPreact

        # First res conv block. If n_blocks != 1 no residual skip 
        # at the first conv block
        use_residual = n_blocks == 1
        self.conv1 = ResBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            same_padding=same_padding,
            normalization=normalization, 
            activation=activation, 
            weight_standardize=weight_standardize,
            use_residual=use_residual
        )

        blocks = list(range(1, n_blocks))
        for i in blocks:
            # apply residual connection at the final conv block
            use_residual = i == blocks[-1]
            conv_block = ResBlock(
                in_channels=out_channels, 
                out_channels=out_channels, 
                same_padding=same_padding, 
                normalization=normalization, 
                activation=activation, 
                weight_standardize=weight_standardize,
                use_residual=use_residual
            )
            self.add_module('conv%d' % (i + 1), conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, conv_block in self.items():
            x = conv_block(x)
            
        return x
