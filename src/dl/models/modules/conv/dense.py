import torch

from .base_conv_block import BaseConvBlock


class DenseConvBlockPreact(BaseConvBlock):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            batch_norm: str="bn",
            activation: str="relu",
            weight_standardize: bool=False
        ) -> None:
        """
        Dense preact conv block that can be used in decoders

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Normalization method. One of: "bn", "bcn", None
            activation (str, default="relu"):
                Activation method. One of: "relu", "swish". "mish"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
        """
        super(DenseConvBlockPreact, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class DenseConvBlock(BaseConvBlock):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            batch_norm: str="bn",
            activation: str="relu",
            weight_standardize: bool=False
        ) -> None:
        """
        Dense conv block that can be used in decoders

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Normalization method. One of: "bn", "bcn", None
            activation (str, default="relu"):
                Activation method. One of: "relu", "swish". "mish"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
        """
        super(DenseConvBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x