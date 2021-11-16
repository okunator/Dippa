from .basic.basic_block import BasicBlock
from .basic.basic_bottleneck_block import BottleneckBasicBlock
from .basic.basic_dws_block import DepthWiseSeparableBasicBlock
from .basic.basic_mbconv_block import MobileInvertedBasicBlock
from .basic.basic_fusedmbconv_block import FusedMobileInvertedBasicBlock
from .dense.dense_block import DenseBlock
from .residual.residual_block import ResidualBlock
from .residual.residual_bottleneck_block import BottleneckResidualBlock
from .residual.residual_dws_block import DepthWiseSeparableResidualBlock
from .residual.residual_mbconv_block import MobileInvertedResidualBlock
from .residual.residual_fusedmbconv_block import FusedMobileInvertedResidualBlock


CONV_LOOKUP = {
    "basic": {
        "basic": "BasicBlock",
        "bottleneck": "BottleneckBasicBlock",
        "mbconv": "MobileInvertedBasicBlock",
        "fusedmbconv": "FusedMobileInvertedBasicBlock",
        "dws": "DepthWiseSeparableBasicBlock"
    },
    "residual": {
        "basic": "ResidualBlock",
        "bottleneck": "BottleneckResidualBlock",
        "mbconv": "MobileInvertedResidualBlock",
        "fusedmbconv": "FusedMobileInvertedResidualBlock",
        "dws": "DepthWiseSeparableResidualBlock"
    },
    "dense": {
        "basic": "DenseBlock"
    }
}


__all__ = [
    "BasicBlock", "BottleneckBasicBlock", "DepthWiseSeparableBasicBlock",
    "DenseBlock", "ResidualBlock", "BottleneckResidualBlock",
    "DepthWiseSeparableResidualBlock", "MobileInvertedResidualBlock",
    "FusedMobileInvertedResidualBlock", "FusedMobileInvertedBasicBlock",
    "MobileInvertedBasicBlock", "CONV_LOOKUP"
]