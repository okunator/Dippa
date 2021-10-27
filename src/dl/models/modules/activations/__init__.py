from .mish import Mish
from .swish import Swish

from torch.nn import (
    ReLU,
    ReLU6,
    RReLU,
    SELU,
    CELU,
    GELU,
    GLU,
    Tanh,
    Sigmoid,
    SiLU,
    PReLU,
    LeakyReLU,
    ELU,
    Hardshrink,
    Tanhshrink,
    Hardsigmoid
)


ACT_LOOKUP = {
    "mish": "Mish",
    "swish": "Swish",
    "relu": "ReLU",
    "relu6": "ReLU6",
    "rrelu": "RReLU",
    "selu": "SELU",
    "celu": "CELU",
    "gelu": "GELU",
    "glu": "GLU",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "silu": "SiLU",
    "prelu": "PReLU",
    "leaky-relu": "LeakyReLU",
    "elu": "ELU",
    "hardshrink": "Hardshrink",
    "tanhshrink": "Tanhshrink",
    "hardsigmoid": "Hardsigmoid",
}