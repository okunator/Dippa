from torch.optim import (
    Adam, 
    RMSprop, 
    SGD, 
    Adadelta, 
    Adagrad, 
    Adamax, 
    AdamW, 
    ASGD
)

from torch_optimizer import (
    AccSGD,
    AdaBound,
    AdaBelief,
    AdamP,
    Apollo,
    AdaMod,
    DiffGrad,
    Lamb,
    NovoGrad,
    PID,
    QHAdam,
    QHM,
    RAdam,
    SGDW,
    Yogi,
    Ranger,
    RangerQH,
    RangerVA,
    Lookahead
)

OPTIM_LOOKUP = {
    "adam": "Adam",
    "rmsprop": "RMSprop",
    "sgd": "SGD",
    "adadelta": "Adadelta",
    "apollo": "Apollo",
    "adabelief": "AdaBelief",
    "adamp": "AdamP",
    "adagrad": "Adagrad",
    "adamax": "Adamax",
    "adamw": "AdamW",
    "asgd": "ASGD",
    "accsgd": "AccSGD",
    "adabound": "AdaBound",
    "adamod": "AdaMod",
    "diffgrad": "DiffGrad",
    "lamb": "Lamb",
    "novograd": "NovoGrad",
    "pid": "PID",
    "qhadam": "QHAdam",
    "qhm": "QHM",
    "radam": "RAdam",
    "sgdw": "SGDW",
    "yogi": "Yogi",
    "ranger": "Ranger",
    "rangerqh": "RangerQH",
    "rangerva": "RangerVA",
}


__all__ = [
    "OPTIM_LOOKUP", "Adam", "RMSprop", "SGD", "Adadelta", "Adagrad", "Adamax",
    "AdamW", "ASGD", "AccSGD", "AdaBound", "AdaBelief", "AdamP", "Apollo",
    "AdaMod", "DiffGrad", "Lamb", "NovoGrad", "PID", "QHAdam", "QHM", "RAdam",
    "SGDW", "Yogi", "Ranger", "RangerQH", "RangerVA", "Lookahead"
]