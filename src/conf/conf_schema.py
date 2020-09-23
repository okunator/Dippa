from typing import List, Dict
from pathlib import Path
from omegaconf import DictConfig, MISSING
from dataclasses import dataclass, field
from src.settings import PATCH_DIR, RESULT_DIR


@dataclass
class DataArgs:
    dataset: str = MISSING
    class_types: str = MISSING
    patches_dtype: str =  MISSING
    hdf5_patches_root_dir: str = Path(PATCH_DIR / "hdf5").as_posix()
    npy_patches_root_dir: str = Path(PATCH_DIR / "npy").as_posix()
    phases: List[str] = field(default_factory=lambda:["train", "test"])
    tissues: List[str] = field(default_factory=lambda:[])


@dataclass
class SharedArgs:
    model_input_size: int = 256
    batch_size: int = 6
    tta: bool = True
    verbose: bool = True


@dataclass
class ExperimentArgs:
    model_name: str = MISSING
    experiment_version: str = MISSING
    experiment_root_dir: str = RESULT_DIR.as_posix()


@dataclass
class TrainingArgs(SharedArgs):
    tta: bool = MISSING
    resume_training: bool = MISSING
    num_epochs: int = MISSING
    num_gpus: int = MISSING
    optimizer_args: Dict[str, float] = field(default_factory=lambda:{
        "lr":MISSING,
        "encoder_lr":MISSING,
        "weight_decay":MISSING,
        "encoder_weight_decay":MISSING
    })
    scheduler_args : Dict[str, float] = field(default_factory=lambda: {
        "factor":MISSING,
        "patience":MISSING
    })
    loss_args : Dict[str, float] = field(default_factory=lambda:{
        "edge_weight":MISSING
    })

        
@dataclass
class PatchingArgs(SharedArgs):
    patch_size: int = MISSING
    stride_size: int = MISSING
    crop_to_input: bool = MISSING


@dataclass
class InferenceArgs(SharedArgs):
    smoothen: bool = MISSING
    data_fold: str = MISSING
    threshold: float = MISSING
    model_weights: str = MISSING
    

@dataclass
class Schema:
    dataset_args: DataArgs = DataArgs()
    experiment_args: ExperimentArgs = ExperimentArgs()
    patching_args: PatchingArgs = PatchingArgs()
    training_args: TrainingArgs = TrainingArgs()
    inference_args: InferenceArgs = InferenceArgs()
        