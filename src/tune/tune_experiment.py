import os
import torch
import segmentation_models_pytorch as smp
from functools import partial
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, Callback #GpuUsageLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter, ExperimentAnalysis
from ray.tune.schedulers import PopulationBasedTraining

from src.dl.lightning_model import SegModel
from src.settings import RESULT_DIR


class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            avg_val_loss=trainer.callback_metrics["avg_val_loss"].item(),
            avg_val_accuracy=trainer.callback_metrics["avg_val_accuracy"].item()
        )


class CheckpointCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        with tune.checkpoint_dir(step=trainer.global_step) as checkpoint_dir:
            trainer.save_checkpoint(Path(checkpoint_dir).joinpath("checkpoint.ckpt").as_posix())           
            
            
def train_tune_checkpoint(training_args: Dict,
                          dataset_args: DictConfig,
                          experiment_args: DictConfig,
                          checkpoint_dir: Any = None,
                          num_epochs: int = 10,
                          num_gpus: int = 1) -> None:
    
   #tt_logger = pl_loggers.TestTubeLogger(
   #    save_dir=RESULT_DIR,
   #    name=experiment_args.model_name,
   #    version=experiment_args.experiment_version
   #)
    
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=tune.get_trial_dir(),
        name="", version="."
    )
    
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,  
        # logger=tt_logger,
        logger=tb_logger,
        progress_bar_refresh_rate=0,
        callbacks=[CheckpointCallback(), TuneReportCallback()],
        profiler=True
    )
    
    # Get the model from checkpoint or from 0
    if checkpoint_dir:
        base_model = smp.Unet(
            encoder_name="resnext50_32x4d", 
            classes=2
        )
        
        pl_model = SegModel(
            base_model, 
            dataset_args,
            experiment_args,
            training_args
        )
        
        # workaround to get the ckpt file since tune fails this 
        #trial_dir = Path(checkpoint_dir).parents[0]
        #ckpt_dir = [d for d in trial_dir.iterdir() if d.is_dir() and "checkpoint_" in d.name][0]
        #ckpt = list(ckpt_dir.glob("*.ckpt"))[0]
        
        checkpoint = pl_load(, map_location=lambda storage, loc: storage)
        #checkpoint = pl_load(ckpt.as_posix(), map_location=lambda storage, loc: storage)
        
        # Load the model
        pl_model.load_state_dict(checkpoint['state_dict'])
        trainer.current_epoch = checkpoint["epoch"]
    else:
        base_model = smp.Unet(
            encoder_name="resnext50_32x4d", 
            classes=2
        )
        
        pl_model = SegModel(
            base_model, 
            dataset_args,
            experiment_args,
            training_args
        )
    
    trainer.fit(pl_model)
    
    
def tune_pbt(conf: DictConfig, 
             num_samples: int = 10, 
             num_epochs: int = 10, 
             gpus_per_trial: int = 1,
             notebook: bool = False) -> ExperimentAnalysis:
    
    mn = conf.experiment_args.model_name
    ev = conf.experiment_args.experiment_version
    
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="avg_val_loss",
        mode="min",
        perturbation_interval=1,
        hyperparam_mutations={
            "lr": lambda: tune.loguniform(1e-4, 1e-1).func(None),
            "batch_size": [4, 8, 16],
            "edge_weight":[1.1, 1.2, 1.5, 2]
        })

    if notebook:
        reporter = JupyterNotebookReporter(
            overwrite=True,
            parameter_columns=["edge_weight", "lr", "batch_size"],
            metric_columns=["avg_val_loss", "avg_val_accuracy", "training_iteration"]
        )
    else:
        reporter = CLIReporter(
            parameter_columns=["edge_weight", "lr", "batch_size"],
            metric_columns=["avg_val_loss", "avg_val_accuracy", "training_iteration"]
        )

    analysis = tune.run(
        partial(
            train_tune_checkpoint,
            dataset_args=conf.dataset_args,
            experiment_args=conf.experiment_args,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=dict(conf.training_args),
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_pbt",
        local_dir=str(Path(RESULT_DIR / mn / f"version_{ev}"))
    )
    
    return analysis
