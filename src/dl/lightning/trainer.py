import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig
from typing import List

from src.utils.file_manager import FileManager


# Inheriting Trainer does not work
class SegTrainer:
    def __init__(self,
                 experiment_args: DictConfig,
                 dataset_args: DictConfig,
                 training_args: DictConfig,
                 extra_callbacks: List[pl.Callback] = None) -> None:
        """
        Initializes lightning trainer based on the experiment.yml

        Args:
            experiment_args (omegaconf.DictConfig):
                Omegaconf DictConfig specifying arguments that
                are used for creating result folders and files.
            dataset_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the dataset that is being used.
            training_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments that are
                used for training a network.
            extra_callbacks (List[pl.CallBack], default=None):
                List of extra callbacks to add to the Trainer
        """
        # init file manager
        fm = FileManager(
            experiment_args=experiment_args,
            dataset_args=dataset_args
        )
        # set test tube logger
        self.tt_logger = pl.loggers.TestTubeLogger(
            save_dir=fm.result_folder,
            name=experiment_args.experiment_name,
            version=experiment_args.experiment_version
        )

        # set save dir to results/{experiment_name}/version_{experiment_version}
        self.ckpt_dir = fm.experiment_dir

        # set checkpoint callback
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath = self.ckpt_dir.as_posix(),
            save_top_k = 1,
            save_last = True,
            verbose = True, 
            monitor = 'avg_val_loss',
            mode = 'min',
            prefix = ''
        )

        # set gpu monitoring callback
        gpu_callback = pl.callbacks.GPUStatsMonitor()

        # set attributes
        self.callbacks = [checkpoint_callback, gpu_callback]
        self.callbacks += extra_callbacks if extra_callbacks is not None else []
        self.gpus = training_args.num_gpus
        self.epochs = training_args.num_epochs
        self.resume_training = training_args.resume_training
        self.last_ckpt = fm.get_model_checkpoint("last") if self.resume_training else None
        
        # set logging dir
        self.logging_dir = fm.experiment_dir / "tf"

 

    @classmethod
    def from_conf(cls, conf: DictConfig, extra_callbacks: List[pl.Callback] = None):
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        training_args = conf.training_args

        c = cls(
            experiment_args,
            dataset_args,
            training_args,
            extra_callbacks
        )
        
        return pl.Trainer(
            max_epochs=c.epochs,
            gpus=c.gpus,
            logger=c.tt_logger,
            callbacks=c.callbacks,
            resume_from_checkpoint=c.last_ckpt,
            profiler=True
        )