import pytorch_lightning as pl
from typing import List
from omegaconf import DictConfig

from src.utils.file_manager import FileHandler


# Inheriting Trainer does not work so init w/ class method only
class SegTrainer:
    def __init__(
            self,
            experiment_name: str,
            experiment_version: str,
            num_gpus: int,
            num_epochs: int,
            resume_training: bool,
            device_stats: bool=False,
            extra_callbacks: List[pl.Callback]=None,
            wandb_logger: bool=False
        ) -> None:
        """
        Initializes lightning trainer based on the experiment.yml

        Args:
        -----------
            experiment_name (str):
                Name of the experiment
            experiment_version (str):
                Name of the experiment version
            num_gpus (int):
                Number of GPUs used for training.
            num_epochs (int):
                Number of training epochs
            resume_training (bool):
                resume training where you left off
            device_stats (bool, default=False):
                Monitor device stats.
            extra_callbacks (List[pl.CallBack], default=None):
                List of extra callbacks to add to the Trainer
            wandb_logger (bool, default=False):
                Flag to use also wandb logger.
        """
        self.gpus = num_gpus
        self.epochs = num_epochs
        self.resume_training = resume_training
        self.device_stats = device_stats

        # init paths
        exp_dir = FileHandler.get_experiment_dir(
            experiment=experiment_name,
            version=experiment_version
        )
        self.ckpt_dir = exp_dir

        # set test tube logger
        self.loggers = []
        self.loggers.append(
            pl.loggers.TensorBoardLogger(save_dir=exp_dir)
        )

        if wandb_logger:
            self.loggers.append(
                pl.loggers.WandbLogger(
                    save_dir=exp_dir,
                    project=experiment_name,
                    name=experiment_version,
                    version=experiment_version,
                    log_model=False, # do not log the checkpoints to wandb
                )
            )
        
        # set checkpoint callback
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.ckpt_dir,
            save_top_k=1,
            save_last=True,
            verbose=True, 
            monitor='val_loss',
            mode='min',
        )
        
        # set attributes
        self.callbacks = [ckpt_callback]
        if extra_callbacks is not None:
            self.callbacks += extra_callbacks
            
        # set device stat monitor callback
        if self.device_stats:
            self.callbacks = pl.callbacks.DeviceStatsMonitor()

        self.last_ckpt = None
        if self.resume_training:
            self.last_ckpt = FileHandler.get_model_checkpoint(
                experiment=experiment_name,
                version=experiment_version,
                which=-1
            )
    
        # set logging dir
        self.logging_dir = exp_dir / "tf"

    @classmethod
    def from_conf(
            cls, 
            conf: DictConfig, 
            extra_callbacks: List[pl.Callback]=None, 
            **kwargs
        ) -> pl.Trainer:
        """
        Class method to initialize the class from a config .yml file.

        Args:
        --------
            conf (omegaconf.DictConfig):
                The experiment.yml file (converted into DictConfig).
            extra_callback (List[pl.Callback], default=None):
                List of extra callbacks to add to the Trainer
        
        Returns:
        --------
            pl.Trainer: The SegTrainer instance.
        """
        experiment_name = conf.experiment_name
        experiment_version = conf.experiment_version
        num_gpus = conf.training.num_gpus
        num_epochs = conf.training.num_epochs
        resume_training = conf.training.resume_training
        wandb_logger = conf.training.wandb
        metrics_to_cpu = conf.training.metrics_to_cpu
        device_stats = conf.training.device_stats

        c = cls(
            experiment_name,
            experiment_version,
            num_gpus,
            num_epochs,
            resume_training,
            device_stats,
            extra_callbacks,
            wandb_logger
        )

        return pl.Trainer(
            max_epochs=c.epochs,
            gpus=c.gpus,
            logger=c.loggers,
            callbacks=c.callbacks,
            resume_from_checkpoint=c.last_ckpt,
            profiler="simple",
            move_metrics_to_cpu=metrics_to_cpu,
            **kwargs
        )