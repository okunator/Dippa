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
            extra_callbacks: List[pl.Callback]=None
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
            extra_callbacks (List[pl.CallBack], default=None):
                List of extra callbacks to add to the Trainer
        """
        # init paths
        exp_dir = FileHandler.get_experiment_dir(
            experiment=experiment_name,
            version=experiment_version
        )

        # set test tube logger
        self.tt_logger = pl.loggers.TestTubeLogger(
            save_dir=FileHandler.result_dir(),
            name=experiment_name,
            version=experiment_version
        )

        self.ckpt_dir = exp_dir

        # set checkpoint callback
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath = self.ckpt_dir,
            save_top_k = 1,
            save_last = True,
            verbose = True, 
            monitor = 'avg_val_loss',
            mode = 'min',
            prefix = ''
        )

        # set gpu monitoring callback
        # gpu_callback = pl.callbacks.GPUStatsMonitor()

        # set attributes
        self.callbacks = [checkpoint_callback] #, gpu_callback]

        if extra_callbacks is not None:
            self.callbacks += extra_callbacks
            
        self.gpus = num_gpus
        self.epochs = num_epochs
        self.resume_training = resume_training

        self.last_ckpt = None
        if self.resume_training:
            self.last_ckpt = FileHandler.get_model_checkpoint(
                experiment=experiment_name,
                version=experiment_version,
                which="last"
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
        Class method to initialize the class from experiment.yml config 
        file

        Args:
        --------
            conf (omegaconf.DictConfig):
                The experiment.yml file (converted into DictConfig).
            extra_callback (List[pl.Callback], default=None):
                List of extra callbacks to add to the Trainer
        
        Returns:
        --------
            The SegTrainer instance.
        """
        experiment_name = conf.experiment_args.experiment_name
        experiment_version = conf.experiment_args.experiment_version
        num_gpus = conf.runtime_args.num_gpus
        num_epochs = conf.runtime_args.num_epochs
        resume_training = conf.runtime_args.resume_training

        c = cls(
            experiment_name,
            experiment_version,
            num_gpus,
            num_epochs,
            resume_training,
            extra_callbacks
        )

        return pl.Trainer(
            max_epochs=c.epochs,
            gpus=c.gpus,
            logger=c.tt_logger,
            callbacks=c.callbacks,
            resume_from_checkpoint=c.last_ckpt,
            log_gpu_memory=True,
            profiler=True #pl.profiler.AdvancedProfiler(),
            # fast_dev_run=False
        )