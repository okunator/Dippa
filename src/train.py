import argparse
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from tensorboard import program
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf 
from src.conf.conf_schema import Schema
from src.conf.config import CONFIG
from src.dl.lightning_model import SegModel

config = OmegaConf.merge(Schema, CONFIG)


def main(config, params):
    # TODO: model builder from arg
    # Use this for testing now
    model = smp.Unet(
        encoder_name="resnext50_32x4d", 
        classes=2
    )
    lightning_model = SegModel.from_conf(model, config)
    # Define lightning logger
    tt_logger = TestTubeLogger(
        save_dir=config.experiment_args.experiment_root_dir,
        name=config.experiment_args.model_name,
        version=config.experiment_args.experiment_version
    )

    checkpoint_dir = (
        Path(tt_logger.save_dir)
        / tt_logger.experiment.name
        / f"version_{tt_logger.experiment.version}"
    )

    # Define checkpoint dir
    checkpoint_callback = ModelCheckpoint(
        filepath = str(checkpoint_dir),
        save_top_k = 1,
        save_last = True,
        verbose = True, 
        monitor = 'avg_val_loss',
        mode = 'min',
        prefix = ''
    )

    # Resume training?
    if config.training_args.resume_training:   
        last_checkpoint_path = lightning_model.fm.model_checkpoint("last")
        trainer = Trainer(
            default_root_dir=config.experiment_args.experiment_root_dir,
            max_epochs=config.training_args.num_epochs, 
            gpus=config.training_args.num_gpus,  
            logger=tt_logger,
            checkpoint_callback=checkpoint_callback,
            resume_from_checkpoint=str(last_checkpoint_path),
            profiler=True,
            show_progress_bar=False
        )

    else:
        trainer = Trainer(
            default_root_dir=config.experiment_args.experiment_root_dir,
            max_epochs=config.training_args.num_epochs, 
            gpus=config.training_args.num_gpus,  
            logger=tt_logger,
            checkpoint_callback=checkpoint_callback,
            profiler=True,
            show_progress_bar=False
        )
    
    # Launch tensorboard. Dunno if this even works
    if params.tensorboard:
        log_dir = (
            Path(tt_logger.save_dir)
            / tt_logger.experiment.name
            / f"version_{tt_logger.experiment.version}"
            / "tf"
        )

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir.as_posix()])
        url = tb.launch()
    
    # find the batch size automatically
    # skip this for now since a hard-to-get-rid-off pickling error
    # if params.auto_bs:
    #     new_batch_size = trainer.scale_batch_size(lightning_model)
    #     lightning_model.batch_size = new_batch_size
        
    trainer.fit(lightning_model)

    if params.plots:
        plot_metrics(conf=config, metric='accuracy', scale='linear', save=True)
        plot_metrics(conf=config, metric='loss', scale='linear', save=True)
        plot_metrics(conf=config, metric='TNR', scale='linear', save=True)
        plot_metrics(conf=config, metric='TPR', scale='linear', save=True)

    
    if params.test:
        trainer.test(
            model=lightning_model,
            ckpt_path = lightning_model.fm.model_checkpoint("best").as_posix()
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="specify the model you want to use", default=0)
    parser.add_argument('--tensorboard', help="Use tensorboard", default=True)
    parser.add_argument('--auto_bs', help="Find a batch size that fits to mem automagically", default=False)
    parser.add_argument('--plots', help="Save the metrics plots from training", default=True)
    parser.add_argument('--test', help="Run model on test set and report training metrics", default=True)
    args = parser.parse_args()
    main(config, args)
    
    
    
