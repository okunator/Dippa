import argparse
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from tensorboard import program
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dl.lightning_model import SegModel, plot_metrics
from src.settings import RESULT_DIR
from src.conf.config import CONFIG

config = CONFIG


def main(conf, params):
    # TODO: model builder from arg
    # Use this for testing now
    model = smp.Unet(
        encoder_name="resnext50_32x4d", 
        classes=2
    )
    lightning_model = SegModel.from_conf(model, conf)
                                         
    # Define lightning logger
    tt_logger = TestTubeLogger(
        save_dir=RESULT_DIR,
        name=conf.experiment_args.model_name,
        version=conf.experiment_args.experiment_version
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
    if conf.training_args.resume_training:   
        last_checkpoint_path = lightning_model.fm.model_checkpoint("last")
        trainer = Trainer(
            default_root_dir=conf.experiment_args.experiment_root_dir,
            max_epochs=conf.training_args.num_epochs, 
            gpus=conf.training_args.num_gpus,  
            logger=tt_logger,
            checkpoint_callback=checkpoint_callback,
            resume_from_checkpoint=str(last_checkpoint_path),
            profiler=True,
            progress_bar_refresh_rate=0
        )

    else:
        trainer = Trainer(
            default_root_dir=conf.experiment_args.experiment_root_dir,
            max_epochs=conf.training_args.num_epochs, 
            gpus=conf.training_args.num_gpus,  
            logger=tt_logger,
            checkpoint_callback=checkpoint_callback,
            profiler=True,
            progress_bar_refresh_rate=0
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
        plot_metrics(conf=conf, metric='accuracy', scale='linear', save=True)
        plot_metrics(conf=conf, metric='loss', scale='linear', save=True)
        plot_metrics(conf=conf, metric='TNR', scale='linear', save=True)
        plot_metrics(conf=conf, metric='TPR', scale='linear', save=True)

    
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
    
    
    
