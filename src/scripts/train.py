import argparse
from pathlib import Path
from omegaconf import OmegaConf

import src.dl.lightning as lightning
from src.config import CONFIG
from src.utils import FileHandler
from src.data import (
    CustomDataModule, PannukeDataModule, ConsepDataModule
)


def main(conf, extra_params):
    # get conf
    conf = OmegaConf.load(extra_params.yml) if extra_params.yml else CONFIG
    
    # set the datamodule
    if extra_params.dataset == "custom":
        assert extra_params, (
            "If dataset == 'custom', the train/test db paths are required"
        )

        datamodule = CustomDataModule(
            train_db_path=extra_params.train_db,
            test_db_path=extra_params.test_db,
            augmentations=conf.training_args.input_args.augmentations,
            normalize=conf.training_args.input_args.normalize_input,
            aux_branch=conf.model_args.decoder_branches.aux_branch,
            type_branch=conf.model_args.decoder_branches.type_branch,
            sem_branch=conf.model_args.decoder_branches.sem_branch,
            edge_weights=conf.training_args.input_args.edge_weights,
            rm_touching_nuc_borders=conf.training_args.input_args.rm_overlaps,
            batch_size=conf.runtime_args.batch_size,
            num_workers=conf.runtime_args.num_workers
        )
    elif extra_params.dataset == "pannuke":
        db_dir = None
        if extra_params.train_db:
            db_dir = Path(extra_params.train_db).parent

        download_dir = None
        if extra_params.download_dir:
            download_dir = extra_params.download_dir

        datamodule = PannukeDataModule(
            database_type=conf.runtime_args.db_type,
            augmentations=conf.training_args.input_args.augmentations,
            normalize=conf.training_args.input_args.normalize_input,
            aux_branch=conf.model_args.decoder_branches.aux_branch,
            type_branch=conf.model_args.decoder_branches.type_branch,
            edge_weights=conf.training_args.input_args.edge_weights,
            rm_touching_nuc_borders=conf.training_args.input_args.rm_overlaps,
            batch_size=conf.runtime_args.batch_size,
            num_workers=conf.runtime_args.num_workers,
            database_dir=db_dir,
            download_dir=download_dir
        )
    elif extra_params.dataset == "consep":
        db_dir = None
        if extra_params.train_db:
            db_dir = Path(extra_params.train_db).parent
        
        download_dir = None
        if extra_params.download_dir:
            download_dir = extra_params.download_dir

        datamodule = ConsepDataModule(
            database_type=conf.runtime_args.db_type,
            augmentations=conf.training_args.input_args.augmentations,
            normalize=conf.training_args.input_args.normalize_input,
            aux_branch=conf.model_args.decoder_branches.aux_branch,
            type_branch=conf.model_args.decoder_branches.type_branch,
            edge_weights=conf.training_args.input_args.edge_weights,
            rm_touching_nuc_borders=conf.training_args.input_args.rm_overlaps,
            batch_size=conf.runtime_args.batch_size,
            num_workers=conf.runtime_args.num_workers,
            database_dir=db_dir,
            download_dir=download_dir
        )


    if conf.runtime_args.resume_training:
        lightning_model = lightning.SegModel.from_experiment(
            name=conf.experiment_args.experiment_name,
            version=conf.experiment_args.experiment_version,
        )
    else:
        # init model
        lightning_model = lightning.SegModel.from_conf(conf)
        

    # init trainer
    extra_callbacks = []
    if conf.runtime_args.wandb:
        extra_callbacks.append(lightning.WandbImageCallback())

    trainer = lightning.SegTrainer.from_conf(
        conf=conf, extra_callbacks=extra_callbacks
    )

    # train
    trainer.fit(lightning_model, datamodule=datamodule)

    # run tests
    if extra_params.run_testing:
        trainer.test(
            model=lightning_model,
            ckpt_path=FileHandler.get_model_checkpoint(
                experiment=conf.experiment_args.experiment_name,
                version=conf.experiment_args.experiment_version,
                which="last"
            ),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_db',
        type=str,
        default=None,
        help=(
            "The path to the training db. Needed for custom datasets.",
            "If the pannuke or consep dataset is used and this param is given",
            "a new database will be written in the parent dir of this path"
        )
    )
    parser.add_argument(
        '--test_db',
        type=str,
        default=None,
        help="The path to the testing db. Needed for custom datasets."
    )
    parser.add_argument(
        '--dataset',
        help="The datamodule to be used",
        choices=["pannuke", "consep", "custom"],
        type=str,
        default="pannuke"
    )
    parser.add_argument(
        '--download_dir',
        help=(
            "If you need to download either pannuke or consep datasets",
            "This is the directory where it will be downloaded"
        )
    )
    parser.add_argument(
        '--yml',
        type=str,
        default=None,
        help=(
            "File path to train an experiment.yml file.",
            "Use if you want to train many models at the same time."
        )
    )
    parser.add_argument(
        '--run_testing',
        help="Run metrics against the test set.",
        type=bool,
        default=False
    )
    args = parser.parse_args()
    main(CONFIG, args)
    
    
    
