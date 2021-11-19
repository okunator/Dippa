import argparse
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, Any

from src.dl.lightning import SegExperiment, SegTrainer
from src.config import get_conf, merge_conf, CONFIG
from src.data.data_modules.utils import prepare_datamodule
from src.dl.models import MultiTaskSegModel


def main(conf: DictConfig, params: Dict[str, Any]) -> None:
    """
    CLI training script

    Args:
    ---------
        conf (DictConfig):
            Config dictionary
        params (Dict[str, Any]):
            Other params not defined in the config dict.
    """
    conf = CONFIG
    if params.dataconf and params.trainconf and params.modelconf:
        trainconf = get_conf(params.trainconf)
        dataconf = get_conf(params.dataconf)
        modelconf = get_conf(params.modelconf)
        conf = merge_conf(trainconf, dataconf, modelconf)
        
    data_kwargs = {}
    data_kwargs["name"] = params.dataset
    if params.dataset == "custom":
        if not (params.train_db or params.test_db):
            raise ValueError(f"""
                `--test_db` and `--train_db` args are required, when
                `--dataset` is set to "custom". """
            )
        data_kwargs["train_db_path"] = params.train_db
        data_kwargs["test_db_path"] = params.test_db
        
    elif params.dataset in ("consep", "pannuke"):
        db_dir = None
        if params.train_db:
            db_dir = Path(params.train_db).parent

        download_dir = None
        if params.download_dir:
            download_dir = params.download_dir
            
        data_kwargs["download_dir"] = download_dir
        data_kwargs["database_dir"] = db_dir
        
    datamodule = prepare_datamodule(**data_kwargs, conf=conf)
        
    if conf.training.resume_training:
        lit_model = SegExperiment.from_experiment(
            name=conf.experiment_name,
            version=conf.experiment_version,
        )
    else:
        model = MultiTaskSegModel.from_conf(conf)
        lit_model = SegExperiment.from_conf(model, conf)
        
    # init trainer
    extra_callbacks = []
    if conf.training.wandb:
        from src.dl.lightning.callbacks import WandbImageCallback

        classes = datamodule.class_dicts
        extra_callbacks.append(
            WandbImageCallback(classes[0], classes[1])
        )

    trainer = SegTrainer.from_conf(
        conf=conf, extra_callbacks=extra_callbacks
    )

    # train
    trainer.fit(lit_model, datamodule=datamodule)


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
        '--dataconf',
        type=str,
        default=None,
        help="File path to .yml conf file specifying datamodule args."       
    )
    parser.add_argument(
        '--trainconf',
        type=str,
        default=None,
        help="File path to .yml file specifying training arguments."
    )
    parser.add_argument(
        '--modelconf',
        type=str,
        default=None,
        help="File path to .yml file specifying model spec."
        
    )

    args = parser.parse_args()
    main(CONFIG, args)
    
    
    
