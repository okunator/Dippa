import argparse
from pathlib import Path
from omegaconf import OmegaConf

import src.dl.lightning as lightning
from src.config import CONFIG


def main(conf, extra_params):
    conf = OmegaConf.load(extra_params.yml) if extra_params.yml else CONFIG
    lightning_model = lightning.SegModel.from_conf(conf, **vars(extra_params))
    trainer = lightning.SegTrainer.from_conf(conf)
    
    trainer.fit(lightning_model)

    trainer.test(
        model=lightning_model,
        ckpt_path=lightning_model.fm.get_model_checkpoint("last").as_posix()
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_db',
        help="File path to train data db.",
        type=str,
        default=None
    )
    parser.add_argument(
        '--test_db',
        help="File path to test db",
        type=str,
        default=None
    )
    parser.add_argument(
        '--valid_db',
        help="File path to validation data db",
        type=str,
        default=None
    )
    parser.add_argument(
        '--n_classes',
        help="number of classes in train data db",
        type=int,
        default=None
    )
    parser.add_argument(
        '--yml',
        help="File path to train the experiment config .yml file.",
        type=str,
        default=None
    )
    args = parser.parse_args()
    main(CONFIG, args)
    
    
    
