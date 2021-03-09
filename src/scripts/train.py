import src.dl.lightning as lightning
from src.config import CONFIG
from pathlib import Path

def main(conf):

    lightning_model = lightning.SegModel.from_conf(CONFIG)
    trainer = lightning.SegTrainer.from_conf(CONFIG)

    lightning_model.train_data = Path("/home/leos/Dippa/datasets/patches/tests/zarr/train_consep2.zarr")
    lightning_model.valid_data = Path("/home/leos/Dippa/datasets/patches/tests/zarr/test_consep.zarr")
    lightning_model.test_data = Path("/home/leos/Dippa/datasets/patches/tests/zarr/test_consep.zarr")

    trainer.fit(lightning_model)

    trainer.test(
        model=lightning_model,
        ckpt_path=lightning_model.fm.get_model_checkpoint("last").as_posix()
    )


if __name__ == '__main__':
    main(CONFIG)
    
    
    
