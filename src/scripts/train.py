import src.dl.lightning as lightning
from src.config import CONFIG

def main(conf):

    lightning_model = lightning.SegModel.from_conf(CONFIG)
    trainer = lightning.SegTrainer.from_conf(CONFIG)

    trainer.fit(lightning_model)

    trainer.test(
        model=lightning_model,
        ckpt_path=lightning_model.fm.get_model_checkpoint("last").as_posix()
    )


if __name__ == '__main__':
    main(CONFIG)
    
    
    
