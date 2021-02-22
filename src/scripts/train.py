import src.dl.lightning as lightning
from src.config import CONFIG

config = CONFIG


def main(conf):
    # get the experiment.yml config file
    config = CONFIG

    # Insert the model to pytorch lightning framework. (Simplifies the training and other stuff)
    lightning_model = lightning.SegModel.from_conf(config)
    trainer = lightning.SegTrainer.from_conf(config)

    trainer.fit(lightning_model)

    trainer.test(
        model=lightning_model,
        ckpt_path=lightning_model.fm.get_model_checkpoint("last").as_posix()
    )


if __name__ == '__main__':
    main(config)
    
    
    
