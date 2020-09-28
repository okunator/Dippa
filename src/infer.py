import argparse
import torch
import segmentation_models_pytorch as smp
from torch import nn
from src.conf.config import CONFIG
from src.dl.inferer import Inferer
from src.dl.lightning_model import SegModel

config = CONFIG

def main(config, params):
    model = smp.Unet(
        encoder_name="resnext50_32x4d", 
        classes=2
    )

    lightning_model = SegModel.from_conf(model, config)
    ckpt = lightning_model.fm.model_checkpoint(config.inference_args.model_weights)
    checkpoint = torch.load(ckpt, map_location = lambda storage, loc : storage)
    lightning_model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    inf = Inferer.from_conf(lightning_model, config)
    infobj = inf.run()
    
    print("Running post-processing")
    inf.post_process()
    
    print("Running benchmarks")
    score_df = inf.benchmark(save=True)
    inf.plot_overlays(ixs=[12], save=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="specify the model you want to use", default=0)
    parser.add_argument('--tensorboard', help="Use tensorboard", default=True)
    parser.add_argument('--auto_bs', help="Find a batch size that fits to mem automagically", default=False)
    parser.add_argument('--plots', help="Save the metrics plots from training", default=True)
    parser.add_argument('--test', help="Run model on test set and report training metrics", default=True)
    args = parser.parse_args()
    main(config, args)