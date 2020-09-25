import torch
import segmentation_models_pytorch as smp
from torch import nn
from src.conf.config import CONFIG
from src.dl.inferer import Inferer
from src.dl.lightning_model import SegModel

config = CONFIG

def main(params):
    pass


if name == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="specify the model you want to use", default=0)
    parser.add_argument('--tensorboard', help="Use tensorboard", default=True)
    parser.add_argument('--auto_bs', help="Find a batch size that fits to mem automagically", default=False)
    parser.add_argument('--plots', help="Save the metrics plots from training", default=True)
    parser.add_argument('--test', help="Run model on test set and report training metrics", default=True)
    args = parser.parse_args()
    main(config, args)