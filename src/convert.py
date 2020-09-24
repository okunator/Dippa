import argparse
from omegaconf import OmegaConf 
from src.conf.conf_schema import Schema
from src.conf.config import CONFIG
from src.utils.file_manager import ProjectFileManager

config = OmegaConf.merge(Schema, CONFIG)

if __name__ == '__main__':
    # Add args
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', help="dadadad")
    # parser.add_argument('--view', help="dadadada")
    # args = parser.parse_args()
    
    config.dataset_args.dataset = "kumar"
    fm_kumar = ProjectFileManager.from_conf(config)
    fm_kumar.handle_raw_data(rm_zips=False, overlays=True) # shld make this method static...
    
    config.dataset_args.dataset = "consep"
    fm_consep = ProjectFileManager.from_conf(config)
    fm_consep.handle_raw_data(rm_zips=False, overlays=True)
    
    config.dataset_args.dataset = "pannuke"
    fm_pannuke = ProjectFileManager.from_conf(config)
    fm_pannuke.handle_raw_data(rm_zips=False, overlays=True)
    
    
    