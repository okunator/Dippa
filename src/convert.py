from src.conf.config import CONFIG
from src.utils.file_manager import ProjectFileManager

config = CONFIG

def main(config):
    config.dataset_args.dataset = "kumar"
    fm_kumar = ProjectFileManager.from_conf(config)
    print("Converting kumar dataset")
    fm_kumar.handle_raw_data(rm_zips=False, overlays=True) # shld make this method static...
    
    config.dataset_args.dataset = "consep"
    fm_consep = ProjectFileManager.from_conf(config)
    print("Converting consep dataset")
    fm_consep.handle_raw_data(rm_zips=False, overlays=True)
    
    config.dataset_args.dataset = "pannuke"
    fm_pannuke = ProjectFileManager.from_conf(config)
    print("Converting kumar dataset")
    fm_pannuke.handle_raw_data(rm_zips=False, overlays=True)


if __name__ == '__main__':
    main(config)
    
    
    