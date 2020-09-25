from omegaconf import OmegaConf 
from src.conf.conf_schema import Schema
from src.conf.config import CONFIG
from src.utils.data_writer import PatchWriter, visualize_db_patches

config = OmegaConf.merge(Schema, CONFIG)


def main(config):
    config.dataset_args.dataset = "kumar"
    writer_kumar = PatchWriter.from_conf(config)
    print("Patching kumar dataset")
    writer_kumar.write_dbs() # shld make this method static...
    
    config.dataset_args.dataset = "consep"
    writer_consep = PatchWriter.from_conf(config)
    print("Patching consep dataset")
    writer_consep.write_dbs()
    
    config.dataset_args.dataset = "pannuke"
    config.patching_args.patch_size = 256
    writer_pannuke = PatchWriter.from_conf(config)
    print("Patching pannuke dataset")
    writer_pannuke.write_dbs()


if __name__ == '__main__':    
    main(config)