from src.conf.config import CONFIG
from src.utils.data_writer import PatchWriter, visualize_db_patches

config = CONFIG

def main(conf):
    conf.dataset_args.dataset = "kumar"
    writer_kumar = PatchWriter.from_conf(conf)
    print("Patching kumar dataset")
    writer_kumar.write_dbs() # shld make this method static...
    
    conf.dataset_args.dataset = "consep"
    writer_consep = PatchWriter.from_conf(conf)
    print("Patching consep dataset")
    writer_consep.write_dbs()
    
    conf.dataset_args.dataset = "pannuke"
    conf.patching_args.patch_size = 256
    writer_pannuke = PatchWriter.from_conf(conf)
    print("Patching pannuke dataset")
    writer_pannuke.write_dbs()


if __name__ == '__main__':    
    main(config)