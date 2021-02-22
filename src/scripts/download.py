# from src.utils.data_downloader import Downloader
# from src.conf.config import CONFIG

# config = CONFIG

# def main(conf):
#     print("Downloading datasets")
#     Downloader.download_datasets()

#     conf.dataset_args.dataset = "kumar"
#     downloader_kumar = Downloader.from_conf(conf)
#     print("Converting kumar dataset")
#     # shld make this method static...
#     downloader_kumar.handle_raw_data(rm_zips=False, overlays=True)

#     conf.dataset_args.dataset = "consep"
#     downloader_consep = Downloader.from_conf(conf)
#     print("Converting consep dataset")
#     downloader_consep.handle_raw_data(rm_zips=False, overlays=True)

#     conf.dataset_args.dataset = "pannuke"
#     downloader_pannuke = Downloader.from_conf(conf)
#     print("Converting pannuke dataset")
#     downloader_pannuke.handle_raw_data(rm_zips=False, overlays=True)

# if __name__ == '__main__':
#     main(config)
