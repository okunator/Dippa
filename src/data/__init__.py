from pathlib import Path
from typing import Union, Dict, Tuple

from .downloaders.pannuke_downloader import PANNUKE
# from .downloaders.consep_downloader import CONSEP
# from .downloaders.kumar_downloader import KUMAR

from .writers.hdf5_writer import HDF5Writer
from .writers.zarr_writer import ZarrWriter