import logging
import pooch
from pathlib import Path
from typing import Union, Dict

from .adhoc import handle_pannuke
from .base_downloader import BaseDownloader


class KUMAR(BaseDownloader):
    def __init__(self, save_dir: Union[str, Path], fold: int, phase: str) -> None:
        """
        Fetches The Kumar dataset from:
        Train: https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA/view
        Test: https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view

        Kumar paper:
        ---------------
        N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, 
        "A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology,"
        in IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550-1560, July 2017

        Args:
        ----------
            save_dir (str, or Path obj):
                directory where the data is downloaded
            fold (int):
                the pannuke fold number
            phase (int):
                One of ("train", "test")
        """
        pass