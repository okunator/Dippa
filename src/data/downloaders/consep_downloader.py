import pooch
from pathlib import Path
from typing import Union, Dict

from .adhoc import handle_consep
from .base_downloader import BaseDownloader


HASH = "23eb8a717551827b4d42642b08bd64370810acf6e6e73a399182c1c915dfe82a"


class CONSEP(BaseDownloader):
    def __init__(
            self, 
            save_dir: Union[str, Path], 
            convert_classes: bool=True
        ) -> None:
        """
        Fetches the CoNseP dataset from:
        https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/.

        CoNSeP dataset paper:
        --------------
        S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, 
        J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation
        and Classification of Nuclei in Multi-Tissue Histology Images." 
        Medical Image Analysis, Sept. 2019. 

        Args:
        --------
            save_dir (str, or Path obj):
                directory where the data is downloaded
            convert_classes (bool, default=True):
                Convert CoNSeP dataset classes same way they did in 
                the paper
        """
        super(CONSEP, self).__init__(
            save_dir=save_dir, 
            dataset_name="consep", 
            handler_func=handle_consep,
            convert_classes=convert_classes
        )

        self.downloader = pooch.HTTPDownloader(progressbar=True)
        path = Path(self.save_dir / "consep/original/")
        self.POOCH = pooch.create(
            path=pooch.os_cache(path.as_posix()),
            base_url="https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/",
            retry_if_failed=2,
            registry={
                "consep.zip": HASH,
            }
        )

    def download(self) -> None:
        return self.POOCH.fetch(
            fname="consep.zip", 
            processor=self.processor, 
            downloader=self.downloader
        )