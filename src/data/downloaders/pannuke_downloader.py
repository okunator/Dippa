import pooch
from pathlib import Path
from typing import Union

from .adhoc import handle_pannuke
from .base_downloader import BaseDownloader


HASHES = {
    "fold_1":"6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b",
    "fold_2":"5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07",
    "fold_3":"c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d"
}


class PANNUKE(BaseDownloader):
    def __init__(
            self, 
            save_dir: Union[str, Path], 
            fold: int, 
            phase: str
        ) -> None:
        """
        Fetches a single fold of the pannuke dataset from: 
        https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke.
        

        Pannuke papers:
        ---------------
        Gamper, J., Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N.
        (2019). PanNuke: an open pan-cancer histology dataset for nuclei
        instance segmentation and classification. 
        In European Congress on Digital Pathology (pp. 11–19).

        Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram,
        S., Azam, A., Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset 
        Extension, Insights and Baselines. arXiv preprint 
        arXiv:2003.10778.

        Args:
        ----------
            save_dir (str, or Path obj):
                directory where the data is downloaded
            fold (int):
                the pannuke fold number
            phase (int):
                One of ("train", "test")
        """
        assert phase in ("train", "test")
        assert 1 <= fold <= 3, f"fold {fold}. Only three folds in the data"
        super(PANNUKE, self).__init__(
            save_dir=save_dir,
            dataset_name="pannuke",
            handler_func=handle_pannuke,
            fold=fold,
            phase=phase
        )

        # Create pooch and set downloader
        self.downloader = pooch.HTTPDownloader(progressbar=True)
        path = Path(self.save_dir / "pannuke/original")
        self.POOCH = pooch.create(
            path=pooch.os_cache(path.as_posix()),
            base_url="https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/",
            retry_if_failed=2,
            registry={
                f"fold_{fold}.zip": HASHES[f"fold_{fold}"],
            }
        )

    def download(self) -> None:
        return self.POOCH.fetch(
            fname=f"fold_{self.kwargs['fold']}.zip", 
            processor=self.processor, 
            downloader=self.downloader
        )
