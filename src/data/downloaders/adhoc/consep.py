from pathlib import Path
from distutils import dir_util, file_util

from src.utils import FileHandler


def handle_consep(orig_dir: Path,
                  imgs_train_dir: Path,
                  anns_train_dir: Path,
                  imgs_test_dir: Path,
                  anns_test_dir: Path) -> None:
    """
    Moves the consep files to the "consep" directory and creates train and testing
    folders for the data.

    Args:
    -----------
        orig_dir (Path): 
            The path where the .zip files are located
        imgs_train_dir (Path): 
            Path to the directory where the training images are saved
        anns_train_dir (Path): 
            Path to the directory where the training gt annotations are saved
        imgs_test_dir (Path): 
            Path to the directory where the testing images are saved
        anns_test_dir (Path): 
            Path to the directory where the testing gt annotations are saved
    """
    FileHandler.create_dir(anns_train_dir)
    FileHandler.create_dir(imgs_train_dir)
    FileHandler.create_dir(anns_test_dir)
    FileHandler.create_dir(imgs_test_dir)

    for item in orig_dir.iterdir():
        if item.is_dir() and item.name == "CoNSeP":
            for d in item.iterdir():
                if d.is_dir() and d.name == "Test":
                    for g in d.iterdir():
                        if g.name == "Images":
                            dir_util.copy_tree(str(g), str(imgs_test_dir))
                        elif g.name == "Labels":
                            dir_util.copy_tree(str(g), str(anns_test_dir))
                elif d.is_dir() and d.name == "Train":
                    for g in d.iterdir():
                        if g.name == "Images":
                            dir_util.copy_tree(str(g), str(imgs_train_dir))
                        elif g.name == "Labels":
                            dir_util.copy_tree(str(g), str(anns_train_dir))
                    