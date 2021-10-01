import importlib
import albumentations as A
from typing import List, Tuple, Optional
from torch.utils.data import Dataset


ds = importlib.import_module("src.data.datasets")
ds_vars = vars(ds)


class DatasetBuilder:
    def __init__(
            self,
            decoder_aux_branch: str=None,
            **kwargs
        ) -> None:
        """
        Initializes the train & test time datsets based on the
        experiment.yml

        Args:
        -----------
            deocder_aux_branch (str, default=None):
                The type of the auxiliary branch. If None, the unet 
                dataset is used in the dataloader.
        """

        self.ds_name = "unet"
        if decoder_aux_branch is not None:
            self.ds_name = decoder_aux_branch

        assert self.ds_name in ("hover", "dist", "contour", "unet", "basic")

    def get_augs(
            self, 
            augs_list: Optional[List[str]]=None,
            input_HW: Tuple[int, int]=(256, 256)
        ) -> A.Compose:
        """
        Compose the augmentations into an augmentation pipeline

        Args:
        -----------
            augs_list (List[str], optional, default=None): 
                List of augmentations. If None, then no augmentations 
                are used
            input_HW (Tuple[int, int], default=(256, 256)):
                Height & width of the input matrices

        Returns:
        -----------
            A.Compose:  albumentations Compose object containing the 
            augmentations.
        """
        kwargs={"height": input_HW[0], "width": input_HW[1]}

        aug_list = []
        if augs_list is not None:
            aug_list = [
                ds_vars[ds.AUGS_LOOKUP[aug_name]](**kwargs) 
                for aug_name in augs_list
            ] 
        aug_list.append(ds.to_tensor())

        return ds.compose(aug_list)

    @classmethod
    def set_train_dataset(
            cls,
            fname: str,
            decoder_aux_branch: str=None,
            augmentations: List[str]=None,
            normalize_input: bool=True,
            rm_touching_nuc_borders: bool=False,
            edge_weights: bool=False,
            type_branch: bool=True,
            semantic_branch: bool=False,
        ) -> Dataset:
        """
        Init the train dataset.

        Args:
        ------------
            fname (str):
                path to the database file
            decoder_aux_branch (str, default=None):
                The type of the auxiliary branch. If None, the unet 
                dataset is used in the dataloader.
            augmentations (List[str]): 
                List of augmentations to be used for training
                One of: "rigid", "non_rigid", "hue_sat", "blur", 
                "non_spatial", "random_crop", "center_crop", "resize"
            normalize_input (bool, default=True):
                If True, minmax normalization for the input 
                images isapplied.
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
            edge_weights (bool, default=False):
                If True, each dataset iteration will create weight maps
                for the nuclear edges. This can be used to penalize
                nuclei edges in cross-entropy based loss functions.
            type_branch (bool, default=False):
                If cell type branch is included in the model, this arg
                signals that the cell type annotations are included per
                each dataset iter. Given that these annotations exist in
                db
            semantic_branch (bool, default=False):
                If the model contains a semnatic area branch, this arg 
                signals that the area annotations are included per each 
                dataset iter. Given that these annotations exist in db

        Returns:
        ------------
            torch.utils.data.Dataset: Initialized torch Dataset object 
        """
        c = cls(decoder_aux_branch)
        aug = c.get_augs(augmentations)
        dataset  = ds.DS_LOOKUP[c.ds_name]

        return ds_vars[dataset](
            fname=fname,
            transforms=aug,
            normalize_input=normalize_input,
            rm_touching_nuc_borders=rm_touching_nuc_borders,
            edge_weights=edge_weights,
            type_branch=type_branch,
            semantic_branch=semantic_branch
        )

    @classmethod
    def set_test_dataset(
            cls,
            fname: str,
            decoder_aux_branch: str=None,
            normalize_input: bool=True,
            rm_touching_nuc_borders: bool=False,
            edge_weights: bool=False,
            type_branch: bool=True,
            semantic_branch: bool=False,
        ) -> Dataset:
        """
        Init the test dataset. No augmentations used. Only ndarray to 
        tensor conversion.

        Args:
        ------------
            fname (str): 
                path to the database file
            decoder_aux_branch (str, default=None):
                The type of the auxiliary branch. If None, the unet 
                dataset is used in the dataloader.
            normalize_input (bool, default=True):
                If True, minmax normalization for the input images
                is applied.
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
            edge_weights (bool, default=False):
                If True, each dataset iteration will create weight maps
                for the nuclear edges. This can be used to penalize
                nuclei edges in cross-entropy based loss functions.
            type_branch (bool, default=False):
                If cell type branch is included in the model, this arg
                signals that the cell type annotations are included per
                each dataset iter. Given that these annotations exist in
                db
            semantic_branch (bool, default=False):
                If the model contains a semnatic area branch, this arg 
                signals that the area annotations are included per each 
                dataset iter. Given that these annotations exist in db

        Returns:
        ------------
            torch.utils.data.Dataset: Initialized torch Dataset object 
        """
        c = cls(decoder_aux_branch)
        aug = c.get_augs()
        dataset  = ds.DS_LOOKUP[c.ds_name]

        return ds_vars[dataset](
            fname=fname,
            transforms=aug,
            normalize_input=normalize_input,
            rm_touching_nuc_borders=rm_touching_nuc_borders,
            edge_weights=edge_weights,
            type_branch=type_branch,
            semantic_branch=semantic_branch
        )