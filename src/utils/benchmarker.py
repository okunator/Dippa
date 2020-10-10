import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Union
from omegaconf import DictConfig
from pathos.multiprocessing import ThreadPool as Pool
from src.utils.file_manager import ProjectFileManager
from src.img_processing.process_utils import remap_label
from src.img_processing.viz_utils import draw_contours
from src.metrics.metrics import (
    PQ, AJI, AJI_plus, DICE2, split_and_merge
)


class Benchmarker(ProjectFileManager):
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig) -> None:

        super(Benchmarker, self).__init__(dataset_args, experiment_args)

    def __sample_idxs(self, n: int = 25) -> np.ndarray:
        assert n <= len(self.images), "Cannot sample more ints than there are images in dataset"
        # Dont plot more than 50 pannuke images
        n = 50 if self.dataset == "pannuke" and n > 50 else n
        return np.random.randint(low=0, high=len(self.images), size=n)

    def compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """
        Computes metrics for one (inst_map, gt_mask) pair.
        Args:
            true (np.ndarray): ground truth annotations
            pred (np.ndarray): instance map
        Returns:
            A Dict[str, float] of the metrics
        """
        if len(np.unique(true)) > 1:
            pq = PQ(remap_label(true), remap_label(pred))
            aji = AJI(remap_label(true), remap_label(pred))
            aji_p = AJI_plus(remap_label(true), remap_label(pred))
            dice2 = DICE2(remap_label(true), remap_label(pred))
            splits, merges = split_and_merge(remap_label(true), remap_label(pred))

            return {
                "AJI": aji,
                "AJI_plus": aji_p,
                "DICE2": dice2,
                "PQ": pq["pq"],  # panoptic quality
                "SQ": pq["sq"],  # segmentation quality
                "DQ": pq["dq"],  # Detection quality i.e. F1-score
                "inst_Sensitivity": pq["sensitivity"],
                "inst_Precision": pq["precision"],
                "splits": splits,
                "merges": merges
            }

    def benchmark(self,
                  inst_maps: List[np.ndarray],
                  gt_masks: List[np.ndarray],  
                  save: bool = False) -> pd.DataFrame:
        """
        Run benchmarking metrics for all of the files in the dataset
        Note that the inst_maps and gt_masks need to be sorted so they align
        when computing metrics.
        
        Args:
            inst_maps (List[np.ndarray]): a list of the inst_maps in order
            gt_masks (List[np.ndarray]): a list of the ground truth masks in order
            save (bool): save the result table to .csv file
        Returns:
            a pandas dataframe of the metrics. Samples are rows and metrics are columns:
            _____________________
            |sample|PQ|SQ|DQ|AJI|
            |img1  |.5|.4|.6|.6 |
            |img2  |.5|.4|.6|.6 |
            
        """
        assert isinstance(inst_maps, list), f"inst_maps: {type(inst_maps)} is not a list of inst_maps"
        assert isinstance(gt_masks, list), f"inst_maps: {type(gt_masks)} is not a list of inst_maps"

        params_list = list(zip(gt_masks, inst_maps))

        with Pool() as pool:
            metrics = pool.starmap(self._compute_metrics, params_list)

        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            self.metrics[f"{fn}_metrics"] = metrics[i]

        score_df = pd.DataFrame(self.metrics).transpose()
        score_df.loc["averages_for_the_set"] = score_df.mean(axis=0)

        if self.dataset == "pannuke":
            df = score_df.rename_axis("fn").reset_index()
            td = {f"{tis}_avg": df[df.fn.str.contains(f"{tis}")].mean(
                axis=0) for tis in self.pannuke_tissues}
            score_df = pd.concat([score_df, pd.DataFrame(td).transpose()])

        if save:
            result_dir = Path(self.experiment_dir / "benchmark_results")
            self.create_dir(result_dir)
            s = "smoothed" if self.smoothed else ""
            t = "tta" if self.tta else ""
            score_df.to_csv(Path(result_dir / f"{s}_{t}_benchmark_result.csv"))

        return score_df

    def plot_segmentations(self,                  
                           inst_maps: List[np.ndarray],
                           gt_masks: List[np.ndarray],
                           save: bool = False) -> None:
        """
        Plot all the binary segmentations after running post_processing.
        Args:
            inst_maps (List[np.ndarray]): a list of the inst_maps in order
            gt_masks (List[np.ndarray]): a list of the ground truth masks in order
        """
        assert self.inst_maps, f"{self.inst_maps}, No instance maps found. Run post_processing first!"
        idxs = self.__sample_idxs(15)
        images = np.asarray(self.images)[idxs] if self.dataset == "pannuke" else self.images
        gt_masks = np.asarray(self.gt_masks)[idxs] if self.dataset == "pannuke" else self.gt_masks

        fig, axes = plt.subplots(len(images), 4, figsize=(65, len(images)*12))
        fig.tight_layout(w_pad=4, h_pad=4)
        for j, path in enumerate(images):
            fn = self.__get_fn(path)
            im = self.read_img(images[j])
            gt = self.read_mask(gt_masks[j])
            inst_map = self.inst_maps[f"{fn}_inst_map"]
            nuc_map = self.soft_maps[f"{fn}_nuc_map"]
            axes[j][0].imshow(im, interpolation="none")
            axes[j][0].axis("off")
            axes[j][1].imshow(nuc_map, interpolation="none")
            axes[j][1].axis("off")
            axes[j][2].imshow(inst_map, interpolation="none")
            axes[j][2].axis("off")
            axes[j][3].imshow(gt, interpolation="none")
            axes[j][3].axis("off")

    def plot_contour_overlays(self,
                              inst_maps: List[np.ndarray],
                              gt_masks: List[np.ndarray],
                              ixs: Union[List[int], int] = -1,
                              save: bool = False) -> None:
        """
        Plot segmentation result and ground truth overlaid on the original image side by side
        Args:
            ixs (List or int): list of the indexes of the image files in Inferer.images. 
                               default = -1 means all images in the data fold are plotted. If 
                               dataset = "pannuke" and ixs = -1, then 25 random images are sampled.
            save (bool): Save the plots
        """
        assert self.inst_maps, f"{self.inst_maps}, No instance maps found. Run post_processing first!"

        message = (
            f"param ixs: {ixs} needs to be a list of ints in"
            " the range of number of images in total or -1."
            f" number of images = {len(self.images)}"
        )

        if isinstance(ixs, List):
            assert all(i <= len(self.images) for i in ixs), message
        elif isinstance(ixs, int):
            assert ixs == -1, message

        if ixs == -1:
            idxs = np.array(range(len(self.images)))
        elif ixs == -1 and self.dataset == "pannuke":
            idxs = self.__sample_idxs(25)
        else:
            idxs = np.array(ixs)

        images = np.asarray(self.images)[idxs]
        gt_masks = np.asarray(self.gt_masks)[idxs]

        fig, axes = plt.subplots(
            len(images), 2, figsize=(40, len(images)*20), squeeze=False
        )

        for j, path in enumerate(images):
            fn = self.__get_fn(path)
            im = self.read_img(path)
            gt = self.read_mask(gt_masks[j])
            inst_map = self.inst_maps[f"{fn}_inst_map"]
            _, im_gt = draw_contours(gt, im)
            _, im_res = draw_contours(inst_map, im)

            axes[j, 0].set_title(f"Ground truth: {fn}", fontsize=30)
            axes[j, 0].imshow(im_gt, interpolation='none')
            axes[j, 0].axis('off')
            axes[j, 1].set_title(f"Segmentation result: {fn}", fontsize=30)
            axes[j, 1].imshow(im_res, cmap='gray', interpolation='none')
            axes[j, 1].axis('off')

        fig.tight_layout(w_pad=4, h_pad=10)

        if save:
            plot_dir = Path(self.experiment_dir / "inference_plots")
            self.create_dir(plot_dir)

            s = "smoothed" if self.smoothed else ""
            t = "tta" if self.tta else ""
            fig.savefig(Path(plot_dir / f"{fn}_{t}_{s}result.png"))
