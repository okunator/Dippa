import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Union
from collections import OrderedDict
from pathos.multiprocessing import ThreadPool as Pool
from tqdm import tqdm

from src.utils.process_utils import remap_label, get_type_instances
from .metrics import PQ, AJI, AJI_plus, DICE2, split_and_merge


class Benchmarker:
    def __init__(self) -> None:
        """
        A classs for benchmarking segmentations against their ground truth annotations

        """
        self.inst_metrics = OrderedDict()
        self.type_metrics = OrderedDict()

    def compute_metrics(self, true_pred: List[np.ndarray]) -> Dict[str, float]:
        """
        Computes metrics for one (inst_map, gt_mask) pair.

        Args:
            true_pred (List[np.ndarray]): 
                Ground truth annotations in true_pred[0] and corresponding 
                predicted instance map in true_pred[1]

        Returns:
            A Dict[str, float] of the metrics
        """

        name = true_pred[0]
        true = true_pred[1]
        pred = true_pred[2]
        if len(np.unique(true)) > 1:
            pq = PQ(remap_label(true), remap_label(pred))
            aji = AJI(remap_label(true), remap_label(pred))
            aji_p = AJI_plus(remap_label(true), remap_label(pred))
            dice2 = DICE2(remap_label(true), remap_label(pred))
            splits, merges = split_and_merge(remap_label(true), remap_label(pred))

            result = {
                "AJI": aji,
                "AJI_plus": aji_p,
                "DICE2": dice2,
                "PQ": pq["pq"],
                "SQ": pq["sq"],
                "DQ": pq["dq"],
                "inst_recall": pq["recall"],
                "inst_precision": pq["precision"],
                "splits": splits,
                "merges": merges
            }

            return name, result

    def benchmark_insts(self,
                        inst_maps: Dict[str, np.ndarray],
                        gt_masks: Dict[str, np.ndarray],
                        save: bool = False,
                        save_dir: Union[str, Path]=None) -> pd.DataFrame:
        """
        Run benchmarking metrics for instance maps for all of the files in the dataset.
        Note that the inst_maps and gt_masks need to be sorted so they align
        when computing metrics.
        
        Args:
            inst_maps (OrderedDict[str, np.ndarray]): 
                A dict of file_name:inst_map key vals in order
            gt_masks (OrderedDict[str, np.ndarray]): 
                A dict of file_name:gt_inst_map key vals in order
            save (bool): 
                Save the result table to .csv file
            save_dir (str or Path):
                directory where to save the result .csv

        Returns:
            a pandas dataframe of the metrics. Samples are rows and metrics are columns:
            _____________________
            |sample|PQ|SQ|DQ|AJI|
            |img1  |.5|.4|.6|.6 |
            |img2  |.5|.4|.6|.6 |
            
        """
        assert isinstance(inst_maps, dict), f"inst_maps: {type(inst_maps)} is not a dict of inst_maps"
        assert isinstance(gt_masks, dict), f"inst_maps: {type(gt_masks)} is not a dict of inst_maps"

        masks = list(zip(inst_maps.keys(), gt_masks.values(), inst_maps.values()))

        metrics = []
        with Pool() as pool:
            for x in tqdm(pool.imap_unordered(self.post_proc_pipeline, masks), total=len(masks)):
                metrics.append(x)


        # TODO FIX FROM HERE
        for i, fn in enumerate(gt_masks.keys()):
            self.inst_metrics[f"{fn}_metrics"] = metrics[i]

        # score_df = pd.DataFrame.from_records([self.inst_metrics]).transpose()
        score_df = pd.DataFrame(self.inst_metrics).transpose()
        score_df.loc["averages_for_the_set"] = score_df.mean(axis=0)

        if self.dataset == "pannuke":
            df = score_df.rename_axis("fn").reset_index()
            td = {f"{tissue}_avg": df[df.fn.str.contains(f"{tissue}")].mean(axis=0) 
                  for tissue in self.pannuke_tissues}
            score_df = pd.concat([score_df, pd.DataFrame(td).transpose()])

        if save:
            result_dir = Path(self.experiment_dir / "benchmark_results")
            self.create_dir(result_dir)
            score_df.to_csv(Path(result_dir / f"{self.exargs.experiment_version}_benchmark_result.csv"))

        return score_df

    def benchmark_per_type(self,
                           inst_maps: Dict[str, np.ndarray],
                           type_maps: Dict[str, np.ndarray],
                           gt_mask_insts: Dict[str, np.ndarray],
                           gt_mask_types: Dict[str, np.ndarray],
                           classes: Dict[str, int],
                           save: bool = False) -> pd.DataFrame:
        """
        Run benchmarking metrics per class type for all of the files in the dataset.
        Note that the inst_maps and gt_masks need to be sorted so they align
        when computing metrics.

        Args:
            inst_maps (Dict[str, np.ndarray]): 
                A dict of file_name:inst_map key vals in order
            type_maps (Dict[str, np.ndarray]): 
                A dict of file_name:panoptic_map key vals in order
            gt_masks_insts (Dict[str, np.ndarray]): 
                A dict of file_name:gt_inst_map key vals in order
            gt_masks_types (Dict[str, np.ndarray]): 
                A dict of file_name:gt_panoptic_map key vals in order
            classes (Dict[str, int]): 
                The class dict e.g. {bg: 0, immune: 1, epithel: 2} background must be 0 class
            save (bool): 
                Save the result table to .csv file
            save_dir (str or Path):
                directory where to save the result .csv

        Returns:
            a pandas dataframe of the metrics. Samples are rows and metrics are columns:
            __________________________
            |sample      |PQ|SQ|DQ|AJI|
            |img1_type1  |.5|.4|.6|.6 |
            |img1_type2  |.5|.4|.6|.6 |
            |img2_type1  |.5|.4|.6|.6 |
            |img2_type2  |.5|.4|.6|.6 |

        """
        # assert self.dataset_args.class_types == "panoptic", "You can only use this when doing panoptic segmentation"
        assert isinstance(inst_maps, dict), f"inst_maps: {type(inst_maps)} is not a dict of inst_maps"
        assert isinstance(type_maps, dict), f"inst_maps: {type(type_maps)} is not a dict of panoptic_maps"
        assert isinstance(gt_mask_insts, dict), f"inst_maps: {type(gt_mask_insts)} is not a dict of inst_maps"
        assert isinstance(gt_mask_types, dict), f"inst_maps: {type(gt_mask_types)} is not a dict of inst_maps"

        df_total = pd.DataFrame()
        for c, ix in list(classes.items())[1:]: # skip bg
            gts_per_class = [get_type_instances(i, t, ix) 
                                for i, t in zip(gt_mask_insts.values(), gt_mask_types.values())]

            insts_per_class = [get_type_instances(i, t, ix) 
                                for i, t in zip(inst_maps.values(), panoptic_maps.values())]

            params_list = list(zip(gts_per_class, insts_per_class))

            with Pool() as pool:
                metrics = pool.starmap(self.compute_metrics, params_list)

            for i, fn in enumerate(gt_mask_insts.keys()): 
                self.type_metrics[f"{fn}_{c}_metrics"] = metrics[i]

            score_df = pd.DataFrame(self.type_metrics).transpose()
            score_df.loc[f"{c}_avg_for_the_set"] = score_df.mean(axis=0)

            if self.dataset == "pannuke":
                df = score_df.rename_axis("fn").reset_index()
                td = {f"{tissue}_avg": df[df.fn.str.contains(f"{tissue}")].mean(axis=0) 
                    for tissue in self.pannuke_tissues}
                score_df = pd.concat([score_df, pd.DataFrame(td).transpose()])

            df_total = pd.concat([df_total, score_df])

        if save:
            result_dir = Path(self.experiment_dir / "benchmark_results")
            self.create_dir(result_dir)
            df_total.to_csv(
                Path(result_dir / f"{self.exargs.experiment_version}_benchmark_per_class_result.csv")
            )
        return df_total


    
