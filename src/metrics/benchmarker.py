import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List, Union
from collections import OrderedDict
from pathos.multiprocessing import ThreadPool as Pool
from tqdm import tqdm

from src.utils.mask_utils import remap_label, get_type_instances
from .metrics import PQ, AJI, AJI_plus, DICE2, split_and_merge


class Benchmarker:
    def compute_metrics(self, true_pred: List[np.ndarray]) -> Dict[str, float]:
        """
        Computes metrics for one (inst_map, gt_mask) pair.

        Args:
            true_pred (List[np.ndarray]): 
                Ground truth annotations in true_pred[1] and corresponding 
                predicted instance map in true_pred[2]

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
                "name":name,
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

            return result

    def benchmark_insts(self,
                        inst_maps: Dict[str, np.ndarray],
                        gt_masks: Dict[str, np.ndarray],
                        pattern_list: List[str]=None,
                        save_dir: Union[str, Path]=None,
                        prefix: str="") -> pd.DataFrame:
        """
        Run benchmarking metrics for instance maps for all of the files in the dataset.
        Note that the inst_maps and gt_masks need to share exact same keys and be sorted
        so that they align when computing metrics.
        
        Args:
            inst_maps (OrderedDict[str, np.ndarray]): 
                A dict of file_name:inst_map key vals in order
            gt_masks (OrderedDict[str, np.ndarray]): 
                A dict of file_name:gt_inst_map key vals in order
            pattern_list (List[str], default=None):
                A list of patterns contained in the gt_mask and inst_map names.
                Averages for the masks containing these patterns will be added
                to the result df.
            save_dir (str or Path):
                directory where to save the result .csv
            prefix (str, default=""):
                adds a prefix to the .csv file name


        Returns:
            a pandas dataframe of the metrics. Samples are rows and metrics are columns:
            _____________________
            |sample|PQ|SQ|DQ|AJI|
            |img1  |.5|.4|.6|.6 |
            |img2  |.5|.4|.6|.6 |
            
        """
        assert isinstance(inst_maps, dict), f"inst_maps: {type(inst_maps)} is not a dict of inst_maps"
        assert isinstance(gt_masks, dict), f"inst_maps: {type(gt_masks)} is not a dict of inst_maps"

        # Sort by file name
        inst_maps = OrderedDict(sorted(inst_maps.items()))
        gt_masks = OrderedDict(sorted(gt_masks.items()))
        assert inst_maps.keys() == gt_masks.keys(), (
            f"inst_maps have different names as gt masks. insts: {inst_maps.keys()}. gt's: {gt_masks.keys()}"
        )

        masks = list(zip(inst_maps.keys(), gt_masks.values(), inst_maps.values()))
        
        metrics = []
        with Pool() as pool:
            for x in tqdm(pool.imap_unordered(self.compute_metrics, masks), total=len(masks)):
                metrics.append(x)
        
        score_df = pd.DataFrame.from_records(metrics).set_index("name").sort_index()
        score_df.loc["averages_for_the_set"] = score_df.mean(axis=0)

        # Add averages to the df of files which contain patterns in the pattern list
        if pattern_list is not None:
            pattern_avgs = {f"{p}_avg": score_df[score_df.index.str.contains(f"{p}")].mean(axis=0) for p in pattern_list}
            score_df = pd.concat([score_df, pd.DataFrame(pattern_avgs).transpose()])

        # Save results to .csv
        if save_dir is not None:
            save_dir = Path(save_dir)
            score_df.to_csv(Path(save_dir / f"{prefix}_inst_benchmark.csv"))

        return score_df

    def benchmark_per_type(self,
                           inst_maps: Dict[str, np.ndarray],
                           type_maps: Dict[str, np.ndarray],
                           gt_mask_insts: Dict[str, np.ndarray],
                           gt_mask_types: Dict[str, np.ndarray],
                           classes: Dict[str, int],
                           pattern_list: List[str]=None,
                           save_dir: Union[str, Path]=None,
                           prefix: str="") -> pd.DataFrame:
        """
        Run benchmarking metrics per class type for all of the files in the dataset.
        Note that the inst_maps and gt_masks need to share exact same keys and be sorted
        so that they align when computing metrics.

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
            pattern_list (List[str], default=None):
                A list of patterns contained in the gt_mask and inst_map names.
                Averages for the masks containing these patterns will be added
                to the result df.
            save_dir (str or Path):
                directory where to save the result .csv
            prefix (str, default=""):
                adds a prefix to the .csv file name

        Returns:
            a pandas dataframe of the metrics. Samples are rows and metrics are columns:
            __________________________
            |sample      |PQ|SQ|DQ|AJI|
            |img1_type1  |.5|.4|.6|.6 |
            |img1_type2  |.5|.4|.6|.6 |
            |img2_type1  |.5|.4|.6|.6 |
            |img2_type2  |.5|.4|.6|.6 |

        """
        assert isinstance(inst_maps, dict), f"inst_maps: {type(inst_maps)} is not a dict of inst_maps"
        assert isinstance(type_maps, dict), f"inst_maps: {type(type_maps)} is not a dict of panoptic_maps"
        assert isinstance(gt_mask_insts, dict), f"inst_maps: {type(gt_mask_insts)} is not a dict of inst_maps"
        assert isinstance(gt_mask_types, dict), f"inst_maps: {type(gt_mask_types)} is not a dict of inst_maps"

        # sort by name
        inst_maps = OrderedDict(sorted(inst_maps.items()))
        type_maps = OrderedDict(sorted(type_maps.items()))
        gt_mask_insts = OrderedDict(sorted(gt_mask_insts.items()))
        gt_mask_types = OrderedDict(sorted(gt_mask_types.items()))
        assert inst_maps.keys() == gt_mask_insts.keys(), (
            f"inst_maps have different names as gt masks. insts: {inst_maps.keys()}. gt's: {gt_mask_insts.keys()}"
        )

        # Loop masks per class
        df_total = pd.DataFrame()
        for c, ix in list(classes.items())[1:]: # skip bg
            gts_per_class = [get_type_instances(i, t, ix) for i, t in zip(gt_mask_insts.values(), gt_mask_types.values())]
            insts_per_class = [get_type_instances(i, t, ix) for i, t in zip(inst_maps.values(), type_maps.values())]

            masks = list(zip(inst_maps.keys(), gts_per_class, insts_per_class))

            metrics = []
            with Pool() as pool:
                for x in tqdm(pool.imap_unordered(self.compute_metrics, masks), total=len(masks)):
                    metrics.append(x)
            
            # drop Nones if no classes are found in an image
            metrics = [metric for metric in metrics if metric] 
            score_df = pd.DataFrame.from_records(metrics).set_index("name").sort_index()
            score_df.loc[f"{c}_avg_for_the_set"] = score_df.mean(axis=0)

            # Add averages to the df of files which contain patterns in the pattern list
            if pattern_list is not None:
                pattern_avgs = {f"{c}_{p}_avg": score_df[score_df.index.str.contains(f"{p}")].mean(axis=0) for p in pattern_list}
                score_df = pd.concat([score_df, pd.DataFrame(pattern_avgs).transpose()])

            df_total = pd.concat([df_total, score_df])

        # Save results to .csv
        if save_dir is not None:
            save_dir = Path(save_dir)
            df_total.to_csv(Path(save_dir / f"{prefix}_type_benchmark.csv"))

        return df_total
    
