import argparse
from typing import Dict, Any, List, Union
from pathlib import Path

import src.dl.lightning as lit
from src.dl.inference.inferer import Inferer
from src.utils import CellMerger, AreaMerger
from src.utils import AreaMerger


def classes2dict(class_str: str) -> Dict[str, int]:
    """
    Convert the class str to dictionary
    
    Args:
    ---------
        class_str (str):
            String containing the classes in order.
            e.g. "bg,cancer,epithel"
            
    Returns:
    ----------
        Dict: dictionary of class names mapped to numerical value.
              e.g. {"bg":0, "cancer":1, "epithel":2}
    """
    class_dict = None
    if class_str is not None:
        class_dict = {
            name.strip(): n 
            for n, name in dict(enumerate(class_str.split(","))).items()
        }

    return class_dict


def default_branch_args(branches: List[str]):
    """
    Default Branch arg dicts needed for the inferer.
    These are based on the model decoder branches
    """
    branch_weights = {}
    branch_acts = {}
    for branch in branches:
        if branch == "aux":
            branch_weights[branch] = True
            branch_acts[branch] = None
        else:
            branch_weights[branch] = False
            branch_acts[branch] = "softmax"
        
    return branch_weights, branch_acts


def rm_tree(pth: Union[str, Path]) -> None:
    """
    Recursively rm a folder and it's contents
    
    Args:
    --------
        pth (str):
            Path to the folder
    """
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()
    

def main(params: Dict[str, Any]):
    """
    Runs inference on image patches extracted from WSI and saves them
    geojson and merges them to one geojson file representing the cell
    objects of the whole image.
    """
    lightning_model = lit.SegExperiment.from_experiment(
        name=params.exp_name, version=params.exp_version, inference_mode=True
    )
    weights, acts = default_branch_args(
        lightning_model.hparams["dec_branches"].keys()
    )
    
    for d in Path(params.in_dir).iterdir():
        print(f"inference: {d}")
        
        inferer = Inferer(
            lightning_model,
            in_data_dir=d,
            gt_mask_dir=params.gt_dir,
            patch_size=(256, 256),
            stride_size=params.stride,
            branch_weights=weights,
            branch_acts=acts,
            post_proc_method=params.post_proc,
            loader_batch_size=1,
            loader_num_workers=1,
            model_batch_size=params.model_bs,
            model_weights=-1,
            fn_pattern="*",
            device=params.device,
            test_mode=False,
            auto_range=bool(params.auto_range),
            n_images=32,
        )
        
        res_dir = Path(params.result_dir) / f"{d.name}"
        inferer.run_inference(
            save_dir=res_dir,
            fformat="geojson",
            offsets=True,
            classes_sem=classes2dict(params.classes_sem),
            classes_type=classes2dict(params.classes_type)
        )
        
        # Merge cell annotations
        if "type" in weights.keys():
            in_dir_cells = Path(res_dir / "cells")
            fname_cells = in_dir_cells.parent / f"{d.name}_cells.json"

            c = CellMerger(in_dir=in_dir_cells)
            c.merge(fname=fname_cells.as_posix())
            
            # rm the patch gsons
            rm_tree(in_dir_cells)
        
        if "sem" in weights.keys():
            in_dir_areas = Path(res_dir / "areas")
            fname_areas = in_dir_areas.parent / f"{d.name}_areas.json"

            a = AreaMerger(in_dir=in_dir_areas)
            a.merge(fname=fname_areas.as_posix())
            
            # rm the patch gsons
            rm_tree(in_dir_areas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--in_dir',
        type=str,
        default=None,
        help=(
            "The path to the image-folder"
        )
    )
    
    parser.add_argument(
        '--gt_dir',
        type=str,
        default=None,
        help=(
            "The path to the GT-mask-folder. Need to be .mat files"
        )
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default=None,
        help=(
            "The path to the dir where the geojson patches are saved."
        )
    )
    
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help=(
            "The experiment name."
        )
    )
    
    parser.add_argument(
        '--exp_version',
        type=str,
        default=None,
        help=(
            "The experiment version."
        )
    )
    
    parser.add_argument(
        '--classes_type',
        type=str,
        default=None,
        help=(
            "Cell type classes in comma separated ordered string",
            "For example: 'bg,neoplastic,inflammatory'",
            "This will be converted into a class dict."
        )
    )
    
    parser.add_argument(
        '--classes_sem',
        type=str,
        default=None,
        help=(
            "Semantic classes in comma separated ordered string",
            "For example: 'bg,neoplastic,inflammatory'",
            "This will be converted into a class dict."
        )
    )
    
    parser.add_argument(
        '--post_proc',
        type=str,
        default="cellpose",
        help=("The post-processing method.")
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default="cuda",
        help=("One of: cpu, cuda")
    )
    
    parser.add_argument(
        '--model_bs',
        type=int,
        default=16,
        help=("The model batch size.")
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=80,
        help=("The stride for the sliding window.")
    )
    
    parser.add_argument(
        '--auto_range',
        type=int,
        default=0,
        help=("..")
    )
    
    args = parser.parse_args()
    main(args)
    