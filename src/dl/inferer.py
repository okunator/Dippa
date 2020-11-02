import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.feature_extraction.image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ttach as tta

from pathlib import Path
from omegaconf import DictConfig
from skimage.exposure import histogram
from skimage.color import label2rgb
from collections import OrderedDict
from pathos.multiprocessing import ThreadPool as Pool
from typing import List, Dict, Tuple, Callable, Optional, Union

import src.img_processing.post_processing as post_proc

from src.utils.patch_extractor import PatchExtractor
from src.utils.benchmarker import Benchmarker
from src.img_processing.viz_utils import draw_contours, KEY_COLORS
from src.dl.torch_utils import (
    argmax_and_flatten, tensor_to_ndarray,
    ndarray_to_tensor, to_device
)

from src.img_processing.augmentations import (
    tta_augs, tta_deaugs, resize, tta_transforms, tta_five_crops
)

class Inferer(Benchmarker, PatchExtractor):
    def __init__(self, 
                 model: nn.Module,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig,
                 inference_args: DictConfig,
                 **kwargs) -> None:
        """
        Inferer for any of the models that are trained with lightning framework 
        in this project (defined in lightning_model.py)
        
        Args: 
            model (nn.Module) : Pytorch model specification. Can be from smp, toolbelt, or a 
                                custom model. ttatch wrappers work also. Basically any model 
                                that inherits nn.Module should work
            dataset_args (DictConfig): omegaconfig DictConfig specifying arguments
                                       related to the dataset that is being used.
                                       config.py for more info
            experiment_args (DictConfig): omegaconfig DictConfig specifying arguments
                                          that are used for creating result folders and
                                          files. Check config.py for more info
            inference_args (DictConfig): omegaconfig DictConfig specifying arguments
                                         that are used for inference and post processing.
                                         Check config.py for more info

        """
    
        super(Inferer, self).__init__(dataset_args, experiment_args)
        self.model = model
        self.batch_size = inference_args.batch_size
        self.input_size = inference_args.model_input_size
        self.smoothen = inference_args.smoothen
        self.verbose = inference_args.verbose
        self.fold = inference_args.data_fold
        self.test_time_augs = inference_args.tta
        self.thresh_method = inference_args.thresh_method
        self.thresh = inference_args.threshold
        self.post_proc = inference_args.post_processing
        self.post_proc_method = inference_args.post_proc_method
        
        # init containers for resluts
        self.soft_insts = OrderedDict()
        self.soft_types = OrderedDict()
        self.inst_maps = OrderedDict()
        self.type_maps = OrderedDict()
        self.panoptic_maps = OrderedDict()

        # Put SegModel to gpu|cpu and eval mode
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.model.eval()
        torch.no_grad()
    
    
    @classmethod
    def from_conf(cls, model: nn.Module, conf: DictConfig):
        model = model
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        inference_args = conf.inference_args
        
        return cls(
            model,
            dataset_args,
            experiment_args,
            inference_args
        )
        
    @property
    def stride_size(self) -> int:
        return self.input_size//2
    
    @property
    def tta_model(self) -> nn.Module:
        return tta.SegmentationTTAWrapper(self.model, tta_transforms())
    
    @property
    def images(self) -> List[str]:
        assert self.fold in self.phases, f"fold param: {self.fold} was not in given phases: {self.phases}" 
        return self.data_folds[self.fold]["img"]
    
    @property
    def gt_masks(self) -> List[str]:
        assert self.fold in self.phases, f"fold param: {self.fold} was not in given phases: {self.phases}"
        return self.data_folds[self.fold]["mask"]

    def __clear_predictions(self) -> None:
        """
        Clear soft_masks if there are any
        """
        self.soft_insts.clear()

    def __get_fn(self, path: str) -> Path:
        return Path(path).name[:-4]

    def __sample_idxs(self, n: int = 25) -> np.ndarray:
        assert n <= len(self.images), "Cannot sample more ints than there are images in dataset"
        # Dont plot more than 50 pannuke images
        n = 50 if self.dataset == "pannuke" and n > 50 else n
        return np.random.randint(low=0, high=len(self.images), size=n)

    def __get_batch(self, arr: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Divide patched image array into batches similarly to DataLoader in pytorch
        convert it to torch.Tensor and move the the tensor to right device

        Args:
            arr (np.ndarray): patched array. Shape (num_patches, H, W, 3)
            batch_size (int): size of the batch
        Yields:
            np.ndarray of shape (B, H, W, 3)
        """
        for i in range(0, arr.shape[0], batch_size):
            yield arr[i:i + batch_size, ::]

    def __get_patch(self, batch: np.ndarray) -> np.ndarray:
        """
        Divide a batch np.ndarray into patches 

        Args:
            batch (np.ndarray): inut image batch array. Shape (B, H, W, 3)
        Yields:
            a np.ndarray of shape (H, W, 3)
        """
        for i in range(batch.shape[0]):
            yield i, batch[i]

    def __apply_thresh_instmap(self,
                               prob_map: np.ndarray,
                               thresh: Optional[Union[float, str]]) -> np.ndarray:
        """
        Apply a thresholding or argmax to an instance segmentation soft mask

        Args:
            prob_map (np.ndarray): soft mask of shape (H, W, 2) for instance map
            thresh (Uninon[float, str]): either a value between [0, 1] or "argmax".
                                         If smoothen is used this is ignored.

        Returns:
            thresholded and labelled np.ndarray instance map of shape (H, W)
        """
        # TODO: add thresholding methods
        assert prob_map.shape[2] == 2, f"Shape: {prob_map.shape} should have only two channels"
        assert self.thresh_method in ("argmax", "sauvola_thresh", "niblack_thresh", None)
        
        if self.smoothen:
            mask = post_proc.smoothed_thresh(prob_map)
        elif self.thresh_method is not None:
            mask = post_proc.__dict__[self.thresh_method](prob_map=prob_map)
        else:
            assert isinstance(self.thresh, float)
            mask = post_proc.naive_thresh_prob(prob_map, thresh)
        return mask

    def __logits(self, patch: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Input an image patch or batch of patches to the network and return logits.

        Args:
            im (np.ndarray): image patch of shape (input_size, input_size, 3)
                             or (B, input_size, input_size, 3)
        Returns:
            A dictionary {"instances":torch.Tensor, "types":Union[torch.Tensor, None]}
        """
        input_tensor = ndarray_to_tensor(patch)  # (1, 3, H, W)
        input_tensor = to_device(input_tensor) # to cpu|gpu
        return self.model(input_tensor)  # Dict[(B, 2, H, W), (B, C, H, W)]

    def __gen_prediction(self, logits: torch.Tensor, squeeze: bool = False) -> np.ndarray:
        """
        Take in a patch or a batch of patches of logits produced by the model and
        use sigmoid activation for instance logits and softmax for semantic logits
        and convert the output to numpy nd.array.

        Args:
            logits (torch.Tensor): a tensor of logits produced by the network.
                                   Shape: (B, C, input_size, input_size)
            squeeze (bool): whether to squeeze the output batch if batch dim is 1
        Returns:
            np.ndarray of the result
        """
        if logits.shape[1] == 2:
            pred = torch.sigmoid(logits)
        else:
            pred = F.softmax(logits, dim=1)

        return tensor_to_ndarray(pred, squeeze=squeeze)

    def __smoothed_dog(self, patch: np.ndarray) -> np.ndarray:
        """
        Use DoG to smoothen soft mask patch.

        Args:
            patch (np.ndarray): the soft mask pach to smoothen. Shape (H, W, C)
        """
        for c in range(patch.shape[2]):
            patch[..., c] = post_proc.activate_plus_dog(patch[..., c])  # (H, W)
        return patch
         
            
    def __gen_ensemble_prediction(self, patch: np.ndarray) -> np.ndarray:
        """
        Tta ensemble prediction pipeline (memory issues with ttach).
    
        Args:
            patch (np.ndarray): the img patch used for ensemble prediction. 
                   shape (input_size, input_size, 3)
        Returns:
            np.ndarray soft mask of shape (input_size, input_size, C)
        
        Following instructions of 'beneficial augmentations' from:
        
        "Towards Principled Test-Time Augmentation"
        D. Shanmugam, D. Blalock, R. Sahoo, J. Guttag 
        https://dmshanmugam.github.io/pdfs/icml_2020_testaug.pdf
        
        1. vflip, hflip and transpose and rotations
        2. custom fivecrops aug
        3. take the mean of predictions
        
        Idea for flips and rotations:
            1. flip or rotate, 
            2. predict, 
            3. deaugment prediction,
            4. save to augmented results
        Idea for five_crops:
            1. crop to one fourth of network input size (corner crop or center crop)
            2. scale to network input size
            3. predict
            4. downscale to crop size
            5. Save result to result matrix
            6. Save result matrix to augmented results
        """
        soft_instances = []
        soft_types = []
        
        # flip ttas
        for aug, deaug in zip(tta_augs(), tta_deaugs()):
            aug_input = aug(image = patch) # (H, W, 3)
            aug_logits = self.__logits(aug_input["image"]) # (1, C, H, W)
            aug_insts = self.__gen_prediction(aug_logits["instances"], squeeze=True) # (H, W, C)
            deaug_insts = deaug(image = aug_insts) # (H, W, C)
            soft_instances.append(deaug_insts["image"])

            if self.class_types == "panoptic":
                aug_types = self.__gen_prediction(aug_logits["types"], squeeze=True)# (H, W, C)
                deaug_types = deaug(image=aug_types)  # (H, W, C)
                soft_types.append(deaug_types["image"])
            
        # five crops tta
        scale_up = resize(patch.shape[0], patch.shape[1])
        scale_down = resize(patch.shape[0]//2, patch.shape[1]//2)
        out_insts = np.zeros((patch.shape[0], patch.shape[1], 2))
        out_types = np.zeros((patch.shape[0], patch.shape[1], len(self.classes)))
        for crop in tta_five_crops(patch):
            cropped_im = crop(image = patch)
            scaled_im = scale_up(image = cropped_im["image"]) # (H, W, C)
            aug_logits = self.__logits(scaled_im["image"])  # (1, C, H, W)
            aug_insts = self.__gen_prediction(aug_logits["instances"], squeeze=True) # (H, W, C)
            downscaled_insts = scale_down(image = aug_insts) # (H//2, W//2, C)
            out_insts[crop.y_min:crop.y_max, crop.x_min:crop.x_max] = downscaled_insts["image"] # (H, W, C)
            soft_instances.append(out_insts)

            if self.class_types == "panoptic":
                aug_types = self.__gen_prediction(aug_logits["types"], squeeze=True) # (H, W, C)
                downscaled_types = scale_down(image=aug_types)  # (H//2, W//2, C)
                out_types[crop.y_min:crop.y_max,crop.x_min:crop.x_max] = downscaled_types["image"]  # (H, W, C)
                soft_types.append(out_types)

        # TODO:
        # 16 crops tta
            
        # take the mean of all the predictions
        return {
            "instances":np.asarray(soft_instances).mean(axis=0),
            "types": np.asarray(soft_types).mean(axis=0)
        }

    def prediction_two_branch(self, batch: np.ndarray) -> np.ndarray:
        """
        Takes in an image batch of shape (B, input_size, input_size, 3) and
        produces a prediction from a network with separate instance and semantic
        segmentation branch.

        Args:
            batch (np.ndarray): image batch for prediction
        """
        pred_batch_insts = np.zeros(
            (batch.shape[0], self.input_size, self.input_size, 2)
        )
        pred_batch_types = np.zeros(
            (batch.shape[0], self.input_size, self.input_size, len(self.classes))
        )

        if not self.test_time_augs:
            pred_logits = self.__logits(batch)
            pred_batch_insts = self.__gen_prediction(pred_logits["instances"]) # (B, H, W, 2)
            pred_batch_types = self.__gen_prediction(pred_logits["types"]) # (B, H, W, C)
        else:
            for i, patch in self.__get_patch(batch):  # (H, W, 3)
                ensemble = self.__gen_ensemble_prediction(patch)
                pred_batch_insts[i, ...] = ensemble["instances"] # (H, W, 2)
                pred_batch_types[i, ...] = ensemble["types"] # (H, W, C)

        if self.smoothen:
            for i, pred_patch in self.__get_patch(pred_batch_insts): # (H, W, C)
                pred_patch = self.__smoothed_dog(pred_patch)
                pred_batch_insts[i, ...] = pred_patch  # (H, W, C)

            for i, pred_patch in self.__get_patch(pred_batch_types): # (H, W, C)
                pred_patch = self.__smoothed_dog(pred_patch)
                pred_batch_types[i, ...] = pred_patch  # (H, W, C)

        return {
            "instances":pred_batch_insts,
            "types":pred_batch_types
        }

    def prediction_single_branch(self, batch: np.ndarray) -> np.ndarray:
        """
        Takes in an image batch of shape (B, input_size, input_size, 3) and produces
        a prediction from a network with only an instance branch.
        Args:
            batch (np.ndarray): image batch for prediction
        """
        pred_batch_insts = np.zeros(
            (batch.shape[0], self.input_size, self.input_size, 2)
        )

        if not self.test_time_augs:
            pred_logits = self.__logits(batch)
            pred_batch_insts = self.__gen_prediction(pred_logits["instances"]) # (B, H, W, 2)
        else:
            for i, patch in self.__get_patch(batch):  # (H, W, 3)
                ensemble = self.ensemble_predict(patch)
                pred_batch_insts[i, ...] = ensemble["instances"] # (H, W, 2)
        if self.smoothen:
            for i, pred_patch in self.__get_patch(pred_batch_insts): # (H, W, C)
                pred_patch = self.__smoothed_dog(pred_patch)
                pred_batch_insts[i, ...] = pred_patch  # (H, W, C)

        return {
            "instances":pred_batch_insts,
        }

    def predict_patches(self, im_patches: np.ndarray) -> np.ndarray:
        """
        Divide patched image array to batches and process and run predictions for batches.
        Use tta and DoG smoothing if specified in config.py.

        Args:
            im_patches (np.ndarray): patched input image of shape 
                                    (num_patches, input_size, input_size, 3)
        Returns:
            np.ndarray containing patched soft mask of shape 
            (num_patches, input_size, input_size, C)
        """
        pred_patches_insts = np.zeros((0, self.input_size, self.input_size, 2))
        pred_patches_types = np.zeros((0, self.input_size, self.input_size, len(self.classes)))
        for batch in self.__get_batch(im_patches, self.batch_size): # (B, H, W, 3)
            if self.class_types == "panoptic":
                pred_patch = self.prediction_two_branch(batch)
                insts = pred_patch["instances"]
                types = pred_patch["types"]
                pred_patches_insts = np.append(pred_patches_insts, insts, axis=0) # (B, H, W, C)
                pred_patches_types = np.append(pred_patches_types, types, axis=0) # (B, H, W, C)
            else:
                pred_patch = self.prediction_single_branch(batch)
                insts = pred_patch["instances"]
                pred_patches_insts = np.append(pred_patches_insts, insts, axis=0) # (B, H, W, C)

        return {
            "instances":pred_patches_insts,
            "types":pred_patches_types
        }
   
    def run_predictions_all(self) -> None:
        """
        Run predictions for all the images in the dataset defined in config.py.
        Saves all the predictions to a container so this will take some memory.
        """        
        if self.soft_insts:
            self.__clear_predictions()
                    
        for path in self.images:
            fn = self.__get_fn(path)
            if self.verbose:
                print(f"Prediction for: {fn}")
                
            im = self.read_img(path)
            if self.dataset == "pannuke":
                result_pred = self.predict_patches(im[None, ...])
                res_insts = result_pred["instances"]
                if self.class_types == "panoptic":
                    res_types = result_pred["types"]
            else:
                patches, shape = self.extract_inference_patches(im, self.stride_size, self.input_size)
                pred_patches = self.predict_patches(patches)
                insts = pred_patches["instances"]
                res_insts = self.stitch_inference_patches(insts, self.stride_size, shape, im.shape)
                if self.class_types == "panoptic":
                    types = pred_patches["types"]
                    res_types = self.stitch_inference_patches(types, self.stride_size, shape, im.shape)
            
            if self.class_types == "panoptic":
                self.soft_types[f"{fn}_soft_types"] = res_types
                self.type_maps[f"{fn}_type_map"] = np.argmax(res_types, axis=2)

            self.soft_insts[f"{fn}_soft_instances"] = res_insts

            
    def post_process_instmap(self,
                             soft_inst: np.ndarray,
                             thresh: Union[float, str]) -> np.ndarray:
        """
        Takes in a soft instance mask of shape (H, W, C) and thresholds it.
        Post processing is applied if defined in config.py

        Args:
            soft_inst (np.ndarray): soft mask of instances. Shape (H, W, C)
            thresh (Union[float, sty]): threshold value for naive thresholding or a str
                                        specifying a thresholding method. For now only 
                                        "argmax" is available.

        Returns:
            np.ndarray with instances labelled
        """

        inst_map = self.__apply_thresh_instmap(soft_inst, thresh)

        # TODO: Add postproc functions
        if self.post_proc:
            assert self.post_proc_method in (
                "shape_index_watershed", "shape_index_watershed2", "inv_dist_watershed", "sobel_watershed"
            ), f"post_proc_method: {self.post_proc_method} not found. Check config.py"
            
            kwargs = {}
            kwargs.setdefault("inst_map", inst_map)
            kwargs.setdefault("prob_map", soft_inst[..., 1])
            inst_map = post_proc.__dict__[self.post_proc_method](**kwargs)
            
        return inst_map

    def panoptic_output(self,
                        inst_map: np.ndarray,
                        type_map: np.ndarray) -> np.ndarray:
        """
        For now only a wrapper for combine_inst_semantic from src.img_processing.post_processing
        """
        # TODO: optional different combining heuristics
        return post_proc.combine_inst_semantic(inst_map, type_map)

                
    def post_process(self) -> None:
        """
        Run post processing pipeline for all the predictions from the network.
        Assumes that 'run_predictions_all' has been run beforehand. If semantic
        classes exist this creates also a panoptic output from the semantic and
        instance segmentations by applying heuristics to combine the outputs to
        a single one. Heuristics are from the HoVer-net paper.
        """
        
        assert self.soft_insts, (
            f"{self.soft_insts}, No predictions found. Run 'run_for_all' first"
        )
        
        inst_preds = [(self.soft_insts[key], self.thresh) 
                      for key in self.soft_insts.keys() 
                      if key.endswith("instances")]
                
        # pickling issues in ProcessPool with typing, hard to fix.. Using ThreadPool instead        
        with Pool() as pool:
            segs = pool.starmap(self.post_process_instmap, inst_preds)
        
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            self.inst_maps[f"{fn}_inst_map"] = segs[i]

        # combine semantic and instance segmentations
        if self.class_types == "panoptic":
            maps = [(self.inst_maps[i].astype("uint16"), self.type_maps[t].astype("uint8"))
                    for i, t in zip(self.inst_maps, self.type_maps)]

            with Pool() as pool:
                panops = pool.starmap(post_proc.combine_inst_semantic, maps)

            for i, path in enumerate(self.images):
                fn = self.__get_fn(path)
                self.panoptic_maps[f"{fn}_panoptic_map"] = panops[i]

    def run_benchmarks(self,
                       save: bool = True) -> None:
        """
        Run benchmarks for instance and panoptic segmentations
        
        Args:
            save (bool): save the results to csv
        """
        assert self.inst_maps, "No instance maps found, Run inference first"
        gt_insts = OrderedDict((self.__get_fn(p), self.read_mask(p)) for p in self.gt_masks)
        inst_metrics = self.benchmark_instmaps(self.inst_maps, gt_insts, save=save)
        class_metrics = None
        if self.class_types == "panoptic":
            gt_types = OrderedDict((self.__get_fn(p), self.read_mask(p, "type_map")) for p in self.gt_masks)
            class_metrics = self.benchmark_panoptic_maps(
                self.inst_maps, 
                self.panoptic_maps,
                gt_insts,
                gt_types,
                self.classes,
                save=save
            )
        return {
            "instance_metrics": inst_metrics,
            "type_metrics": class_metrics
        }
            
                  
    def plot_outputs(self, 
                     out_type: str,
                     ixs: Union[List[int], int] = -1,
                     gt_mask: bool = False,
                     contour: bool = False,
                     save: bool = False) -> None:
        """
        Plot different outputs this instance holds. Options are: inst_maps,
        soft_insts, soft_types, panoptic_maps.

        Args:
            out_type (str): output type to plot. Options are listed above.
                            If soft_types are soft_insts are used then gt_masks and
                            contour args will be ignored.
            ixs (List or int): list of the indexes of the image files in the dataset. 
                               default = -1 means all images in the data fold are 
                               plotted. If dataset = "pannuke" and ixs = -1, then 25
                               random images are sampled.
            gt_mask (bool): plot the corresponding panoptic or instance gt next to the
                            corresponding inst or panoptic map. Ignored if soft masks
                            are plotted
            contour (bool): Plots instance contours on top of the original image instead
                            plotting a mask. Ignored if soft masks are plotted
            save (bool): Save the plots

        """
        # THIS IS A HUGE KLUDGE. TODO someday make this pretty
        assert out_type in ("inst_maps", "soft_insts", "soft_types", "panoptic_maps", "type_maps")

        assert self.__dict__[out_type], (
            f"outputs for {out_type} not found. Run predictions and then" 
            " post processing to get them all"
        )

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

        outputs = [list(self.__dict__[out_type].items())[i] for i in idxs]
        images = np.asarray(self.images)[idxs]
        gt_masks = np.asarray(self.gt_masks)[idxs]

        if "soft" in outputs[0][0]:
            ncol = outputs[0][1].shape[-1]
        elif gt_mask:
            ncol = 2
        else:
            ncol = 1

        fig, axes = plt.subplots(
            len(outputs), ncol, figsize=(ncol*25, len(outputs)*25), squeeze=False
        )

        # class_names = {y: x for x, y in self.classes.items()}
        for j, (name, out) in enumerate(outputs):
            for c in range(ncol):
                kwargs = {}
                if "soft" in outputs[0][0]:
                    x = out[..., c]
                    # name = f"{n}_{class_names[c]}"
                else:
                    if gt_mask and divmod(2, c+1)[0] == 1:
                        inst = self.read_mask(gt_masks[j])
                    else:
                        if out_type == "inst_maps":
                            n = name  
                        elif out_type == "panoptic_maps":
                            n = name.replace("panoptic", "inst")
                        elif out_type == "type_maps":
                            n = name.replace("type", "inst")
                        inst = self.inst_maps[n]
                    kwargs.setdefault("label", inst)
                    
                    if out_type == "panoptic_maps":
                        if gt_mask and divmod(2, c+1)[0] == 1:
                            type_map = self.read_mask(gt_masks[j], key="type_map")
                        else:
                            type_map = out.astype("uint8")
                        kwargs.setdefault("type_map", type_map)
                        kwargs.setdefault("classes", self.classes)

                    if contour:
                        im = self.read_img(images[j])
                        kwargs.setdefault("image", im)
                        x = draw_contours(**kwargs)
                    else:
                        x = label2rgb(inst, bg_label=0)
                    
                axes[j, c].set_title(f"{name}", fontsize=30)
                axes[j, c].imshow(x, interpolation='none')
                axes[j, c].axis('off')
        
        if out_type == "panoptic_maps" and contour:
            colors = {k: KEY_COLORS[k] for k, v in self.classes.items()}
            patches = [mpatches.Patch(color=np.array(colors[k])/255., label=k) for k, v in self.classes.items()]
            fig.legend(handles=patches, loc=1, borderaxespad=0.2, bbox_to_anchor=(1.26, 1), fontsize=50,)
        
        fig.tight_layout(w_pad=4, h_pad=10)

        if save:
            plot_dir = Path(self.experiment_dir / "inference_plots")
            self.create_dir(plot_dir)

            s = "smoothed" if self.smoothed else ""
            t = "tta" if self.tta else ""
            fig.savefig(Path(plot_dir / f"{name}_{t}_{s}_result.png"))
