import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from collections import OrderedDict
from typing import List, Dict
from omegaconf import DictConfig
from pathos.multiprocessing import ThreadPool as Pool
from skimage.color import label2rgb

import src.dl.torch_utils as torch_utils
from src.utils.file_manager import ProjectFileManager
from src.dl.inference.postprocessor import PostProcessor
from src.dl.inference.predictor import Predictor
from src.img_processing.patching.tiler_stitcher import TilerStitcher
from src.img_processing.viz_utils import draw_contours, KEY_COLORS


class Inferer(ProjectFileManager):
    def __init__(self,
                 model: nn.Module,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig,
                 inference_args: DictConfig,
                 **kwargs) -> None:
        """
        Inferer class for the lightning model

        Args: 
            model (nn.Module) : 
                Pytorch model specification.
            dataset_args (DictConfig): 
                omegaconfig DictConfig specifying arguments related to the 
                dataset that is being used. config.py for more info
            experiment_args (DictConfig): 
                Omegaconfig DictConfig specifying arguments that are used for
                creating result folders and files. Check config.py for more info
            inference_args (DictConfig): 
                Omegaconfig DictConfig specifying arguments that are used for 
                inference and post processing. Check config.py for more info
        """
        super(Inferer, self).__init__(dataset_args, experiment_args)
        self.model = model
        self.batch_size = inference_args.batch_size
        self.input_size = inference_args.model_input_size
        self.smoothen = inference_args.smoothenbatch_size
        self.verbose = inference_args.verbose
        self.fold = inference_args.data_fold
        self.test_time_augs = inference_args.tta
        self.thresh_method = inference_args.thresh_method
        self.thresh = inference_args.threshold
        self.post_proc = inference_args.post_processing
        self.post_proc_method = inference_args.post_proc_method

        # init containers for results
        self.soft_insts = OrderedDict()
        self.soft_types = OrderedDict()
        self.inst_maps = OrderedDict()
        self.type_maps = OrderedDict()
        self.aux_maps = OrderedDict()
        self.result_maps = OrderedDict()

        # Put SegModel to gpu|cpu and eval mode
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()
        self.model.model.eval()
        torch.no_grad()

        # Init predictor
        self.predictor = Predictor(self.model)

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
    def images(self) -> List[str]:
        assert self.fold in self.phases, f"fold param: {self.fold} was not in given phases: {self.phases}" 
        return self.data_folds[self.fold]["img"]
    
    @property
    def gt_masks(self) -> List[str]:
        assert self.fold in self.phases, f"fold param: {self.fold} was not in given phases: {self.phases}"
        return self.data_folds[self.fold]["mask"]

    def __get_fn(self, path: str) -> Path:
        return Path(path).name[:-4]

    def __get_batch(self, arr: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Divide patched image array into batches

        Args:
            arr (np.ndarray): patched array. Shape (num_patches,torch.no_grad() H, W, 3)
            batch_size (int): size of the batch
            
        Yields:
            np.ndarray of shape (batch_size, H, W, 3)
        """
        for i in range(0, arr.shape[0], batch_size):
            yield arr[i:i + batch_size, ...]

    def __get_patch(self, batch: np.ndarray) -> np.ndarray:
        """
        Divide a batch np.ndarray into patches 
        pred_patches_insts = np.zeros((0, self.input_size, self.input_size, 2))
        pred_patches_aux = np.zeros((0, self.input_size, self.input_size, 2))
        pred_patches_types = np.zeros((0, self.input_size, self.input_size, len(self.classes)))
        Args:
            batch (np.ndarray): inut image batch array. Shape (B, H, W, 3)
            
        Yields:
            a np.ndarray of shape (H, W, 3)
        """
        for i in range(batch.shape[0]):
            yield i, batch[i]

    # TODO: tta
    def gen_prediction(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Takes in an image batch of shape (B, input_size, input_size, 3) and produces
        a prediction dictionary containing prediction from inst, type and aux branch 

        Args:
            batch (Dict[str, np.ndarray]): 
                Dict of predictions from the image batch. 
                e.g. {"instances":np.ndarray, "types":np.ndarray, "aux":np.ndarray}
        """        
        logits = self.predictor.forward_pass(batch)
        pred_inst = self.predictor.classify(logits["instances"])

        # type cls branch
        pred_type = None
        if "types" in logits.keys():
            pred_type = self.predictor.classify(logits["types"])

        # aux regression branch
        pred_aux = None
        if logits["aux"] is not None:
            pred_aux = torch_utils.tensor_to_ndarray(logits["aux"])

        return {
            "instances":pred_inst,
            "types":pred_type,
            "aux":pred_aux
        }

    def run_predictions(self):
        """
        Run predictions for all files
        """
        for path in self.images:
            fn = self.__get_fn(path)
            if self.verbose:
                print(f"Prediction for: {fn}")
        
            im = self.read_img(path)
            patcher = TilerStitcher(im.shape, (self.input_size, self.input_size, 3), self.stride_size, padding=True)

            # patch input image
            patches = patcher.extract_patches_quick(im)
           
            # Loop over batches
            pred_patches_inst = np.zeros((0, self.input_size, self.input_size, 2))
            pred_patches_type = np.zeros((0, self.input_size, self.input_size, len(self.classes)))
            pred_patches_aux = np.zeros((0, self.input_size, self.input_size, 2)) # fix channel dim 
            for batch in self.__get_batch(patches, self.batch_size):
                pred = self.gen_prediction(batch)
                pred_patches_inst = np.append(pred_patches_inst, pred["instances"], axis=0)
                
                if pred["types"] is not None:
                    pred_patches_type = np.append(pred_patches_inst, pred["types"], axis=0) 
                if pred["aux"] is not None:
                    pred_patches_aux = np.append(pred_patches_inst, pred["aux"], axis=0)

            # stitch prediction patches back to full images
            # TODO: check Nones
            res_insts = patcher.stitch_patches(pred_patches_inst, n_channels=pred_patches_inst.shape[-1])
            res_types = patcher.stitch_patches(pred_patches_type, n_channels=pred_patches_type.shape[-1])
            res_aux = patcher.stitch_patches(pred_patches_aux, n_channels=pred_patches_aux.shape[-1])

            # save results to containers
            self.soft_types[f"{fn}_soft_types"] = res_types
            self.type_maps[f"{fn}_type_map"] = np.argmax(res_types, axis=2)
            self.soft_insts[f"{fn}_soft_instances"] = res_insts
            self.aux_maps[f"{fn}_aux_map"] = res_aux

    def post_proc_inst(self,
                       soft_inst: np.ndarray,
                       soft_aux: np.ndarray = None) -> np.ndarray:
        """
        Post-processing pipeline for soft mask. i.e. threshold probability map -> post-process

        Args:
            soft_inst (np.ndarray, np.float64): 
                soft mask of instances. Shape (H, W, C)
            soft_aux: (np.ndarray): 
                aux branch output

        Returns:
            np.ndarray with instances labelled
        """
        pre_inst_map = PostProcessor.threshold(soft_inst, method=self.thresh_method, thresh=self.thresh)
        return PostProcessor.post_process(pre_inst_map, soft_inst[..., 1], soft_aux)

    def post_process(self) -> None:
        """
        Run post processing pipeline for all the predictions from the network.
        Assumes that 'run_predictions' has been run beforehand.
        """
        
        assert self.soft_insts, (f"{self.soft_insts}, No predictions found. Run 'run_predictions' first")
        
        inst_preds = [(self.soft_insts[key], self.thresh) 
                     for key in self.soft_insts.keys() 
                     if key.endswith("soft_instances")]

        # Add aux branch preds to params
        if self.soft_aux:
            inst_aux = [self.soft_aux[key] for key in self.soft_aux.keys() if key.endswith("aux_map")]
            for i, inst in enumerate(inst_preds):
                inst = inst + (inst_aux[i], )
                inst_preds[i] = inst

        # Run inst map post-processing
        with Pool() as pool:
            segs = pool.starmap(self.post_process_instmap, inst_preds)
        
        # Save inst seg results to a container
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            self.inst_maps[f"{fn}_inst_map"] = segs[i]

        # combine semantic and instance segmentations and save to a container
        if self.type_maps:
            maps = [(self.inst_maps[i].astype("int32"), self.type_maps[t].astype("uint8"))
                    for i, t in zip(self.inst_maps, self.type_maps)]

            with Pool() as pool:
                results = pool.starmap(PostProcessor.combine_inst_semantic, maps)

            for i, path in enumerate(self.images):
                fn = self.__get_fn(path)
                self.final_results[f"{fn}_seg_map"] = results[i]


    def plot_outputs(self, 
                     out_type: str,
                     ixs: Union[List[int], int] = -1,
                     gt_mask: bool = False,
                     contour: bool = False,
                     save: bool =- False) -> None:
        """
        Plot different results this instance holds.

        Args:
            out_type (str): 
                Output type to plot. One of: 
                [inst_maps, soft_insts, soft_types, seg_maps]. 
                If soft_types or soft_insts are used then 
                gt_masks and contour args will be ignored.
            ixs (List or int): 
                List of the indexes of the image files in the dataset. 
                default = -1 means all images in the data fold are 
                plotted. If dataset = "pannuke" and ixs = -1, then 25
                random images are sampled.
            gt_mask (bool): 
                plot the corresponding panoptic or instance gt next to the
                corresponding inst or panoptic map. Ignored if soft masks are plotted
            contour (bool): 
                Plots instance contours on top of the original image instead
                plotting a mask. Ignored if soft masks are plotted
            save (bool): 
                Save the plots to the results dir

        """
        # THIS IS A HUGE KLUDGE. TODO someday make this pretty
        assert out_type in ("inst_maps", "soft_insts", "soft_types", "seg_maps", "type_maps", "soft_aux")

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
                        elif out_type == "seg_maps":
                            n = name.replace("panoptic", "inst")
                        elif out_type == "type_maps":
                            n = name.replace("type", "inst")
                        inst = self.inst_maps[n]
                    kwargs.setdefault("label", inst)
                    
                    if out_type == "seg_maps":
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
        
        if out_type == "seg_maps" and contour:
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


    



    