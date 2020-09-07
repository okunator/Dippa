import cv2
import torch
import scipy.io
import numpy as np
import sklearn.feature_extraction.image
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.filters import difference_of_gaussians
from skimage.exposure import histogram
from collections import OrderedDict
from multiprocessing import Pool

from utils.file_manager import ProjectFileManager
from img_processing.process_utils import remap_label
from img_processing.post_processing import (activation, 
                                            naive_thresh_logits, 
                                            smoothed_thresh, 
                                            inv_dist_watershed)

from img_processing.augmentations import (tta_augs,
                                          tta_deaugs,
                                          tta_five_crops,
                                          resize)
from metrics.metrics import (PQ,
                             AJI,
                             AJI_plus,
                             DICE2,
                             split_and_merge)


class Inferer(ProjectFileManager):
    def __init__(self, 
                 model,
                 dataset,
                 data_dirs,
                 database_root,
                 phases,
                 batch_size,
                 input_size,
                 smoothen,
                 fold,
                 test_time_augs,
                 threshold,
                 class_dict,
                 verbose):
        """
        Inferer for any of the models that are trained with lightning framework 
        in this project (defined in lightning_model.py)
        
        Args: 
            model (SegModel) : SegModel that has been used in training.
                               See: Train_lightning.ipynb for example how to define it
            dataset (str) : one of ("kumar", "consep", "pannuke", "other")
            data_dirs (dict) : dictionary of directories containing masks and images. Keys of this
                               dict must be the same as ("kumar", "consep", "pannuke", "other")
            database_root_dir (str) : directory where the databases are written
            phases (list) : list of the phases (["train", "valid", "test"] or ["train", "test"])
            batch_size (int) : Number of input patches used for every iteration
            input_size (int) : Size of the input patch that is fed to the network
            smoothen (bool) : Use Gaussian differences to smoothen every predicted patch. This will
                              get rid of the checkerboard pattern after the patches are stithced to
                              full size images and makes the thresholding of the soft masks trivial.
            fold (str) : One of ("train", "valid", "test"). Do predictions on one of these data folds.
                         Naturally "test" is the one to use for results
            test_time_augs (bool) : apply test time augmentations with ttatch library. TTA takes so
                                    much memory that the inference is automatically done on CPU.
                                    Note that TTA increases inferece time significantly.
            threshold (float) : threshold for the softmasks that have values b/w [0, 1]
            class_dict (Dict) : the dict specifying pixel classes. e.g. {"background":0,"nuclei":1}
            verbose (bool) : wether or not to print the progress of running inference
        """
        
        
        super(Inferer, self).__init__(dataset, data_dirs, database_root, phases)
        self.model = model
        self.batch_size = batch_size
        self.input_size = input_size
        self.smoothen = smoothen
        self.verbose = verbose
        self.fold = fold
        self.test_time_augs = test_time_augs
        self.thresh = threshold
        self.classes = class_dict
        self.verbose = verbose
        
        # init containers for resluts
        self.soft_maps = OrderedDict()
        self.metrics = OrderedDict()
        self.inst_maps = OrderedDict()
    
    @classmethod
    def from_conf(cls, model, conf):
        model = model
        dataset = conf["dataset"]["args"]["dataset"]
        data_dirs = conf["paths"]["data_dirs"]
        database_root = conf["paths"]["database_root_dir"]
        phases = conf["dataset"]["args"]["phases"]
        batch_size = conf["training_args"]["batch_size"]
        input_size = conf["patching_args"]["input_size"]
        smoothen = conf["inference_args"]["smoothen"]
        fold = conf["inference_args"]["data_fold"]
        test_time_augs = conf["inference_args"]["test_time_augmentation"]
        thresh = conf["inference_args"]["threshold"]
        verbose = conf["inference_args"]["verbose"]
        class_type = conf["dataset"]["args"]["class_types"]
        class_dict = conf["dataset"]["class_dicts"][class_type] # clumsy
        
        return cls(
            model,
            dataset,
            data_dirs,
            database_root,
            phases,
            batch_size,
            input_size,
            smoothen,
            fold,
            test_time_augs,
            thresh,
            class_dict,
            verbose
        )
    
    
    @property
    def stride_size(self):
        return self.input_size//2
    
        
    @property
    def tta_model(self):
        """
        test time augmentations defined in augmentations.py
        """
        return tta.SegmentationTTAWrapper(self.model, tta_transforms())
    
    
    @property
    def images(self):
        # TODO: check the phases. if there is no valid phase, then notify
        return self.data_folds[self.fold]["img"]
    
    
    @property
    def gt_masks(self):
        return self.data_folds[self.fold]["mask"]
    
    
    @property
    def get_model(self, model="best"):
        #TODO: get the best or the last model based on config
        assert model in ("best", "last"), "model param needs to be one of ('best', 'last')"
    
    
    def __read_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    
    def __read_mask(self, path):
        return scipy.io.loadmat(path)["inst_map"].astype("uint16")
    
    
    def __get_fn(self, path):
        return path.split("/")[-1][:-4]
    
    
    def __to_device(self, tensor):
        if torch.cuda.is_available():
            tensor = tensor.type("torch.cuda.FloatTensor")
        else:
            tensor = tensor.type("torch.FloatTensor")
        return tensor
    
    
    def __predict_batch(self, batch, batch_size):
        """
        Do a prediction for a batch of size self.batch_size, or for a single patch i.e 
        batch_size = 1. If tta is used or dataset is pannuke then batch_size = 1. Otherwise
        batch size will be self.batch_size
        """
        if batch_size == 1:
            tensor = self.__to_device(torch.from_numpy(batch.transpose(2, 0, 1))[None, ...])
            out_batch = self.model(tensor).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        else:
            tensor = self.__to_device(torch.from_numpy(batch.transpose(0, 3, 1, 2)))
            # out_batch = self.tta_model(batch) # ttach tta prediction
            out_batch = self.model(tensor).detach().cpu().numpy()
        return out_batch
    

    def __ensemble_predict(self, patch):
        """
        Own implementation of tta ensemble prediction pipeline (memory issues with ttach).
        This will loop every patch in a batch and do an ensemble prediction for every 
        single patch.
        
        Following instructions of 'beneficial augmentations' from:
        
        "Towards Principled Test-Time Augmentation"
        D. Shanmugam, D. Blalock, R. Sahoo, J. Guttag 
        link : https://dmshanmugam.github.io/pdfs/icml_2020_testaug.pdf
        
        1. vflip, hflip and transpose and rotations
        2. custom fivecrops aug
        
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
        
        soft_masks = []
        
        # flip ttas
        for aug, deaug in zip(tta_augs(), tta_deaugs()):
            aug_input = aug(image = patch)
            aug_output = self.__predict_batch(aug_input["image"], 1)
            deaug_output = deaug(image = aug_output)
            soft_masks.append(deaug_output["image"])
            
        # five crops tta
        scale_up = resize(patch.shape[0], patch.shape[1])
        scale_down = resize(patch.shape[0]//2, patch.shape[1]//2)
        out = np.zeros((patch.shape[0], patch.shape[1], len(self.classes)))
        for crop in tta_five_crops(patch):
            cropped_im = crop(image = patch)
            scaled_im = scale_up(image = cropped_im["image"])
            output = self.__predict_batch(scaled_im["image"], 1)
            downscaled_out = scale_down(image = output)
            out[crop.y_min:crop.y_max, crop.x_min:crop.x_max] = downscaled_out["image"]
            soft_masks.append(out)
            
        # take the mean of all the predictions
        return np.asarray(soft_masks).mean(axis=0).transpose(2, 0, 1)
    
    
    def __smoothen(self, prob_map):
        """
        Use gaussian differences from skimage to smoothen out prediction.
        Effectively removes checkerboard effect after the tiles are merged and eases
        thresholding from the prediction histogram.
        
        prob_map.shape = (class, width, height) (torch tensor shape)
        """
        for c in range(len(self.classes)):
            prob_map[c, ...] = difference_of_gaussians(prob_map[c, ...], 1, 50)
            prob_map[c, ...] = activation(prob_map[c, ...], 'relu')
            prob_map[c, ...] = difference_of_gaussians(prob_map[c, ...], 1, 10)
            prob_map[c, ...] = activation(prob_map[c, ...], 'sigmoid')
        return prob_map
    
    
    def __divide_batch(self, arr, batch_size):
        """
        Divide patched image array into batches similarly to DataLoader in pytorch
        """
        for i in range(0, arr.shape[0], batch_size):  
            yield arr[i:i + batch_size, ::] 
    
    
    def __predict_patches(self, im_patches):
        """
        Divide patched image to batches and process and run predictions for batches.
        Pannuke imgs are not divided to patches and are utilized as is.
        """
        pred_patches = np.zeros((0, len(self.classes), self.input_size, self.input_size))
        for batch in self.__divide_batch(im_patches, self.batch_size):
            
            pred_batch = np.zeros(
                (batch.shape[0], len(self.classes), self.input_size, self.input_size)
            )
            
            if not self.test_time_augs:
                pred_batch = self.__predict_batch(batch, self.batch_size)

            for i in range(batch.shape[0]):
                if self.test_time_augs:
                    pred_batch[i, ...] = self.__ensemble_predict(batch[i, ...])

                if self.smoothen:
                    pred_batch[i, ...] = self.__smoothen(pred_batch[i, ...])
                else:
                    pred_batch[i, ...] = activation(pred_batch[i, ...], "sigmoid")
                    
            pred_patches = np.append(pred_patches, pred_batch, axis=0)

        pred_patches = pred_patches.transpose((0, 2, 3, 1))
        return pred_patches
    
    
    def __extract_patches(self, im):
        """
        Extract network input sized patches from images bigger than the network input size
        """
        # add reflection padding
        pad = self.stride_size//2
        io = np.pad(im, [(pad, pad), (pad, pad), (0, 0)], mode="reflect")

        # add extra padding to match an exact multiple of 32 (unet) patch size, 
        extra_pad_row = int(np.ceil(io.shape[0] / self.input_size)*self.input_size - io.shape[0])
        extra_pad_col = int(np.ceil(io.shape[1] / self.input_size)*self.input_size - io.shape[1])
        io = np.pad(io, [(0, extra_pad_row), (0, extra_pad_col), (0, 0)], mode="constant")
        
        # extract the patches from input images
        arr_out = sklearn.feature_extraction.image.extract_patches(
            io, (self.input_size, self.input_size, 3), self.stride_size
        )
        
        # shape the dimensions to correct sizes for pytorch model
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1, self.input_size, self.input_size, 3)
        return arr_out, arr_out_shape
        
    
    def __stitch_patches(self, pred_patches, im_shape, patches_shape):
        """
        Back stitch all the soft map patches to full size img
        """
        #turn from a single list into a matrix of tiles
        pred_patches = pred_patches.reshape(
            patches_shape[0], 
            patches_shape[1], 
            self.input_size, 
            self.input_size,
            pred_patches.shape[3]
        )
        
        # remove the padding from each tile, we only keep the center
        pad = self.stride_size//2
        pred_patches = pred_patches[:, :, pad:-pad, pad:-pad, :]
    
        # turn all the tiles into an image
        pred = np.concatenate(np.concatenate(pred_patches, 1), 1)
    
        # incase there was extra padding to get a multiple of patch size, remove that as well
        # remove paddind, crop back
        pred = pred[0:im_shape[0], 0:im_shape[1], :]
        return pred

    
    def run(self):
        """
        Do inference on the given dataset, with the pytorch lightining model that 
        has been used for training. See lightning_model.py and Train_lightning.ipynb.
        """
        
        # Put SegModel to gpu
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()    
        self.model.model.eval()
        torch.no_grad()
        
        if len(self.soft_maps) > 0:
            print("Clearing previous predictions")
            self.clear_predictions()
                    
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            if self.verbose:
                print(f"Prediction for: {fn}")
                
            im = self.__read_img(path)
            if self.dataset == "pannuke":
                result_pred = self.__predict_patches(im[None, ...])
                result_pred = result_pred.squeeze()
            else:
                im_patches, patches_shape = self.__extract_patches(im)
                pred_patches = self.__predict_patches(im_patches)
                result_pred = self.__stitch_patches(pred_patches, im.shape, patches_shape)
            
            nuc_map = result_pred[..., 1]
            bg_map = result_pred[..., 0]
            self.soft_maps[f"{fn}_nuc_map"] = nuc_map
            self.soft_maps[f"{fn}_bg_map"] = bg_map
            
    
    def _post_process_pipeline(self, prob_map, thresh, 
                               postproc_func=inv_dist_watershed):
        # threshold first
        if self.smoothen:
            mask = smoothed_thresh(prob_map)
        else:
            mask = naive_thresh_logits(prob_map, thresh)
                
        # post-processing after thresholding
        return postproc_func(mask)        
    
    
    def _compute_metrics(self, true, pred):
        # Count scores for each file if gt has annotations
        if len(np.unique(true)) > 1: 
            pq = PQ(remap_label(true), remap_label(pred))
            aji = AJI(remap_label(true), remap_label(pred))
            aji_p = AJI_plus(remap_label(true), remap_label(pred))
            dice2 = DICE2(remap_label(true), remap_label(pred))
            splits, merges = split_and_merge(remap_label(true), remap_label(pred))

            return {
                "AJI": aji, 
                "AJI plus": aji_p, 
                "DICE2": dice2, 
                "PQ": pq["pq"], # panoptic quality
                "SQ": pq["sq"], # segmentation quality
                "DQ": pq["dq"], # Detection quality i.e. F1-score
                "inst Sensitivity": pq["sensitivity"],
                "inst Precision": pq["precision"],
                "splits":splits,
                "merges":merges
            }

            
    def post_process(self):
        """
        Run post processing pipeline for all the predictions given by the network
        """
        
        assert self.soft_maps, f"{self.soft_maps}, No predictions found. Run predictions first"
        self.model.cpu() # put model to cpu (avoid pool errors)
        
        preds = [(self.soft_maps[key], self.thresh) 
                 for key in self.soft_maps.keys() 
                 if key.endswith("nuc_map")]
        
        segs = []
        if self.dataset == "pannuke":
            # Pool fails when using pannuke for some reason 
            for pred in preds:
                segs.append(self._post_process_pipeline(pred, self.thresh))
        else:     
            with Pool() as pool:
                segs = pool.starmap(self._post_process_pipeline, preds)
        
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            self.inst_maps[f"{fn}_inst_map"] = segs[i]
            
            
    def benchmark(self, save=False):
        """
        Run benchmarking metrics for all of the files in the dataset
        Masks are converted to uint16 for memory purposes
        """
        
        assert self.inst_maps, f"{self.inst_maps}, No instance maps found. Run post_processing first!"
        
        inst_maps = [self.inst_maps[key].astype("uint16") for key in self.inst_maps.keys()]
        gts = [self.__read_mask(f) for f in self.gt_masks]
        params_list = list(zip(gts, inst_maps))
        
        metrics = []
        if self.dataset == "pannuke":
            # Pool fails when using pannuke for some reason
            for true, pred in params_list:
                metrics.append(self._compute_metrics(true, pred))
        else:
            with Pool() as pool:
                metrics = pool.starmap(self._compute_metrics, params_list)
        
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            self.metrics[f"{fn}_metrics"] = metrics[i]
            
        # Create pandas df of the result metrics
        score_df = pd.DataFrame(self.metrics).transpose()
        score_df.loc["averages for the set"] = score_df.mean(axis=0)
        return score_df
    
    
    def clear_predictions(self):
        """
        Clear predictions OrderedDict
        """
        self.soft_maps.clear()
    
    
    def __plot_pannuke(self):
        # TODO
        pass
    
    
    def plot_predictions(self):
        """
        Plot the probability maps after running inference.
        """
        
        assert len(self.soft_maps) != 0, "No predictions found"
    
        fig, axes = plt.subplots(len(self.images), 3, figsize=(65, len(self.images)*12))
        fig.tight_layout(w_pad=4, h_pad=4)
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            gt = scipy.io.loadmat(self.gt_masks[i])
            gt = gt["inst_map"]
            nuc_map = self.soft_maps[f"{fn}_nuc_map"]
            bg_map = self.soft_maps[f"{fn}_bg_map"]
            axes[i][0].imshow(nuc_map, interpolation="none")
            axes[i][0].axis("off")

            axes[i][1].imshow(bg_map, interpolation="none")
            axes[i][1].axis("off")

            axes[i][2].imshow(gt, interpolation="none")
            axes[i][2].axis("off")


    def __sample_idxs(self):
        """
        Sample paths of images given a list of image paths
        """
        return np.random.randint(low = 0, high=len(self.images), size=25)
    
    
    def plot_histograms(self):
        """
        Plot the histograms of the probability maps after running inference.
        """
        
        assert self.soft_maps, "No predictions found"
        idxs = self.__sample_idxs()
        images = np.asarray(self.images)[idxs] if self.dataset == "pannuke" else self.images 
        
        figg, axes = plt.subplots(len(images)//3, 4, figsize=(30,15))
        axes = axes.flatten()
        for i, path in enumerate(images):
            fn = self.__get_fn(path)
            nuc_map = self.soft_maps[f"{fn}_nuc_map"]
            hist, hist_centers = histogram(nuc_map)
            axes[i].plot(hist_centers, hist, lw=2)
            
            
    def save_outputs(self, output_dir):
        """
        Save predictions to .mat files (python dictionary). Key for accessing after
        reading one file is "pred_map":
        
        f = scipy.io.loadmat(file)
        f = f["pred_map"]
        
        Args:
            output_dir (str) : path to the directory where predictions are saved.
        """
        
        assert self.soft_maps, "No predictions found"
        
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            new_fn = fn + "_pred_map.mat"
            print("saving: ", fn)
            nuc_map = self.soft_maps[f"{fn}_nuc_map"]
            # scipy.io.savemat(new_fn, mdict={"pred_map": nuc_map})
            
            
    def plot_segmentations(self):
        """
        Plot the binary segmentations after running post_processing.
        """
        #TODO: Fix take of gt masks here
        assert self.inst_maps, f"{self.inst_maps}, No instance maps found. Run post_processing first!"
        idxs = self.__sample_idxs()
        images = np.asarray(self.images)[idxs] if self.dataset == "pannuke" else self.images
        gt_masks = np.asarray(self.gt_masks)[idxs] if self.dataset == "pannuke" else self.gt_masks
    
        fig, axes = plt.subplots(len(images), 4, figsize=(65, len(images)*12))
        fig.tight_layout(w_pad=4, h_pad=4)
        for i, path in enumerate(images):
            fn = self.__get_fn(path)
            im = self.__read_img(images[i])
            gt = self.__read_mask(gt_masks[i])
            inst_map = self.inst_maps[f"{fn}_inst_map"]
            nuc_map = self.soft_maps[f"{fn}_nuc_map"]
            
            axes[i][0].imshow(im, interpolation="none")
            axes[i][0].axis("off")
            
            axes[i][1].imshow(nuc_map, interpolation="none")
            axes[i][1].axis("off")
            
            axes[i][2].imshow(inst_map, interpolation="none")
            axes[i][2].axis("off")

            axes[i][3].imshow(gt, interpolation="none")
            axes[i][3].axis("off")
    

            
            
            
        
        
            