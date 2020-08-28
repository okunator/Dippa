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
from img_processing.post_processing import *
from img_processing.augmentations import *
from img_processing.process_utils import *
from metrics.metrics import *


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
                 class_dict,
                 verbose=True):
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
                                    much memory that the inference is automatically done on CPU
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
        self.classes = class_dict

        self.n_files = len(self.images)
        self.n_classes = len(self.classes)
        self.stride_size = self.input_size//2

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
            class_dict
        )
    
    
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
    
    
    def __read_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    
    def __divide_batch(self, arr, batch_size):
        """
        Divide image array into batches similarly to DataLoader in pytorch
        """
        for i in range(0, arr.shape[0], batch_size):  
            yield arr[i:i + batch_size, ::] 
    
    
    def __smoothen_batches(self, output_batch):
        """
        Use gaussian differences from skimage to smoothen out prediction batches.
        Effectively removes checkerboard effect after the tiles are merged and eases
        thresholding from the prediction histogram.
        """
        for i in range(output_batch.shape[0]):
            for c in range(self.n_classes):
                # class 0 = background
                output_batch[i, c, ...] = difference_of_gaussians(output_batch[i, c, ...], 1, 50)
                output_batch[i, c, ...] = torch.from_numpy(output_batch[i, c, ...]).relu().cpu().numpy()
                output_batch[i, c, ...] = difference_of_gaussians(output_batch[i, c, ...], 1, 10)
                output_batch[i, c, ...] = torch.from_numpy(output_batch[i, c, ...]).sigmoid().cpu().numpy()
            
        return output_batch
    
            
    def __extract_patches(self, im):
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
    
    
    def __ensemble_predictions(self, batch):
        """
        ttatch test time augmentation
        """
        # TODO
        pass
        #masks = []
        #for transformer in self.transforms:
        #    # augment image
        #    print(output_batch.shape)
        #    augmented_image = transformer.augment_image(output_batch)
        #    # pass to model
        #    model_output = self.model(augmented_image)
        #    print(model_output.shape)
        #    # reverse augmentation for mask and label
        #    deaug_mask = transformer.deaugment_mask(model_output)
        #    print(deaug_mask.shape)
        #    # save results
        #    masks.append(deaug_mask)
        
        #    del augmented_image
        #    del deaug_mask
        #    del model_output
        #    torch.cuda.empty_cache()

        #del output_batch
        #torch.cuda.empty_cache()
        #print(torch.mean(masks).shape)
        #return torch.mean(torch.cat(masks, dim=0))

    
    
    def __predictions(self, im_patches):
        # Use model to predict batches
        pred_patches = np.zeros((0, self.n_classes, self.input_size, self.input_size))
        for batch in self.__divide_batch(im_patches, self.batch_size):
            # shape for pytorch and predict
            batch_d = torch.from_numpy(batch.transpose(0, 3, 1, 2))
            if torch.cuda.is_available() and not self.test_time_augs:
                batch_d = batch_d.type("torch.cuda.FloatTensor")
            else:
                batch_d = batch_d.type("torch.FloatTensor")
            
            if self.test_time_augs:
                pred_batch = self.tta_model(batch_d)
            else:
                pred_batch = self.model(batch_d)
                
            # back to cpu
            pred_batch = pred_batch.detach().cpu().numpy()
            
            if self.smoothen:
                pred_batch = self.__smoothen_batches(pred_batch)

            pred_patches = np.append(pred_patches, pred_batch, axis=0)

        pred_patches = pred_patches.transpose((0, 2, 3, 1))
        return pred_patches
        
    
    def __create_prediction(self, pred_patches, im_shape, patches_shape):
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

    
    def clear_predictions(self):
        """
        Clear predictions OrderedDict
        """
        self.soft_maps.clear()
    
    
    def __get_fn(self, path):
        return path.split("/")[-1][:-4]
    
    
    def run(self):
        """
        Do inference on the given dataset, with the pytorch lightining model that 
        has been used for training. See lightning_model.py and Train_lightning.ipynb.
        """
        
        # Put SegModel to gpu
        self.model.cuda() if torch.cuda.is_available() and not self.test_time_augs else self.model.cpu()    
        self.model.model.eval()
        torch.no_grad()
        
        if len(self.soft_maps) > 0:
            print("Clearing previous predictions")
            clear_predictions()
                    
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            if self.verbose:
                print(f"Prediction for: {fn}")
                
            im = self.__read_img(path)
            im_patches, patches_shape = self.__extract_patches(im)
            pred_patches = self.__predictions(im_patches)
            result_pred = self.__create_prediction(pred_patches, im.shape, patches_shape)
            nuc_map = result_pred[..., 1]
            bg_map = result_pred[..., 0]
            self.soft_maps[f"{fn}_nuc_map"] = nuc_map
            self.soft_maps[f"{fn}_bg_map"] = bg_map
            
    
    def __rm_model(self):
        pass
        
                
    def plot_predictions(self):
        """
        Plot the probability maps after running inference.
        """
        
        assert len(self.soft_maps) != 0, "No predictions found"
    
        fig, axes = plt.subplots(self.n_files, 3, figsize=(65, self.n_files*12))
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
            
            
    def plot_histograms(self):
        """
        Plot the histograms of the probability maps after running inference.
        """
        
        assert len(self.soft_maps) != 0, "No predictions found"
        
        figg, axes = plt.subplots(self.n_files//3, 4, figsize=(30,15))
        axes = axes.flatten()
        for i, path in enumerate(self.images):
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
        
        assert len(self.soft_maps) != 0, "No predictions found"
        
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
        
        assert self.inst_maps, f"{self.inst_maps}, No instance maps found. Run post_processing first!"
    
        fig, axes = plt.subplots(self.n_files, 3, figsize=(65, self.n_files*12))
        fig.tight_layout(w_pad=4, h_pad=4)
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            gt = scipy.io.loadmat(self.gt_masks[i])
            gt = gt["inst_map"]
            inst_map = self.inst_maps[f"{fn}_inst_map"]
            nuc_map = self.soft_maps[f"{fn}_nuc_map"]
            
            axes[i][0].imshow(nuc_map, interpolation="none")
            axes[i][0].axis("off")
            
            axes[i][1].imshow(inst_map, interpolation="none")
            axes[i][1].axis("off")

            axes[i][2].imshow(gt, interpolation="none")
            axes[i][2].axis("off")
            
    
    def _post_process_pipeline(self, prob_map, thresh=2, 
                               postproc_func=inv_dist_watershed):
        # threshold first
        if self.smoothen:
            # Find the steepest drop in the histogram
            hist, hist_centers = histogram(prob_map)
            d = np.diff(hist)
            b = d == np.min(d)
            b = np.append(b, False) # append one ince np.diff loses one element in arr
            thresh = hist_centers[b] + 0.05
            mask = naive_thresh_logits(prob_map, thresh)
        else:
            mask = naive_thresh(prob_map, thresh)
                
        # post-processing after thresholding
        mask = postproc_func(mask, 15)
        mask[mask > 0] = 1
        mask = ndi.binary_fill_holes(mask)
        mask = ndi.label(mask)[0]        
        return mask
    
    
    def _compute_metrics(self, true, pred):
        # Count scores for each file
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
        
        preds = [self.soft_maps[key] for key in self.soft_maps.keys() if key.endswith("nuc_map")]
        
        segs = None
        with Pool() as pool:
            segs = pool.map(self._post_process_pipeline, preds)
        
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
        gts = [scipy.io.loadmat(f)["inst_map"].astype("uint16") for f in self.gt_masks]
        params_list = list(zip(gts, inst_maps))
        
        metrics = None
        with Pool() as pool:
            metrics = pool.starmap(self._compute_metrics, params_list)
        
        for i, path in enumerate(self.images):
            fn = self.__get_fn(path)
            self.metrics[f"{fn}_metrics"] = metrics[i]
            
        # Create pandas df of the result metrics
        score_df = pd.DataFrame(self.metrics).transpose()
        score_df.loc["averages for the test set"] = score_df.mean(axis=0)
        return score_df
    
    
    def __plot_pannuke(self):
        # TODO
        pass
    
    
    def __infer_pannukse(self):
        # TODO
        pass
    

            
            
            
        
        
            