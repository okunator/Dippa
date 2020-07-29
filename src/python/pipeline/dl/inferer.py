import os
import sys
import glob

import cv2
import torch
import scipy.io
import numpy as np
import sklearn.feature_extraction.image
import pandas as pd
import matplotlib.pyplot as plt

from skimage.filters import difference_of_gaussians
from skimage.exposure import histogram
from collections import OrderedDict
from multiprocessing import Pool
from joblib import Parallel, delayed

from .post_processing import *
from .metrics import *


class Inferer(object):
    def __init__(self, batch_size, patch_size, model, n_classes, 
                 smoothen=True, dataset='kumar', verbose=True):
        """
        Inferer for any of the models that are trained with lightning framework defined in lightning_model.py
        inference can be done for these datasets: Kumar, CoNSep and PanNuke dataset. Datasets need dto be in right format.
        Documentation in progress...
        
        Args: 
            batch_size (int) : batch_size used in the DataLoader when training
            patch_size (int) : The size (width x height) of an image patch that goes through the network. Usually for U-net family 
                               models this is a multiple of 32. For now we have used 224x224 for smp models.
            model (SegModel) : SegModel that has been used in training. See: Train_lightning.ipynb for example how to define it
            n_classes (int)  : number of classes the model outputs. For binary segmentation n_classes = 2 (background vs. nuclei)
            smoothen (bool)  : wether to use gaussian differences to smooth out the tiles of predictions. (Helpful in downstream)
            dataset (str)    : one of ('kumar', 'consep', 'pannuke'). These are 3 benchmarking datasets for this project
            verbose (bool)   : wether or not to print the progress of running inference
        """
        
        assert dataset in ('kumar', 'consep', 'pannuke'), "dataset param not in ('kumar', 'consep', 'pannuke')"
        
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.model = model
        self.n_classes = n_classes
        self.dataset = dataset
        self.smoothen = smoothen
        self.verbose = verbose
        self.stride_size = self.patch_size//2
        
        # Dataset directories
        self.data_dirs = {
            'kumar':{
                'kumar_im':"../../../../datasets/Nucleisegmentation-Kumar/test/Images",
                'kumar_gt':"../../../../datasets/Nucleisegmentation-Kumar/test/Labels",
                'gt_sfx':'/*.mat',
                'im_sfx':'/*.tif'
            },
            'consep':{
                'consep_im':"../../../../datasets/Nucleisegmentation-CoNSeP/test/Images",
                'consep_gt':"../../../../datasets/Nucleisegmentation-CoNSeP/test/Labels",
                'gt_sfx':'/*.mat',
                'im_sfx':'/*.png'
            },
            'pannuke': {
                'pannuke_im':"../../../../datasets/Nucleisegmentation-PanNuke/test/Images",
                'pannuke_gt':"../../../../datasets/Nucleisegmentation-PanNuke/test/Labels",
                'gt_sfx':'/*.mat',
                'im_sfx':'/*.png'
            }
        }
        
        self.dir_im = self.data_dirs[self.dataset][f"{self.dataset}_im"]
        self.dir_gt = self.data_dirs[self.dataset][f"{self.dataset}_gt"]
        im_sfx = self.data_dirs[self.dataset]["im_sfx"]
        gt_sfx = self.data_dirs[self.dataset]["gt_sfx"]
        
        self.files = sorted(glob.glob(self.dir_im + im_sfx))
        self.gt_masks = sorted(glob.glob(self.dir_gt + gt_sfx))
        self.n_files = len(self.files)

        self.outputs = OrderedDict()
        self.metrics = OrderedDict()
        self.inst_maps = OrderedDict()
           
            
    def _divide_batch(self, arr, batch_size):
        """
        Divide image array into batches similarly to DataLoader in pytorch
        """
        for i in range(0, arr.shape[0], batch_size):  
            yield arr[i:i + batch_size, ::] 
    
    
    def _smoothen_batches(self, output_batch):
        """
        Use differences of gaussians from skimage to smoothen out prediction batches.
        Effectively removes checkerboard effect after the tiles are merged and eases thresholding
        from the prediction histogram.
        """
        for i in range(output_batch.shape[0]):
            for c in range(self.n_classes):
                # class 0 = background
                output_batch[i, c, ...] = difference_of_gaussians(output_batch[i, c, ...], 1, 50)
                output_batch[i, c, ...] = torch.from_numpy(output_batch[i, c, ...]).relu().cpu().numpy()
                output_batch[i, c, ...] = difference_of_gaussians(output_batch[i, c, ...], 1, 10)
                output_batch[i, c, ...] = torch.from_numpy(output_batch[i, c, ...]).sigmoid().cpu().numpy()
            
        return output_batch
    
    
    def _read_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        
    def _extract_patches(self, im):
        # add reflection padding
        pad = self.stride_size//2
        io = np.pad(im, [(pad, pad), (pad, pad), (0, 0)], mode="reflect")
                
        # add extra padding to match an exact multiple of 32 (unet) patch size, 
        extra_pad_row = int(np.ceil(io.shape[0] / self.patch_size)*self.patch_size - io.shape[0])
        extra_pad_col = int(np.ceil(io.shape[1] / self.patch_size)*self.patch_size - io.shape[1])
        io = np.pad(io, [(0, extra_pad_row), (0, extra_pad_col), (0, 0)], mode="constant")
        
        # extract the patches from input images
        arr_out = sklearn.feature_extraction.image.extract_patches(
            io, (self.patch_size, self.patch_size, 3), self.stride_size
        )
        
        # shape the dimensions to correct sizes for pytorch model
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1, self.patch_size, self.patch_size, 3)
        return arr_out, arr_out_shape
    
    
    def _predictions(self, im_patches):
        # Use model to predict batches
        pred_patches = np.zeros((0, self.n_classes, self.patch_size, self.patch_size))
        for batch in self._divide_batch(im_patches, self.batch_size):
            # shape for pytorch and predict
            batch_d = torch.from_numpy(batch.transpose(0, 3, 1, 2))
            if torch.cuda.is_available():
                batch_d = batch_d.type('torch.cuda.FloatTensor')
            else:
                batch_d = batch_d.type('torch.FloatTensor')
            
            pred_batch = self.model(batch_d) 
            pred_batch = pred_batch.detach().cpu().numpy()
            
            if self.smoothen:
                pred_batch = self._smoothen_batches(pred_batch)

            pred_patches = np.append(pred_patches, pred_batch, axis=0)

        pred_patches = pred_patches.transpose((0, 2, 3, 1))
        return pred_patches
        
    
    def _create_prediction(self, pred_patches, im_shape, patches_shape):
        #turn from a single list into a matrix of tiles
        pred_patches = pred_patches.reshape(
            patches_shape[0], 
            patches_shape[1], 
            self.patch_size, 
            self.patch_size,
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
        self.outputs.clear()
    
    
    def _get_fn(self, path):
        return path.split('/')[-1][:-4]
    
    
    def run(self):
        """
        Do inference on the given dataset, with the pytorch lightining model that has been used for training.
        See lightning_model.py and Train_lightning.ipynb.
        """
        
        # Put SegModel to gpu
        self.model.cuda() if torch.cuda.is_available() else self.model.cpu()    
        self.model.model.eval()
        
        if len(self.outputs) > 0:
            print('Clearing previous predictions')
            clear_predictions()
                    
        for i, path in enumerate(self.files):
            fn = self._get_fn(path)
            if self.verbose:
                print(f"Prediction for: {fn}")
                
            im = self._read_img(path)
            im_patches, patches_shape = self._extract_patches(im)
            pred_patches = self._predictions(im_patches)
            result_pred = self._create_prediction(pred_patches, im.shape, patches_shape)
            nuc_map = result_pred[..., 1]
            bg_map = result_pred[..., 0]
            self.outputs[f"{fn}_nuc_map"] = nuc_map
            self.outputs[f"{fn}_bg_map"] = bg_map
            
    
    def _rm_model(self):
        pass
        
                
    def plot_predictions(self):
        """
        Plot the probability maps after running inference.
        """
        
        assert len(self.outputs) != 0, "No predictions found"
    
        fig, axes = plt.subplots(self.n_files, 3, figsize=(65, self.n_files*12))
        fig.tight_layout(w_pad=4, h_pad=4)
        for i, path in enumerate(self.files):
            fn = self._get_fn(path)
            gt = scipy.io.loadmat(self.gt_masks[i])
            gt = gt['inst_map']
            nuc_map = self.outputs[f"{fn}_nuc_map"]
            bg_map = self.outputs[f"{fn}_bg_map"]
            axes[i][0].imshow(nuc_map, interpolation='none')
            axes[i][0].axis('off')

            axes[i][1].imshow(bg_map, interpolation='none')
            axes[i][1].axis('off')

            axes[i][2].imshow(gt, interpolation='none')
            axes[i][2].axis('off')
            
            
    def plot_histograms(self):
        """
        Plot the histograms of the probability maps after running inference.
        """
        
        assert len(self.outputs) != 0, "No predictions found"
        
        figg, axes = plt.subplots(self.n_files//3, 4, figsize=(30,15))
        axes = axes.flatten()
        for i, path in enumerate(self.files):
            fn = self._get_fn(path)
            nuc_map = self.outputs[f"{fn}_nuc_map"]
            hist, hist_centers = histogram(nuc_map)
            axes[i].plot(hist_centers, hist, lw=2)
            
            
    def save_outputs(self, output_dir):
        """
        Save predictions to .mat files (python dictionary). Key for accessing after reading one file
        is 'pred_map'
        
        f = scipy.io.loadmat(file)
        f = f['pred_map']
        
        Args:
            output_dir (str) : path to the directory where predictions are saved.
        """
        
        assert len(self.outputs) != 0, "No predictions found"
        
        for i, path in enumerate(self.files):
            fn = self._get_fn(path)
            new_fn = fn + '_pred_map.mat'
            print('saving: ', fn)
            nuc_map = self.outputs[f"{fn}_nuc_map"]
            # scipy.io.savemat(new_fn, mdict={'pred_map': nuc_map})
            
            
    def plot_segmentations(self):
        """
        Plot the binary segmentations after running post_processing.
        """
        
        assert any([key for key in self.outputs.keys() if key.endswith("inst_map")]), "Run post_processing first!"
    
        fig, axes = plt.subplots(self.n_files, 3, figsize=(65, self.n_files*12))
        fig.tight_layout(w_pad=4, h_pad=4)
        for i, path in enumerate(self.files):
            fn = self._get_fn(path)
            gt = scipy.io.loadmat(self.gt_masks[i])
            gt = gt['inst_map']
            inst_map = self.outputs[f"{fn}_inst_map"]
            nuc_map = self.outputs[f"{fn}_nuc_map"]
            
            axes[i][0].imshow(nuc_map, interpolation='none')
            axes[i][0].axis('off')
            
            axes[i][1].imshow(inst_map, interpolation='none')
            axes[i][1].axis('off')

            axes[i][2].imshow(gt, interpolation='none')
            axes[i][2].axis('off')
            
    
    def _post_process_pipeline(self, prob_map, thresh=2, postproc_func=inv_dist_watershed):
        
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
        pq = PQ(true, pred)
        aji = AJI(true, pred)
        aji_p = AJI_plus(true, pred)
        dice2 = DICE2(true, pred)
        splits, merges = split_and_merge(true, pred)
        
        return {
            'AJI': aji, 
            'AJI plus': aji_p, 
            "DICE2": dice2, 
            "PQ": pq['pq'], # panoptic quality
            "SQ": pq['sq'], # segmentation quality
            "DQ": pq['dq'], # Detection quality i.e. F1-score
            "inst Sensitivity": pq['sensitivity'], # Sensitivity in detecting matching nucleis
            "inst Precision": pq['precision'],  # Specificity in detecting matching nucleis
            "splits":splits,
            "merges":merges
        }

            
    def post_process(self):
        """
        Run post processing pipeline for all the predictions given by the network
        """
        
        assert len(self.outputs) != 0, "No predictions found"
        self.model.cpu()
        
        preds = [self.outputs[key] for key in self.outputs.keys() if key.endswith("nuc_map")]
        
        segs = None
        with Pool() as pool:
            segs = pool.map(self._post_process_pipeline, preds)
        
        for i, path in enumerate(self.files):
            fn = self._get_fn(path)
            self.outputs[f"{fn}_inst_map"] = segs[i]
            
            
    def benchmark(self, save=False):
        """
        Run benchmarking metrics for all of the files in the dataset
        """
        
        assert any([key for key in self.outputs.keys() if key.endswith("inst_map")]), "Run post_processing first!"
        
        inst_maps = [self.outputs[key] for key in self.outputs.keys() if key.endswith("inst_map")]
        gts = [scipy.io.loadmat(f)['inst_map'].astype("uint32") for f in self.gt_masks]
        params_list = list(zip(gts, inst_maps))
        
        metrics = None
        with Pool() as pool:
            metrics = pool.starmap(self._compute_metrics, params_list)
        
        for i, path in enumerate(self.files):
            fn = self._get_fn(path)
            self.metrics[f"{fn}_metrics"] = metrics[i]
            
        # Create pandas df of the result metrics
        score_df = pd.DataFrame(self.metrics).transpose()
        score_df.loc['averages for the test set'] = score_df.mean(axis=0)
        return score_df
        
    

            
            
            
        
        
            