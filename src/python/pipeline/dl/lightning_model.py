import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from catalyst.contrib.nn import DiceLoss, IoULoss
from catalyst.contrib.nn import Ralamb, RAdam, Lookahead
from torch import optim
from torch.utils.data import DataLoader
from catalyst.dl import utils

from .augmentations import *
from .datasets import *



class SegModel(pl.LightningModule):
    def __init__(self, model, criterion, hparams, *args, **kwargs):
        """
        Pytorch Lightning abstraction for any pytorch segmentation model architecture used in this project.
        
        Args:
            model (nn.Module) : Pytorch model specification. Can be from smp, toolbelt, or a custom model. 
                                ttatch wrappers work also. Basically any model that inherits nn.Module shld work
            criterion (nn.Module) : Pytorch loss function. 
            haparams (dict) : hyperparameters for every component of the training process.
        """
        super(SegModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.test_dir = hparams['dirs']['test_dir']
        self.train_dir = hparams['dirs']['train_dir']
        self.edge_weight = hparams['loss_args']['edge_weight']
        self.batch_size = hparams['dataloader_args']['batch_size']
        self.n_classes = hparams['dataloader_args']['n_classes']
        self.lr = hparams['optimizer_args']['lr']
        self.encoder_lr = hparams['optimizer_args']['encoder_lr']
        self.weight_decay = hparams['optimizer_args']['weight_decay']
        self.encoder_weight_decay = hparams['optimizer_args']['encoder_weight_decay']
        self.factor = hparams['scheduler_args']['factor']
        self.patience = hparams['scheduler_args']['patience']
        self.save_hyperparameters(hparams)

    # Lightning framework boilerplate:
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, train_batch, batch_idx) :
        x = train_batch['image']
        y = train_batch['mask']
        y_weight = train_batch['mask_weight']

        x = x.float()
        y_weight = y_weight.float()
        y = y.long()
        
        yhat = self.forward(x)
        loss_matrix = self.criterion(yhat, y)
        loss = (loss_matrix * (self.edge_weight**y_weight)).mean()
        
        logs = {'train_loss': loss}
        return {'loss':loss, 'log': logs}
    
    
    def validation_step(self, val_batch, batch_idx):
        x = val_batch['image']
        y = val_batch['mask']
        y_weight = val_batch['mask_weight']

        x = x.float()
        y_weight = y_weight.float()
        y = y.long()
        
        # Compute loss
        yhat = self.forward(x)
        loss_matrix = self.criterion(yhat, y)        
        loss = (loss_matrix * (self.edge_weight**y_weight)).mean()
        
        # Compute confusion matrix for accuracy
        pred = yhat.detach().cpu().numpy()
        predflat = np.argmax(pred, axis=1).flatten()
        yflat = y.cpu().numpy().flatten()
        cmatrix = confusion_matrix(yflat, predflat, labels=range(self.n_classes))
        TN = cmatrix[0, 0]
        TP = cmatrix[1, 1]
        FP = cmatrix[0, 1]
        FN = cmatrix[1, 0]
        TNR = TN / (TN + FP + 1e-08)
        TPR = TP / (TP + FN + 1e-08)
        accuracy = (TP + TN)/(TN + TP + FP + FN+ 1e-08)
        
        TNR = torch.from_numpy(np.asarray(TNR))
        TPR = torch.from_numpy(np.asarray(TPR))
        accuracy = torch.from_numpy(np.asarray(accuracy))
        
        logs = {'val_loss': loss, 'TN': TN, 'TP':TP, 'FP':FP, 'FN':FN, 'TNR':TNR, 'TPR':TPR, 'acc':accuracy}
        return {'val_loss': loss, 'accuracy':accuracy, 'log':logs}
    
    
    def validation_epoch_end(self, outputs):
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        TNR = torch.stack([x['TNR'] for x in outputs]).mean()
        TPR = torch.stack([x['TPR'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss, 'accuracy':accuracy, 'TNR':TNR, 'TPR':TPR}
        return {'accuracy':accuracy, 'val_loss': avg_loss, 'log': tensorboard_logs}

    
    def configure_optimizers(self):
        layerwise_params = {"encoder*": dict(lr=self.encoder_lr, weight_decay=self.encoder_weight_decay)}
        
        # Remove weight_decay for biases and apply layerwise_params for encoder
        model_params = utils.process_model_params(self.model, layerwise_params=layerwise_params)
        
        # Base Optimizer
        base_optimizer = Ralamb(
            model_params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Lookahead optimizer
        optimizer = Lookahead(base_optimizer)
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience
        )
        return [optimizer], [scheduler]
    
    
    def prepare_data(self):
        # compose transforms
        transforms = compose([hard_transforms(patch_size=224)])
        
        self.trainset = BinarySegmentationDataset(
            fname = self.train_dir, 
            transforms = transforms,
        )
        
        self.testset = BinarySegmentationDataset(
            fname = self.test_dir, 
            transforms = transforms,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8
        )

   