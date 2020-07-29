import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import confusion_matrix
from catalyst.contrib.nn import DiceLoss, IoULoss
from catalyst.contrib.nn import Ralamb, RAdam, Lookahead
from torch import optim
from torch.utils.data import DataLoader
from catalyst.dl import utils
from catalyst.contrib.tools.tensorboard import SummaryItem, SummaryReader

from .augmentations import *
from .datasets import *


class SegModel(pl.LightningModule):
    def __init__(self, model, criterion, hparams, dataset='kumar'):
        """
        Pytorch Lightning abstraction for any pytorch segmentation model architecture used in this project.
        
        Args:
            model (nn.Module) : Pytorch model specification. Can be from smp, toolbelt, or a custom model. 
                                ttatch wrappers work also. Basically any model that inherits nn.Module shld work
            criterion (nn.Module) : Pytorch loss function. 
            haparams (dict) : hyperparameters for every component of the training process.
        """
        super(SegModel, self).__init__()
        
        assert dataset in ('kumar', 'consep', 'pannuke'), "dataset param not in ('kumar', 'consep', 'pannuke')"
        
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
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
        
        # HDF5 database directories
        self.data_dirs = {
            'kumar':{
                'test_dir': "../../../../databases/Kumar/patch_224/test_Kumar.pytable",
                'train_dir': "../../../../databases//Kumar/patch_224/train_Kumar.pytable"
            },
            'consep':{
                'test_dir': "../../../../databases/ConSeP/patch_224/test_Kumar.pytable",
                'train_dir': "../../../../databases/ConSeP/patch_224/train_Kumar.pytable"
            },
            'pannuke': {
                'test_dir': "../../../../databases/PanNuke/patch_224/test_Kumar.pytable",
                'train_dir': "../../../../databases/PanNuke/patch_224/train_Kumar.pytable"
            }
        }
        
        self.train_dir = self.data_dirs[self.dataset]["train_dir"]
        self.test_dir = self.data_dirs[self.dataset]["test_dir"]

    # Lightning framework stuff:
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
        self.logger.experiment.add_scalars("losses", {"train_loss": loss}, 'progress_bar': {'train_loss': loss})

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
        
        # plot to the same plot as train loss
        self.logger.experiment.add_scalars("losses", {"val_loss": loss})
        
        logs = {'val_loss': loss, 'TN': TN, 'TP':TP, 'FP':FP, 'FN':FN, 'TNR':TNR, 'TPR':TPR, 'acc':accuracy}
        return {'val_loss': loss, 'accuracy':accuracy, 'log':logs, 'progress_bar': {'val_loss': loss}}
    
    
    def validation_epoch_end(self, outputs):
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        TNR = torch.stack([x['TNR'] for x in outputs]).mean()
        TPR = torch.stack([x['TPR'] for x in outputs]).mean()
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_val_loss, 'accuracy':accuracy, 'TNR':TNR, 'TPR':TPR}
        return {'accuracy':accuracy, 'val_loss': avg_loss, 'log': tensorboard_logs}
    
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_train_loss': avg_loss}
        return {'avg_train_loss': avg_loss, 'log': tensorboard_logs}

    
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

    
    # NOT WORKING YET
    #def plot_loss(self, scale='log'):
    #    """
    #    Plot the training and validation loss to same graph
    #    Args:
    #        scale (str) : y-axis scale. One of ('log', 'normal').  
    #    """
    #    
    #    assert scale in ('log', 'normal')
    #    ldir = Path(logdir)
    #    logdirs = {str(x).split('/')[-1]:x 
    #               for x in ldir.glob("**/*") 
    #               if x.is_dir() and str(x).split('/')[-1].startswith("version")}
    #    
    #    train_losses_all = {}
    #    avg_valid_losses_all = {}
    #    avg_valid_accuracies_all = {}
    #    processed_batches_all = {}
    #    epochs_all = {}
    #
    #    for key in logdirs.keys():
    #        try:
    #            train_losses_all[key] = []
    #            avg_valid_losses_all[key] = []
    #            avg_valid_accuracies_all[key] = []
    #            processed_batches_all[key] = []
    #            epochs_all[key] = []
    #            summary_reader = SummaryReader(logdirs[key], types=['scalar'])
    #            
    #            for item in summary_reader:
    #                processed_batches_all[key].append(item.step)
    #                if item.tag == 'train_loss':
    #                    train_losses_all[key].append(item.value)
    #                elif item.tag == 'epoch':
    #                    epochs_all[key].append(item.value)
    #                elif item.tag == 'accuracy':
    #                    avg_valid_accuracies_all[key].append(item.value)
    #                elif item.tag == 'avg_val_loss':
    #                    avg_valid_losses_all[key].append(item.value)
    #
    #        except:
    #            pass
    #        
    #    np_train_losses = np.array(train_losses)
    #    np_valid_losses = np.array(valid_losses)
    #    np_epochs = np.array(epochs)
    #
    #    plt.rcParams["figure.figsize"] = (20,10)
    #
    #    df = pd.DataFrame({'training loss': train_losses, 'validation loss': valid_losses, 'epoch': epochs})
    #    ax = plt.gca()
    #    df.plot(kind='line',x='epoch',y='training loss',ax=ax)
    #    df.plot(kind='line',x='epoch',y='validation loss', color='red', ax=ax)
    #    plt.yscale(scale)
    #    plt.show()
        
   