from omegaconf import OmegaConf

# Modify this when you want to run new experiments
CONFIG = OmegaConf.create(
    {   
        
        # Define the model name and the experiment
        # These will be used to write the result files to the right folders
        "experiment_args":{
            "model_name":"Unet",
            "experiment_version":"test_pannuke_unet2",
        },
        
        # General dataset constants and args
        "dataset_args": {
            # What dataset you want to use? Has to be one of ("kumar", "consep", "pannuke")
            "dataset":"consep", 
            
            # This depends on the dataset. Binary segmentation can be done to all datasets
            # and semantic segmentation can be done to consep and pannuke datasets
            # change this according to your needs. has to be one of ("binary", "types")
            # Things won't crash even if types is used for a dataset that can be used only for 
            # binary segmentation
            "class_types":"binary", 
            
            # if phases = ["train", "valid", "test"]. The train set is also split to 
            # validation set. If phases = ["train", "test"], no splitting is done.
            # If dataset = 'pannuke' and phases = ["train", "valid", "test"] the folds
            # of your choosing are treated as train, valid and test folds. If dataset 
            # = 'pannuke' and phases = ["train", "test"] then two folds that are set for
            # train and valid are combined to one big training set remaining fold is the 
            # test set. Folds can be modified in pannyke.yml
            "phases":["train", "valid", "test"], # ["train", "valid", "test"] or ["train", "test"]
            
            # Use either patches written to hdf5 db or .npy files. One of ("npy", "hdf5")
            "patches_dtype":"hdf5"
        },
              
        # Change these according to your needs. The more there are patches (trtaining data).
        # The better your model performs. Patch_size is recommended to be at least twice the
        # input size to the network like in hovernet. You'll get more training data by selecting
        # small enough stride size. Training will get slower though...
        "patching_args":{
            "patch_size":512, # Size of an image patch that gets written to hdf5 db
            "stride_size":80, # Size of window stride
            "crop_to_input":True,
            "verbose":False
        },
    
        # Model training args
        "training_args": {
            "tta":False, # use test time augmentation during training. Note: very slow w ttatch
            "resume_training":False, # continue training where you left off?
            "num_epochs":7,
            "num_gpus":1,
            "optimizer_args":{
                "lr":0.001,
                "encoder_lr":0.0005,
                "weight_decay":0.0003,
                "encoder_weight_decay":0.00003,
            },

            "scheduler_args": {
                "factor":0.25,
                "patience":2,
            },

            "loss_args" : {
                "edge_weight" : 1.1, # How much weight is applied to nuclei borders  
            },
        },
        
        # Inference args
        "inference_args" : {
            "smoothen":False, # Inference time slightly slower. Gets rid of checkerboard. May lower PQ.
            "data_fold":"test", # what data fold (phase) to use in inference
            "tta":False, # Inference time slightly slower
            "threshold":0.5, # if smoothen is not used, then this is used for threshing soft masks
            # For each experiment the model weights at the final epoch and best model against 
            # validation data will be saved. Choose one of ('best', 'last')
            "model_weights":"best",
            "verbose":True,
        },
    }
)