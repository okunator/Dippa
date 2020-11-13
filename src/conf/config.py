from omegaconf import OmegaConf

# Modify this when you want to run new experiments
CONFIG = OmegaConf.create(
    {   
        
        # Define the model name and the experiment
        # These will be used to write the result files to the right folders
        "experiment_args":{
            "model_name":"UNET",
            "experiment_version": "panoptic_hover_testtt",
        },
        
        # General dataset constants and args
        "dataset_args": {

            # What dataset you want to use? Has to be one of ("kumar", "consep", "pannuke")
            "dataset":"consep", 
            
            # This depends on the dataset. instance segmentation can be done to all datasets
            # and panoptic segmentation can be done to consep and pannuke datasets
            # change this according to your needs. has to be one of ("instance", "panopotic")
            # Things won't crash even if types is used for a dataset that can be used only for 
            # instance segmentation
            "class_types":"panoptic",
            
            # if phases = ["train", "valid", "test"]. The train set is also split to 
            # validation set. If phases = ["train", "test"], no splitting is done.
            # If dataset = 'pannuke' and phases = ["train", "valid", "test"] the folds
            # of your choosing are treated as train, valid and test folds. If dataset 
            # = 'pannuke' and phases = ["train", "test"] then two folds that are set for
            # train and valid are combined to one big training set remaining fold is the 
            # test set. Folds can be modified in pannyke.yml
            "phases":["train", "test"], # ["train", "valid", "test"] or ["train", "test"]
            
            # Use either patches written to hdf5 db or .npy files. One of ("npy", "hdf5")
            "patches_dtype":"hdf5",

            # Use auxilliary regression branch in the model. One of (None, "hover", TODO:"micro")
            # Requires special GT masks
            "aux_branch": "hover",
        },
              
        # Change these according to your needs. The more there are patches (trtaining data).
        # The better your model performs. Patch_size is recommended to be at least twice the
        # input size to the network like in hovernet. You'll get more training data by selecting
        # small enough stride size. Training will get slower though...
        "patching_args":{
            "batch_size":6,
            "model_input_size":256,
            "patch_size":512, # Size of an image patch that gets written to hdf5 db
            "stride_size":80, # Size of window stride
            "crop_to_input":True,
            "verbose":False
        },
    
        # Model training args
        "training_args": {
            "batch_size":8,

            # This needs to be same as in the patching args
            "model_input_size":256,

            "tta":False, # use test time augmentation during training. Note: very slow w ttatch
            "resume_training":True, # continue training where you left off?
            "num_epochs":40,
            "num_gpus":1,
            
            # optimizer args
            "lr":0.001,
            "encoder_lr":0.0005,
            "weight_decay":0.0003,
            "encoder_weight_decay":0.00003,
            
            #scheduler args
            "factor":0.25,
            "patience":2,
            
            # loss args

            # One of ("wCE", "wSCE", "wFocal", "IoU_wCE", "IoU_wSCE", "DICE_wCE", "DICE_wSCE", "DICE_wFocal")
            # More about these in losses.py. This loss will be used for instance segmentation branch
            "inst_branch_loss":"DICE_wFocal",

            # One of ("wCE", "wSCE", "wFocal", "IoU_wCE", "IoU_wSCE", "DICE_wCE", "DICE_wSCE", "DICE_wFocal")
            # This loss will be used for type segmentation branch. This is optional
            "semantic_branch_loss":"DICE_wFocal",

            # Whether to apply weights at nuclei borders when computing the loss
            "edge_weights":False,

            # How much weight is applied to nuclei borders. 1.0 = no weight.
            # This is ignored if "edge_weights" is False
            "edge_weight": 1.0,

            # Apply weights to different classes. Weights are computed by from the number of pixels
            # belonging to each class and the less number of pixels there is in a class the bigger
            # weight it will get. All weights are b/w [0, 1] 
            "class_weights":False
        },
        
        # Inference args
        "inference_args" : {
            
            "batch_size": 8,

            # This needs to be same as in the patching args
            "model_input_size": 256,

            # For each experiment the model weights at the final epoch and best model against
            # validation data will be saved. Choose one of ('best', 'last')
            "model_weights": "last",

            # what data fold (phase) to use in inference
            "data_fold":"test",

            # Inference time increases. Usually increases different metrics a little (not always)
            "tta":False,

            # Inference time increases. Gets rid of checkerboard. May lower PQ.
            "smoothen": False,

            # One of ("argmax", "sauvola_thresh", "niblack_thresh", None)
            # If this is None then naive thresholding is used and thresh level
            #  is the argument below this.
            "thresh_method":None,

            # if 'smoothen' is False or 'thresh_method' is None then naive thresholding
            # with this threshold is used to theshold the soft masks 
            "threshold": 0.5,

            # apply watershed based post_processing. If False then only thresholding or argmax is used
            "post_processing":True,

            # Choose the post processing method
            # One of ("shape_index_watershed", "shape_index_watershed2", "sobel_watershed", "inv_dist_watershed",
            # "post_proc_hover", "post_proc_hover2").
            # This is ignored if post_processing=False. Then only thresholding is applied to the softmasks.
            # This is also ignored if aux_branch is "hover" or "micro" which use dedicated post proc pipelines 
            "post_proc_method": "post_proc_hover",

            # Print inference progress
            "verbose":True,
        },
    }
)
