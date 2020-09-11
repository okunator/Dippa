from omegaconf import OmegaConf

conf = OmegaConf.create(
    {   
        
        # Define the model name and the experiment
        # These will be used to write the result files to the right folders
        "experiment_args":{
            "model_name":"FPN",
            "experiment_version":"test_consep2",
        },
        
        # General dataset constants and args
        "dataset": {
            "args": {
                # What dataset you want to use? Has to be one of ("kumar", "consep", "pannuke")
                "dataset":"consep", 
                
                # This depends on the dataset. Binary segmentation can be done to all datasets
                # change this according to your needs. has to be one of ("binary", "consep", "pannuke")
                "class_types":"binary", 
                
                # if phases = ["train", "valid", "test"]. The train set is also split to 
                # validation set. If phases = ["train", "test"], no splitting is done.
                # If dataset = 'pannuke' and phases = ["train", "valid", "test"] the fold1
                # is used as training set, fold2 is used as validation set and fold3 is used
                # as test set. If dataset = 'pannuke' and phases = ["train", "test"] then
                # fold1 and fold2 are combined to training set and fold3 remains as test set
                "phases":["train", "test"], # ["train", "valid", "test"] or ["train", "test"]
            },
            "class_dicts": {
                "binary":{
                    "background":0,
                    "nuclei":1
                },
                "consep":{
                    "background":0,
                    "Miscellaneous":1, # ! Please ensure the matching ID is unique
                    "Inflammatory":2,
                    "Epithelial":3,
                    "Spindle":4,
                },
                "pannuke":{
                    "background":0,
                    # TODO: rest off the classeshan one
                }
            }
        },
        
        # Do not change anything except the "other" part in "data_dirs" for your own dataset
        "paths":{
            
            # Directories where the raw data is located after downloading from the internet
            # Recomendation is to place the files after downloading them to these directories
            # Otherwise a lot of stuff will prbly have to modified and things break..
            # TODO
            "raw_data_dirs":{
                "kumar":{
                    "train":"../../",
                    "test":"../../",
                },
                "consep":{
                    "train":"../../",
                    "test":"../../",
                },
                "pannuke":{
                    "train":"../../",
                    "valid":"../../",
                    "test":"../../",
                },
            },
            
            "data_dirs": {
                "kumar": {
                    "train_im":"../../datasets/nucleisegmentation_kumar/train/Images",
                    "train_gt":"../../datasets/nucleisegmentation_kumar/train/Labels",
                    "test_im":"../../datasets/nucleisegmentation_kumar/test/Images",
                    "test_gt":"../../datasets/nucleisegmentation_kumar/test/Labels",
                },
                "consep":{
                    "train_im":"../../datasets/nucleisegmentation_consep/train/Images",
                    "train_gt":"../../datasets/nucleisegmentation_consep/train/Labels",
                    "test_im":"../../datasets/nucleisegmentation_consep/test/Images",
                    "test_gt":"../../datasets/nucleisegmentation_consep/test/Labels",
                },
                "pannuke": {
                    "train_im":"../../datasets/nucleisegmentation_pannuke/train/Images",
                    "train_gt":"../../datasets/nucleisegmentation_pannuke/train/Labels",
                    "valid_im":"../../datasets/nucleisegmentation_pannuke/valid/Images",
                    "valid_gt":"../../datasets/nucleisegmentation_pannuke/valid/Labels",
                    "test_im":"../../datasets/nucleisegmentation_pannuke/test/Images",
                    "test_gt":"../../datasets/nucleisegmentation_pannuke/test/Labels",
                },
                "other": {
                    # TODO:
                    "train_im":"../../datasets/other/...",
                    "train_gt":"../../datasets/other/...",
                    "test_im":"../../datasets/other/...",
                    "test_gt":"../../datasets/other/...",
                }
            },
            
            # root dir for HDF5 databases storing patches
            "database_root_dir":"../../patches/hdf5",
            
            # root dir for patches stored in .npy file 
            "patches_root_dir":"../../patches/npy",
            
            # root dir for files that are created in training the network
            "experiment_root_dir":"../../results/tests/",
            
        },
                
        # Change these according to your needs. The more there are patches (trtaining data).
        # The better your model performs. Patch_size is recommended to be at least twice the
        # input size to the network like in hovernet. You'll get more training data by selecting
        # small enough stride size. Training will get slower though...
        "patching_args":{
            "patch_size":512, # Size of an image patch that gets written to hdf5 db
            "stride_size":96, # Size of window stride
            "input_size":256, # network input size (multiple of 32)
            "crop_to_input":True,
            "verbose":False
        },
    
        # Model training args
        "training_args": {
            "batch_size":6,
            "resume_training":False, # continue training where you left off?
            "num_epochs":30,
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
                "edge_weight" : 1.1,
            },
        },
        
        # Inference args
        "inference_args" : {
            "smoothen":False, # Inference time slightly slower. Gets rid of checkerboard. May lower PQ.
            "data_fold":"test", # what data fold (phase) to use in inference
            "test_time_augmentation":True, # Inference time slightly slower
            "threshold":0.5, # if smoothen is not used, then this is used for threshing soft masks
            # For each experiment the model weights at the final epoch and best model against validation 
            # data will be saved. Choose one of ('best', 'last')
            "model_weights":"best",
            "verbose":True,
        }
    }
)