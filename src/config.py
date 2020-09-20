from omegaconf import OmegaConf, DictConfig
from pathlib import Path

# modify this
user_conf = OmegaConf.create(
    {   
        
        # Define the model name and the experiment
        # These will be used to write the result files to the right folders
        "experiment_args":{
            "model_name":"FPN",
            "experiment_version":"test_pannuke",
            "model_input_size":256, # network input size (multiple of 32)
            "batch_size":6,
        },
        
        # General dataset constants and args
        "dataset_args": {
            # What dataset you want to use? Has to be one of ("kumar", "consep", "pannuke")
            "dataset":"pannuke", 
            
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
            # = 'pannuke' and phases = ["train", "test"] then two folds of your choosing
            # are combined to training set remaining fold is the test set
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
            # For each experiment the model weights at the final epoch and best model against 
            # validation data will be saved. Choose one of ('best', 'last')
            "model_weights":"best",
            "verbose":True,
        },
    }
)

def get_conf(conf:DictConfig, dataset:str) -> DictConfig:
    """
    Generates a config file in the correct format from the user_conf and .yml confs
    for different datasets.
    
    Args: 
        conf (DictConfig): the above conf variable
        dataset (str): dataset that is being used. This key specifies what dataset args
                       the resulting config contains.
    Return:
        DictConfig
    """
    conf.dataset_args.dataset = dataset
    yml_path1 = [f for f in Path("../conf").iterdir() if conf.dataset_args.dataset in f.name][0]
    yml_path2 = [f for f in Path("../conf").iterdir() if f.name == "general.yml"][0]
    data_conf = OmegaConf.load(yml_path1)
    general_conf = OmegaConf.load(yml_path2)

    # pick the right args for the final config
    training_args = conf.training_args
    patching_args = conf.patching_args
    inference_args = conf.inference_args

    config = OmegaConf.create()
    config.training_args = training_args
    config.patching_args = patching_args
    config.inference_args = inference_args

    classes = data_conf.class_types[conf.dataset_args.class_types]
    data_args = data_conf
    data_args.classes = classes
    data_args.class_types = conf.dataset_args.class_types

    patch_dtype = conf.dataset_args.patches_dtype
    dataset_args = OmegaConf.merge(conf.dataset_args, general_conf[patch_dtype], data_args)
    experiment_args = OmegaConf.merge(conf.experiment_args, general_conf.experiment)

    # save the correct values to the final conf
    config.dataset_args = dataset_args
    config.experiment_args = experiment_args
    return config