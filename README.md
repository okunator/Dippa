# Dippa
[Master's thesis](https://aaltodoc.aalto.fi/handle/123456789/108225) code repository (Aalto University)

Benchmarking framework for nuclei segmentation models.
Using encoders from: [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
Borrowing functions and utilities from [HoVer-Net repository](https://github.com/vqdang/hover_net)

## Contains
 - Easy model building and training with single .yml file.
 - Image patching
 - Inference
 - Benchmarking  

## Download scripts for datasets
* [x] [Kumar](https://ieeexplore.ieee.org/document/7872382) (Kumar et al.)
* [x] [CoNSep](https://arxiv.org/pdf/1812.06499) (Graham, Vu, et al.)
* [x] [Pannuke](https://arxiv.org/abs/2003.10778) (Gamper et al. Note: [License](https://creativecommons.org/licenses/by-nc-sa/4.0/))
* [ ] [MoNuSac](https://monusac-2020.grand-challenge.org/) (coming)

## Set Up
1. Clone the repository
2. cd to the repository `cd <path>/Dippa/`
3. Create environment (optional but recommended) 
```
conda create --name DippaEnv python=3.6
conda activate DippaEnv
```
or 

```
python3 -m venv DippaEnv
source DippaEnv/bin/activate
pip install -U pip
```

4. Install dependencies 
```
pip install -r requirements.txt
```

## Training Example
 
 1. modify the **experiment.yml** file
 2. Train the model. (See the notebooks)

### Training Script 

```python
import src.dl.lightning as lightning
from src.config import CONFIG

config = CONFIG
lightning_model = lightning.SegModel.from_conf(config)
trainer = lightning.SegTrainer.from_conf(config)

# use pannuke dataset
pannuke = PannukeDataModule(
    database_type="hdf5",
    augmentations=["hue_sat", "non_rigid", "blur"],
    normalize=False,
)

# Train the model. the dataset will be downloaded at first run
trainer.fit(model=lightning_model, datamodule=pannuke)
```


### experiment.yml

```yaml
experiment_args:
  experiment_name: testi2
  experiment_version: radam

dataset_args:
  n_classes: 6

model_args:
  architecture_design:
    module_args:
      activation: leaky-relu    # One of (relu, mish, swish, leaky-relu)
      normalization: bn         # One of (bn, bcn, gn, nope)
      weight_standardize: False # Weight standardization
      weight_init: he           # One of (he, TODO: eoc)
    encoder_args:
      in_channels: 3            # RGB input images
      encoder: efficientnet-b5  # https://github.com/qubvel/segmentation_models.pytorch
      pretrain: True            # Use imagenet pre-trained encoder
      depth: 5                  # Number of layers in encoder
    decoder_args:
      n_layers: 1               # Num multi conv blocks in one decoder level 
      n_blocks: 2               # Num of conv blocks inside a multi conv block
      preactivate: False        # If True, BN & RELU applied before CONV
      short_skips: null         # One of (residual, dense, null) (decoder)
      long_skips: unet          # One of (unet, unet++, unet3+, null)
      merge_policy: concatenate # One of (summation, concatenate) (long skips)
      upsampling: fixed_unpool  # One of (fixed_unpool) TODO: interp, transconv
      decoder_channels:         # Num of out channels for the decoder layers
        - 256
        - 128
        - 64
        - 32
        - 16

  decoder_branches:
    type_branch: True
    aux_branch: hover            # One of (hover, dist, contour, null)

training_args:
  normalize_input: False         # minmax normalize input images after augs
  freeze_encoder: False          # freeze the weights in the encoder
  weight_balancing: null         # TODO: One of (gradnorm, uncertainty, null)

  optimizer_args:
    optimizer: radam             # https://github.com/jettify/pytorch-optimizer 
    lr: 0.0005
    encoder_lr: 0.00005
    weight_decay: 0.0003
    encoder_weight_decay: 0.00003
    lookahead: True
    bias_weight_decay: True
    scheduler_factor: 0.25
    scheduler_patience: 3

  loss_args:
    inst_branch_loss: dice_ce
    type_branch_loss: dice_ce
    aux_branch_loss: mse_ssim
    edge_weight: null            # penalize nuclei edges in ce-based losses

runtime_args:
  resume_training: False
  num_epochs: 2
  num_gpus: 1
  batch_size: 4
  num_workers: 8                 # number workers for data loader
  model_input_size: 256          # size of the model input
  db_type: hdf5                  # One of (hdf5, zarr). 
```

## Inference Example

1. Run inference script. (See notebooks)

```python
from src.dl.inference.inferer import Inferer
import src.dl.lightning as lightning
from src.config import CONFIG

in_dir = "my_input_dir" # input directory for the image files
exp_name = "my_experiment" # name of the experiment (directory)
exp_version = "dense_skip_test" # name of the experiment version (sub directory inside the experiment dir)
lightning_model = lightning.SegModel.from_experiment(name=exp_name, version=exp_version)

inferer = Inferer(
    lightning_model,
    in_data_dir=in_dir,
    patch_size=(256, 256),
    stride_size=80,
    fn_pattern="*",
    model_weights="last",
    apply_weights=True,
    post_proc_method="cellpose",
    loader_batch_size=1,
    loader_num_workers=1,
    model_batch_size=16,
    n_images=32,
)

inferer.run_inference(
    save_dir=".../geojson",
    fformat="geojson",
    offsets=True
)
```

## References

- [1] N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, "A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology," in IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550-1560, July 2017 
- [2] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019.
- [3] Q D Vu, S Graham, T Kurc, M N N To, M Shaban, T Qaiser, N A Koohbanani, S A Khurram, J Kalpathy-Cramer, T Zhao, R Gupta, J T Kwak, N Rajpoot, J Saltz, K Farahani. Methods for Segmentation and Classification of Digital Microscopy Tissue Images. Frontiers in Bioengineering and Biotechnology 7, 53 (2019).  
- [4] Gamper, Jevgenij, Navid Alemi, Koohbanani, Simon, Graham, Mostafa, Jahanifar, Syed Ali, Khurram, Ayesha, Azam, Katherine, Hewitt, and Nasir, Rajpoot. "PanNuke Dataset Extension, Insights and Baselines".arXiv preprint arXiv:2003.10778 (2020).
- [5] Gamper, Jevgenij, Navid Alemi, Koohbanani, Ksenija, Benet, Ali, Khuram, and Nasir, Rajpoot. "PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification." . In European Congress on Digital Pathology (pp. 11–19).2019.
- [6] Caicedo, J.C., Goodman, A., Karhohs, K.W. et al. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nat Methods 16, 1247–1253 (2019)
