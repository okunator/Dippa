# Dippa
(Master's thesis) Work in progress...
Benchmarking framework for nuclei segmentation models.
Using encoders from: [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch). 

## Download scripts for datasets
* [x] [Kumar](https://ieeexplore.ieee.org/document/7872382) (Kumar et al.)
* [x] [CoNSep](https://arxiv.org/pdf/1812.06499) (Graham, Vu, et al.)
* [x] [Pannuke](https://arxiv.org/abs/2003.10778) (Gamper et al.)
* [ ] [MoNuSac](https://monusac-2020.grand-challenge.org/) (Not yet published) (coming)
* [ ] [Dsb2018](https://www.kaggle.com/c/data-science-bowl-2018) (Caicedo et al.) (coming)

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

Intructions to run coming later...

## experiment.yml

```yaml
experiment_args:
  experiment_name: baseline_consep
  experiment_version: unet_test

dataset_args:
  train_dataset: consep         # One of (consep, pannuke, kumar, monusac)

model_args:
  architecture_design:
    module_args:
      activation: relu          # One of (relu, mish, swish)
      normalization: bn         # One of (bn, bcn, nope)
      weight_standardize: False # Weight standardization
      weight_init: he           # One of (he, eoc, fixup) (only for decoder if pretrain)
    encoder_args:
      in_channels: 3            # RGB input images
      encoder: resnet50         # https://github.com/qubvel/segmentation_models.pytorch
      pretrain: True            # Use imagenet pre-trained encoder
      encoder_depth: 5          # Number of layers in encoder
    decoder_args:
      n_blocks: 2               # Number of convolutions blocks in each decoder block
      short_skips: nope         # One of (residual, dense, nope) (for decoder branch only)
      long_skips: unet          # One of (unet, unet++, unet3+, nope)
      merge_policy: sum         # One of (sum, cat) (for long skips)
      upsampling: fixed_unpool  # One of (interp, max_unpool, transconv, fixed_unpool)
      decoder_channels:         # Number of out channels for every decoder layer
        - 256
        - 128
        - 64
        - 32
        - 16 

  decoder_branches:
    type_branch: True
    aux_branch: null        # One of (hover, dist, contour, null)

training_args:
  normalize_input: True          # minmax normalize input images after augs
  freeze_encoder: False          # freeze the weights in the encoder (for fine tuning)
  weight_balancing: null         # One of (gradnorm, uncertainty, null)
  augmentations:
    - hue_sat
    - non_rigid
    - blur

  optimizer_args:
    optimizer: adam              # One of https://github.com/jettify/pytorch-optimizer 
    lr: 0.0005
    encoder_lr: 0.00005
    weight_decay: 0.0003
    encoder_weight_decay: 0.00003
    lookahead: False
    bias_weight_decay: True
    scheduler_factor: 0.25
    scheduler_patience: 3

  loss_args:
    inst_branch_loss: dice_ce
    type_branch_loss: dice_ce
    aux_branch_loss: mse_ssim
    edge_weight: 1.1         # (float) Give penalty to nuclei borders in cross-entropy based losses
    class_weights: False       # Weight classes by the # of class pixels in the data

runtime_args:
  resume_training: False
  num_epochs: 5
  num_gpus: 1
  batch_size: 8
  num_workers: 8              # number workers for data loader
  model_input_size: 256       # size of the model input (input_size, input_size)
  db_type: zarr             # The type of the input data db. One of (hdf5, zarr). 
```

Borrowing functions and utilities from:

- HoVer-Net [repository](https://github.com/vqdang/hover_net)


## References

- [1] N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, "A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology," in IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550-1560, July 2017 
- [2] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019.
- [3] Q D Vu, S Graham, T Kurc, M N N To, M Shaban, T Qaiser, N A Koohbanani, S A Khurram, J Kalpathy-Cramer, T Zhao, R Gupta, J T Kwak, N Rajpoot, J Saltz, K Farahani. Methods for Segmentation and Classification of Digital Microscopy Tissue Images. Frontiers in Bioengineering and Biotechnology 7, 53 (2019).  
- [4] Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A., Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset Extension, Insights and Baselines arXiv preprint arXiv:2003.10778.
- [5] Caicedo, J.C., Goodman, A., Karhohs, K.W. et al. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nat Methods 16, 1247–1253 (2019)
