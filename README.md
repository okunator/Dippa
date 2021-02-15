# Dippa
Work in progress...
Benchmarking framework for nuclei segmentation and cell type classification.
Datasets contain only H&amp;E stained images for now
Using encoders from: [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch).
Also nulti-task segmentation model wrappers for smp models. 

Also borrowing a lot of utilities from:

- HoVer-Net [repository](https://github.com/vqdang/hover_net)
- pytorch-toolbelt [repository](https://github.com/BloodAxe/pytorch-toolbelt) 

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

**Note**: dl framework is PyTorch (torch==1.4.0) which expects cuda 10.1. If you want to use the repo with other version of torch and cuda, just follow the installation details at https://pytorch.org/. At least 1.6.0 with cuda 10.1 worked ok. 


Intsuctions to run coming later...

## experiment.yml

```yaml
experiment_args:
  experiment_name: test
  experiment_version: something

dataset_args:
  train_dataset: consep         # One of (consep, pannuke, kumar, monusac)
  infer_dataset: consep         # One of (consep, pannuke, kumar, monusac, other)
  other_path: null              # Path to own dataset if infer_dataset=other

model_args:
  architecture_design:
    module_args:
      activation: relu          # One of (relu, mish, swish)
      normalization: bn         # One of (bn, bcn, nope)
      weight_standardize: False # Weight standardization
      weight_init: he           # One of (he, eoc, fixup) (only for decoder if pretrain)
    encoder_args:
      in_channels: 3            # RGB input images
      encoder: resnext50        # https://github.com/qubvel/segmentation_models.pytorch
      pretrain: True            # Use imagenet pre-trained encoder
      encoder_depth: 5          # Number of layers in encoder
    decoder_args:
      n_blocks: 2               # Number of convolutions blocks in each decoder block
      short_skips: residual     # One of (residual, dense, nope) (for decoder branch only)
      long_skips: unet          # One of (unet, unet++, unet3+, nope)
      merge_policy: cat         # One of (sum, cat) (for long skips)
      upsampling: fixed_unpool  # One of (interp, max_unpool, transconv, fixed_unpool)

  decoder_branches:
    type: True
    aux: True
    aux_type: unet              # One of (hover, dist, contour, unet) (ignored if aux=False)

training_args:
  resume_training: False
  num_epochs: 3
  num_gpus: 1
  weight_balancing: null        # One of (gradnorm, uncertainty, null)
  augmentations:
    - hue_sat
    - non_rigid
    - blur
    - non-spatial

  optimizer_args:
    optimizer: adamw            # One of https://github.com/jettify/pytorch-optimizer 
    lr: 0.0005
    encoder_lr: 0.00005
    weight_decay: 0.0003
    encoder_weight_decay: 0.00003
    lookahead: True
    bias_weight_decay: False
    scheduler_factor: 0.25
    schduler_patience: 2

  loss_args:
    inst_branch_loss: dice_focal
    type_branch_loss: tversky_focal
    aux_branch_loss: mse
    edge_weight: False
    class_weights: False


inference_args:
  model_weights: last         # One of (last, best)
  data_fold: test             # One of (train, test)
  tta: False
  verbose: True

runtime_args:
  batch_size: 6
  model_input_size: 256       # Multiple of 32. Tuple(int, int) 

```

## References

- [1] N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, "A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology," in IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550-1560, July 2017 
- [2] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019.
- [3] Q D Vu, S Graham, T Kurc, M N N To, M Shaban, T Qaiser, N A Koohbanani, S A Khurram, J Kalpathy-Cramer, T Zhao, R Gupta, J T Kwak, N Rajpoot, J Saltz, K Farahani. Methods for Segmentation and Classification of Digital Microscopy Tissue Images. Frontiers in Bioengineering and Biotechnology 7, 53 (2019).  
- [4] Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A., Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset Extension, Insights and Baselines arXiv preprint arXiv:2003.10778.
- [5] Caicedo, J.C., Goodman, A., Karhohs, K.W. et al. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nat Methods 16, 1247–1253 (2019)
