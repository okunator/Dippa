# Dippa
Benchmarking of deep learning and other segmentation methods for H&amp;E images  
Most of the models are just wrappers for models implemented in: [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch).

Also borrowing a lot of utilities from:

- HoVer-Net [repository](https://github.com/vqdang/hover_net)
- pytorch-toolbelt [repository](https://github.com/BloodAxe/pytorch-toolbelt) 

## Supported datasets
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


##  Instructions for running the experiments
1. Download the data `src/download.py`.
2. Run an experiment.
    1. Modify the parameters in the config file. `src/conf/config.py`
    2. Extract patches from the downloaded images and save them to hdf5 or numpy files. `src/patch.py`
    3. Train a model with the extracted patches.  `src/train.py` (use notebook for now)
    4. Infer, post process and benchmark.  `src/infer.py` (use notebook for now)
    - Optionally you can just run the notebooks in `notebooks/` which do the exact same.
    - **Note:** you don't have to repeat part **i** and **ii** if you've already done them and you want to run new experiments. If you want to patch the images differently (different stride or window size) for your experiments then modify the config and run part **ii** again.

## Repository structure
- `Dippa`/
    - `datasets/` Location for the datasets after running `src/download.py` 
    - `notebooks/` Notebooks for running the codes instead of running the pyton scripts 
    - `patches/` Location for the patched datasets after running `src/write_patches.py`
    - `results/` Location for the results from training and inference
    - `src/` 
        - `conf/`
            - `config.py` THE CONFIG FILE TO MODIFY FOR DIFFERENT EXPERIMENTS
            - `consep.yml` data related to consep dataset
            - `dsb2018.yml` data related to dsb2018 dataset TODO
            - `kumar.yml` data related to the kumar dataset
            - `pannuke.yml` data related to the pannuke dataset
        - `dl/`
            - `filters/` couple files containing utilities for convolving kernels on matrices
            - `losses/` contains all different losses and utilities for them
            - `models/` contains all the different models and utilities for them
            - `datasets.py` pytorch DataSet class for the experiments
            - `inferer.py` class for inference, post processing and benchmarking of the trained models
            - `lightning_model.py` pytorch lightning class abstraction for any pytorch model used in this project
            - `torch_utils.py` utility functions for torch tensors
        - `img_processing/`
            - `augmentations.py` data augmentations
            - `post_processing.py` functions used in post processing inst maps 
            - `pre_processing.py` functions used for pre processing patches before training
            - `process_utils.py` utility functions for img and mask processing
            - `viz_utils.py` utility functions for plotting results
        - `metrics/`
            - `metrics.py` benchmarking metrics functions 
        - `tune/`
            - `tune_experiment.py` Hyper parameter tuning (TODO)
        - `utils/`
            - `adhoc_conversions.py` functions to move the and modify the downloaded data around the repo
            - `benchmarker.py` class for benchmarking segmentation results
            - `data_downloader` class for downloading and converting data to right format
            - `data_writer.py` class for writing image and mask patches for training
            - `file_manager.py` class for handling the downloaded data and managing all files and paths
            - `patch_extractor.py` class for extracting patches from images
        - `download.py` Download the datasets and convert the downloaded data to right format
        - `infer.py` Run inference, post processing and benchmarking
        - `patch.py` Run patching for downloaded images
        - `settings.py` A convention for building paths
        - `train.py` Run training for pytorch models

## References

- [1] N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, "A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology," in IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550-1560, July 2017 
- [2] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019.
- [3] Q D Vu, S Graham, T Kurc, M N N To, M Shaban, T Qaiser, N A Koohbanani, S A Khurram, J Kalpathy-Cramer, T Zhao, R Gupta, J T Kwak, N Rajpoot, J Saltz, K Farahani. Methods for Segmentation and Classification of Digital Microscopy Tissue Images. Frontiers in Bioengineering and Biotechnology 7, 53 (2019).  
- [4] Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A., Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset Extension, Insights and Baselines arXiv preprint arXiv:2003.10778.
- [5] Caicedo, J.C., Goodman, A., Karhohs, K.W. et al. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nat Methods 16, 1247–1253 (2019)
