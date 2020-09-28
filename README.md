# Dippa
Benchmarking of deep learning and other segmentation methods for H&amp;E images  
Borrowing a lot of methods from HoVer-Net [repo](https://github.com/vqdang/hover_net): and methodology from HoVer-Net paper [2]  

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
1. Download the data.
2. Run an experiment.
    1. Convert the data that you downloaded. `src/convert.py`
    2. Modify the parameters in the config file. `src/conf/config.py`
    3. Extract patches from the downloaded images and save them to hdf5 or numpy files. `src/patch.py`
    4. Train a model with the extracted patches.  `src/train.py`
    5. Infer, post process and benchmark.  `src/infer.py`
    - Optionally you can just run the notebooks in `notebooks/` which do the exact same.
    - **Note:** you don't have to repeat part **i** and **ii** if you've already done them and you want to run new experiments. If you want to patch the images differently (different stride or window size) for your experiments then modify the config and run part **ii** again.

## Instructions for downloading the datasets
1. Download the datsets from the links below
2. Move the downloaded zips or extract the zips to the corresponding folders in `datasets/`
3. Move to **Part 2** in the instructions above

#### Data download links:
1. **kumar**  
          - Train: https://drive.google.com/file/d/1JZN9Jq9km0rZNiYNEukE_8f0CsSK3Pe4/view .   
          - Test: https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view  
          
2. **consep** - https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/
3. **cpm** - TODO
4. **pannuke** - https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke
5. **kaggle dsb 2018**: https://bbbc.broadinstitute.org/BBBC038

## Repository structure
- `Dippa`/
    - `datasets/` Location for the raw and processed datasets after downloading and running `src/convert.py`
    - `notebooks/` Notebooks for running the codes instead of running the pyton scripts 
    - `patches/` Location for the patched datasets after running `src/write_patches.py`
    - `results/` Location for the results from training and inference
    - `src/` 
        - `conf/`
            - `config.py` THE CONFIG FILE TO MODIFY FOR DIFFERENT EXPERIMENTS
            - `consep.yml` data related to consep dataset
            - `cpm.yml` data related to cpm dataset TODO
            - `dsb2018.yml` data related to dsb2018 dataset TODO
            - `kumar.yml` data related to the kumar dataset
            - `pannuke.yml` data related to the pannuke dataset
        - `dl/`
            - `datasets.py` pytorch DataSet class for the experiments
            - `inferer.py` class for inference, post processing and benchmarking of the trained models
            - `lightning_model.py` pytorch lightning class abstraction for any pytorch model used in this project
        - `img_processing/`
            - `augmentations.py` data augmentations
            - `post_processing.py` functions used in post processing inst maps 
            - `pre_processing.py` functions used for pre processing patches before training
            - `process_utils.py` utility functions for img and mask processing
            - `viz_utils.py` utility functions for plotting results
        - `metrics/`
            - `metrics.py` benchmarking metrics functions 
        - `tune/`
            - `tune_experiment.py` TODO
        - `utils/`
            - `data_writer.py` class for writing image and mask patches for training
            - `file_manager.py` class for handling the downloaded data and managing all files and paths
            - `patch_extractor.py` class for extracting patches from images
        - `convert.py` Convert the downloaded data to right format
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
