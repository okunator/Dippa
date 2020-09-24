# Dippa
Benchmarking of deep learning and other segmentation methods for H&amp;E images  
Borrowing a lot of methods from HoVer-Net [repo](https://github.com/vqdang/hover_net): and methodology from HoVer-Net paper [2]  
**Note**: dl framework is PyTorch 

## Set Up
1. Clone the repository
2. cd to the repository `cd <path>/Dippa/`
3. Create environment (optional but recommended) 
```
conda create --name Dippa python=3.6
conda activate Dippa
```
Or 

```
venv stuff
```

4. Install dependencies 
```
pip3 -r requirements.txt
```

##  Instructions for running the experiments
1. Download the data
2. Run notebooks or the runner scripts
    1. Convert the data that you downloaded. `src/convert_raw_data.py` script or `1_Convert_downloaded_data.ipynb
    2. Modify the parameters in the config file. `src/conf/config.py`
    3. Extract or just save tiles/patches from the images you just downloaded and save them to hdf5 format or numpy files. `2_Create_databases.ipynb` or `src/write_patches.py`
    4. Train a model. `3_Train_model.ipynb` or `src/train.py`
    5. Infer and benchmark. `4_Infer_and_benchmark.ipynb` or `infer.py`

## Instructions for downloading the datasets
1. Download the datsets from the links below
2. Move the downloaded zip files or extract the contents of the zip files to the corresponding folders in `datasets/` folder
3. Move to `Part 2` in the instructions above

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
    - `datasets/` Where the downloaded datasets should be stored and where the processed datasets will be stored after running `src/convert_raw_data.py` script or `1_Convert_downloaded_data.ipynb

    - `notebooks/` Where the notebooks for running the codes are located 
    - `patches/` Where the patched datasets are located after running `2_Create_databases.ipynb` or `src/write_patches.py`
    - `src/` 
        - `conf/`
            - `config_schema.py` dataclasses defining the config file schema in this project
            - 'config.py' THE CONFIG FILE TO MODIFY FOR DIFFERENT EXPERIMENTS
            - `consep.yml` data related to consep dataset
            - `cpm.yml` data related to cpm dataset TODO
            - `dsb2018.yml` data related to dsb2018 dataset TODO
            - `kumar.yml` data related to the kumar dataset
            - `pannuke.yml` data related to the pannuke dataset
        - `dl/`
            - `datasets.py` pytorch DataSet class for the experiments
            - `inferer.py` class for the inference, post processing and benchmarking of the trained models on different datasets
            - `lightning_model.py` pytorch lightning class abstraction for any pytorch model used in this project
        - `img_processing/`
            - `augmentations.py` data augmentations
            - `post_processing.py` functions used in post processing inst maps 
            - `pre_processing.py` functions used for patches pre processing patches before training
            - `process_utils.py` utility functions for img and mask processing
            - `viz_utils.py` utility functions for plotting results
        - `metrics/`
            - `metrics.py` benchmarking metrics functions 
        - `tune/`
            - `tune_experiment.py` TODO
        - `utils/`
            - `data_writer.py` class for writing image and mask patches for training
            - `file_manager.py` class for handling the downloaded data and managing the processed images and mask files and paths
            - `patch_extractor.py` class for extracting patches from images
        - `settings.py` A convention for building paths

## References

- [1] N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, "A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology," in IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550-1560, July 2017 
- [2] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019. [doi]
- [3] Q D Vu, S Graham, T Kurc, M N N To, M Shaban, T Qaiser, N A Koohbanani, S A Khurram, J Kalpathy-Cramer, T Zhao, R Gupta, J T Kwak, N Rajpoot, J Saltz, K Farahani. Methods for Segmentation and Classification of Digital Microscopy Tissue Images. Frontiers in Bioengineering and Biotechnology 7, 53 (2019).  
- [4] Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A., Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset Extension, Insights and Baselines arXiv preprint arXiv:2003.10778.
- [5] Caicedo, J.C., Goodman, A., Karhohs, K.W. et al. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nat Methods 16, 1247–1253 (2019)
