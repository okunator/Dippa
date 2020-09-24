# Dippa
Benchmarking of deep learning and other segmentation methods for H&amp;E images  
Borrowing a lot of methods from HoVer-Net [repo](https://github.com/vqdang/hover_net): and methodology from HoVer-Net paper [2]  
**Note**: dl framework is PyTorch 

## Set Up
1. Clone the repository
2. cd to the repository `cd <path>/Dippa/`
3. Create environment (Optional but recommended) 
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

## Instructions for downloading the datasets
1. Download the datsets from the links below
2. Move the downloaded zip files or extract the contents of the zip files to the corresponding folders in `datasets/` folder
3. Run the `src/convert_raw_data.py` script or `1_Convert_downloaded_data.ipynb` notebook to convert the data to right format for training and benchmarking.

#### Data download links:
1. **kumar**  
          - Train: https://drive.google.com/file/d/1JZN9Jq9km0rZNiYNEukE_8f0CsSK3Pe4/view .   
          - Test: https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view  
          
2. **consep** - https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/
3. **cpm** - TODO
4. **pannuke** - https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke
5. **kaggle dsb 2018**: https://bbbc.broadinstitute.org/BBBC038

## References

- [1] N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, "A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology," in IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550-1560, July 2017 
- [2] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019. [doi]
- [3] Q D Vu, S Graham, T Kurc, M N N To, M Shaban, T Qaiser, N A Koohbanani, S A Khurram, J Kalpathy-Cramer, T Zhao, R Gupta, J T Kwak, N Rajpoot, J Saltz, K Farahani. Methods for Segmentation and Classification of Digital Microscopy Tissue Images. Frontiers in Bioengineering and Biotechnology 7, 53 (2019).  
- [4] Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A., Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset Extension, Insights and Baselines arXiv preprint arXiv:2003.10778.
- [5] Caicedo, J.C., Goodman, A., Karhohs, K.W. et al. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nat Methods 16, 1247–1253 (2019)
