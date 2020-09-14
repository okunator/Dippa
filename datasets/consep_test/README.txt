Colorectal Nuclear Segmentation and Phenotypes (CoNSeP) Dataset

----------------------------------------------------------------------------------------------------
Overview:

This dataset was first used in our paper named,

"HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images"

If using any part of this dataset or the code associated, you must give appropriate citation to our paper, published in Medical Image Analysis.

----------------------------------------------------------------------------------------------------

Dataset Description:

Each ground truth file is stored as a .mat file, with the keys:
'inst_map'
'type_map'
'inst_type'
'inst_centroid'
 
'inst_map' is a 1000x1000 array containing a unique integer for each individual nucleus. i.e the map ranges from 0 to N, where 0 is the background and N is the number of nuclei.

'type_map' is a 1000x1000 array where each pixel value denotes the class of that pixel. The map ranges from 0 to 7, where 7 is the total number of classes in CoNSeP.

'inst_type' is a Nx1 array, indicating the type of each instance (in order of inst_map ID)

'inst_centroid' is a Nx2 array, giving the x and y coordinates of the centroids of each instance (in order of inst map ID).

Note, 'inst_type' and 'inst_centroid' are only used while computing the classification statistics. 

The values within the class map indicate the category of each nucleus.  

Class values: 1 = other
	      2 = inflammatory
	      3 = healthy epithelial
	      4 = dysplastic/malignant epithelial
              5 = fibroblast
              6 = muscle
	      7 = endothelial

In our paper we combine classes 3 & 4 into the epithelial class and 5,6 & 7
into the spindle-shaped class.

Total number of nuclei = 24,319

Note, as of 14/03/2020, we switched the GT from .npy files to .mat files for 
consistency with the Github repo

------------------------------------------------------------------------------------------
