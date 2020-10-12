import math
import numpy as np
import sklearn.feature_extraction.image
from typing import List, Dict, Tuple

# Patch Extractor from hover-net repo
# https://github.com/vqdang/hover_net/blob/master/src/misc/patch_extractor.py
# Added some methods
class PatchExtractor:
    """
    Extractor to generate patches with or without padding.
    """
    def __get_patch(self, x, ptx, patch_size):
        pty = (ptx[0]+patch_size[0],
               ptx[1]+patch_size[1])
        win = x[ptx[0]:pty[0], 
                ptx[1]:pty[1]]
        
        assert win.shape[0] == patch_size[0] and \
               win.shape[1] == patch_size[1],    \
               '[BUG] Incorrect Patch Size {0}'.format(win.shape)
        
        return win
    
    def extract_valid(self, 
                      x: np.ndarray,
                      patch_size: Tuple[int],
                      stride_size: Tuple[int]) -> np.ndarray:
        """
        Extracts patches without padding, only work in case patch_size > stride_size
        
        Note: to deal with the remaining portions which are at the boundary a.k.a
        those which do not fit when slide left->right, top->bottom), we flip 
        the sliding direction then extract 1 patch starting from right / bottom edge. 
        There will be 1 additional patch extracted at the bottom-right corner
        Args:
            x           : input image, should be of shape HWC
            patch_size  : a tuple of (h, w)
            stride_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x
        """

        im_h = x.shape[0] 
        im_w = x.shape[1]

        def extract_infos(length, patch_size, stride_size):
            flag = (length - patch_size) % stride_size != 0
            last_step = math.floor((length - patch_size) / stride_size)
            last_step = (last_step + 1) * stride_size
            return flag, last_step

        h_flag, h_last = extract_infos(im_h, patch_size[0], stride_size[0])
        w_flag, w_last = extract_infos(im_w, patch_size[1], stride_size[1])    

        sub_patches = []
        #### Deal with valid block
        for row in range(0, h_last, stride_size[0]):
            for col in range(0, w_last, stride_size[1]):
                win = self.__get_patch(x, (row, col), patch_size[0])
                sub_patches.append(win)  
        #### Deal with edge case
        if h_flag:
            row = im_h - patch_size[0]
            for col in range(0, w_last, stride_size[1]):
                win = self.__get_patch(x, (row, col), patch_size[0])
                sub_patches.append(win)  
        if w_flag:
            col = im_w - patch_size[1]
            for row in range(0, h_last, stride_size[0]):
                win = self.__get_patch(x, (row, col), patch_size[0])
                sub_patches.append(win)  
        if h_flag and w_flag:
            ptx = (im_h - patch_size[0], im_w - patch_size[1])
            win = self.__get_patch(x, ptx, patch_size[0])
            sub_patches.append(win)  
        return np.asarray(sub_patches)
    
    def extract_mirror(self, 
                       x: np.ndarray,
                       patch_size: Tuple[int],
                       stride_size: Tuple[int]) -> np.ndarray:
        """
        Extracted patches with mirror padding the boundary such that the 
        central region of each patch is always within the orginal (non-padded)
        image while all patches' central region cover the whole orginal image
        Args:
            x         : input image, should be of shape HWC
            patch_size  : a tuple of (h, w)
            stride_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x
        """

        diff_h = patch_size[0] - stride_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = patch_size[1] - stride_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        pad_type = 'reflect'
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.extract_valid(x, patch_size, stride_size)
        return sub_patches

    @staticmethod
    def extract_test_patches(io: np.ndarray, 
                             input_size: int) -> np.ndarray:
        """
        Extract patches from the test set. (No need for mirror padding and overlapping patches)
        The size of the patches is the input size of the network. This is used when patches are
        saved to the hdf5 db
        Args:
            io (np.ndarray): input image
            input_size (int): size of the network input
        Returns:
            np.ndarray on shape (num_patches, input_size, input_size, 3)
        """
        # add extra padding to match an exact multiple of 32 patch size,
        extra_pad_row = int(np.ceil(io.shape[0] / input_size)*input_size - io.shape[0])
        extra_pad_col = int(np.ceil(io.shape[1] / input_size)*input_size - io.shape[1])
        io = np.pad(io, [(0, extra_pad_row), (0, extra_pad_col),
                        (0, 0)], mode="constant")

        # extract the patches from input images
        arr_out = sklearn.feature_extraction.image.extract_patches(
            io, (input_size, input_size, io.shape[-1]), input_size
        )

        # shape the dimensions to correct sizes for pytorch model
        test_patches = arr_out.reshape(-1, input_size, input_size, io.shape[-1])
        return test_patches

    @staticmethod
    def extract_inference_patches(im: np.ndarray,
                                  stride_size: int,
                                  input_size: int) -> np.ndarray:
        """
        Extract patches of (input_size, input_size, 3) from the input image
        of shape (H, W, 3) and append them to 4D array of shape 
        (num_patches, input_size, input_size, 3) where input_size is the size of
        the image that can be input to the CNN. This is used in inference

        Args:
            im (np.ndarray): input image if shape (H, W, 3)
            stride_size (int): size of the x & y axiss strides
            input_size (int): size of the square network input img.
        Returns:
            4D numpy array of shape (num_patches, input_size, input_size, 3)
            and a Tuple defining patches in a grid of shape
            (n_tile_x, n_tile_y, input_size, input_size, C)
        """
        # add reflection padding
        padx = stride_size//2
        pady = stride_size//2
        io = np.pad(im, [(padx, padx), (pady, pady), (0, 0)], mode="reflect")

        # add extra padding to match an exact multiple of 32 (smp models) patch size,
        extra_pad_row = int(
            np.ceil(io.shape[0] / input_size)*input_size - io.shape[0])
        extra_pad_col = int(
            np.ceil(io.shape[1] / input_size)*input_size - io.shape[1])
        io = np.pad(io, [(0, extra_pad_row),
                         (0, extra_pad_col), (0, 0)], mode="constant")

        # extract the patches from input images
        arr_out = sklearn.feature_extraction.image.extract_patches(
            io, (input_size, input_size, 3), stride_size
        )

        # shape the dimensions to correct sizes for pytorch model
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1, input_size, input_size, 3)
        return arr_out, arr_out_shape

    @staticmethod
    def stitch_inference_patches(patches: np.ndarray,
                                 stride_size: int,
                                 patches_shape: Tuple[int],
                                 im_shape: Tuple[int]) -> np.ndarray:
        """
        Stitch back patches. Inversion for self.extract_inference_patches
        Used in inference.
        Args:
            patches (np.ndarray): patches of shape (num_patches, input_size, input_size, C)
            stride_size (int): size of the x & y axis stride when img was patched
            input_size (int): size of the square network input img.
            patches_shape (Tuple[int]): Shape of patch grid (n_tile_x, n_tile_y, input_size, input_size, C)
            im_shape (Tuple[int]): shape of the original image (H, W, 3)
        """
        #turn from a single list into a matrix of tiles
        # (num_tile_x, num_tile_y, input_size, input_size, C)
        patches = patches.reshape(
            patches_shape[0],
            patches_shape[1],
            patches_shape[-3],
            patches_shape[-2],
            patches.shape[-1]
        )

        # remove the mirror padding from each tile, we only keep the center
        # (num_tile_x, num_tile_y, input_size//2, input_size//2, C)
        pad = stride_size//2
        patches = patches[:, :, pad:-pad, pad:-pad, :]

        # turn all the tiles into an image
        # (H+extra_padding, W+extra_padding, C)
        pred = np.concatenate(np.concatenate(patches, 1), 1)

        # incase there was extra padding to get a multiple of patch size, remove that as well
        # remove paddind, crop back (H, W, C)
        pred = pred[0:im_shape[0], 0:im_shape[1], :]
        return pred


        
