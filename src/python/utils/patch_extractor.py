import math
import numpy as np

# Patch Extractor from hover-net repo
# https://github.com/vqdang/hover_net/blob/master/src/misc/patch_extractor.py
#####
class PatchExtractor:
    """
    Extractor to generate patches with or without padding.
    Args:
        x         : input image, should be of shape HWC
        patch_size  : a tuple of (h, w)
        stride_size : a tuple of (h, w)
    Return:
        a list of sub patches, each patch has dtype same as x
    Examples:
        >>> xtractor = PatchExtractor((450, 450), (120, 120))
        >>> img = np.full([1200, 1200, 3], 255, np.uint8)
        >>> patches = xtractor.extract(img, 'mirror')
    """
    def __init__(self, patch_size, stride_size):

        self.patch_type = 'mirror'
        self.patch_size  = patch_size
        self.stride_size = stride_size
        self.counter = 0

    def __get_patch(self, x, ptx):
        pty = (ptx[0]+self.patch_size[0],
               ptx[1]+self.patch_size[1])
        win = x[ptx[0]:pty[0], 
                ptx[1]:pty[1]]
        
        assert win.shape[0] == self.patch_size[0] and \
               win.shape[1] == self.patch_size[1],    \
               '[BUG] Incorrect Patch Size {0}'.format(win.shape)
        
        return win
    
    def __extract_valid(self, x):
        """
        Extracted patches without padding, only work in case patch_size > stride_size
        
        Note: to deal with the remaining portions which are at the boundary a.k.a
        those which do not fit when slide left->right, top->bottom), we flip 
        the sliding direction then extract 1 patch starting from right / bottom edge. 
        There will be 1 additional patch extracted at the bottom-right corner
        Args:
            x         : input image, should be of shape HWC
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

        h_flag, h_last = extract_infos(im_h, self.patch_size[0], self.stride_size[0])
        w_flag, w_last = extract_infos(im_w, self.patch_size[1], self.stride_size[1])    

        sub_patches = []
        #### Deal with valid block
        for row in range(0, h_last, self.stride_size[0]):
            for col in range(0, w_last, self.stride_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        #### Deal with edge case
        if h_flag:
            row = im_h - self.patch_size[0]
            for col in range(0, w_last, self.stride_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        if w_flag:
            col = im_w - self.patch_size[1]
            for row in range(0, h_last, self.stride_size[0]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        if h_flag and w_flag:
            ptx = (im_h - self.patch_size[0], im_w - self.patch_size[1])
            win = self.__get_patch(x, ptx)
            sub_patches.append(win)  
        return np.asarray(sub_patches)
    
    def __extract_mirror(self, x):
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

        diff_h = self.patch_size[0] - self.stride_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = self.patch_size[1] - self.stride_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        pad_type = 'reflect'
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.__extract_valid(x)
        return sub_patches

    def extract(self, x, patch_type):
        patch_type = patch_type.lower()
        self.patch_type = patch_type
        if patch_type == 'valid':
            return self.__extract_valid(x)
        elif patch_type == 'mirror':
            return self.__extract_mirror(x)
        else:
            assert False, 'Unknown Patch Type [%s]' % patch_type
        return
#####
