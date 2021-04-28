import numpy as np
from skimage import img_as_ubyte


def percentile_normalize(img: np.ndarray, channels: str="HWC") -> np.ndarray:
    """ 
    1-99 percentile normalization per image channel. Numpy version

    Args:
    -----------
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(C, H, W)
        channels (str, default="HWC"):
            The order of image dimensions

    Returns:
    -----------
        np.ndarray = Normalized image. Same shape as input
    """
    assert channels in ("HW", "HWC", "CHW")

    im = img.copy()
    axis = (0, 1)

    if channels == "CHW":
        im = im.transpose(1, 2, 0)

    if np.all(np.ptp(im, axis=axis) > 0.0):
        percentile1 = np.percentile(im, q=1, axis=axis)
        percentile99 = np.percentile(im, q=99, axis=axis)
        im = (im - percentile1) / (percentile99 - percentile1 + 1e-7)

    if channels == "CHW":
        im = im.transpose(2, 0, 1)

    return im


def percentile_normalize_and_clamp(im: np.ndarray, a_min: float=-1, a_max: float=1) -> np.ndarray:
    """
    1-99-percentile normalization + clamping to [a_min, a_max].
    Assumes that input image is "HWC" or "HW". Does not handle
    different channel orderings

    Args:
    ----------
        im (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(H, W)
        a_min (float):
            clamp min value
        a_max (float):
            clamp max value
    
    Returns:
        np.ndarray normed input image of same shape as input.
    """
    percentile99 = np.percentile(im, q=99, axis=(0, 1))
    percentile1 = np.percentile(im, q=1, axis=(0, 1))
    percentiles = np.stack([percentile99, percentile1])
    colmax = np.max(percentiles, axis=0)
    normed = np.clip((im / (colmax + 1e+7)), a_min=a_min, a_max=a_max)
    return normed


def normalize(img: np.ndarray, channels: str="HWC", standardize: bool=True) -> np.ndarray:
    """
    Mean center or standardize per image channel

    Args:
    -----------
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(C, H, W)
        channels (str, default="HWC"):
            The order of image dimensions
        standardize (bool, default=True):
            If True, divide with standard deviation after mean centering

    Returns:
    -----------
        np.ndarray = Normalized image. Same shape as input
    """
    im = img.copy()
    axis = (0, 1)

    if channels == "CHW":
        im = im.transpose(1, 2, 0)

    if np.all(np.ptp(im, axis=axis) > 0.0):
        im = im - im.mean(axis=axis, keepdims=True)
        
        if standardize:
            im /= im.std(axis=axis, keepdims=True)

    if channels == "CHW":
        im = im.transpose(2, 0, 1)

    return im


def minmax_normalize(img: np.ndarray, channels: str="HWC") -> np.ndarray:
    """
    Min-max normalization per image channel

    Args:
    -----------
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(C, H, W)
        channels (str, default="HWC"):
            The order of image dimensions

    Returns:
    -----------
        np.ndarray = Min-max normalized image. Same shape as input
    """
    im = img.copy()
    axis = (0, 1)

    if channels == "CHW":
        im = im.transpose(1, 2, 0)

    if np.all(np.ptp(im, axis=axis) > 0.0):
        im = (im - im.min()) / (im.max() - im.min())  

    if channels == "CHW":
        im = im.transpose(2, 0, 1)

    return im


def float2ubyte(mat: np.ndarray, channels: str="HWC", normalize: bool=False) -> np.ndarray:
    """
    Convert float64 to uint8. float matrix values need to be in [-1, 1] for img_as_ubyte
    So the image is normalized or clamped before conversion.

    Args:
    ------------
        mat (np.ndarray):
            float64 matrix. Shape (H, W, C)|(C, H, W)
        channels (str, default="HWC"):
            The order of image dimensions
        normalize (bool, default=False):
            Normalizes input to [0, 1] first. If not True,
            clips values between [-1, 1].

    Returns:
    ------------
        np.ndarray = uint8 matrix. Shape (H, W, C)
    """
    m = mat.copy()

    if channels == "CHW":
        m = m.transpose(1, 2, 0)
    
    if normalize:
        m = minmax_normalize(m)
    else:
        m = np.clip(m, a_min=-1, a_max=1)
    
    return img_as_ubyte(m)


def overlays(im: np.ndarray, mask:np.ndarray) -> np.ndarray:
    """
    Overlay mask with original image where there is no background
    mask is assumed to have shape (HxW)

    Args:
    ------------
        im (np.ndarray): 
            Original image of shape (H, W, C)
        mask (np.ndarray): 
            Instance or type mask of the nucleis in the image
    """
    return np.where(mask[..., None], im, 0)