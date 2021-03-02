import numpy as np


def percentile_normalize(img: np.ndarray, channels: str="HWC") -> np.ndarray:
    """ 
    1-99 percentile normalization per image channel. Numpy version

    Args:
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(C, H, W)
        channels (str, default="HWC"):
            The order of image dimensions

    Returns:
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
        im = (im - percentile1) / (percentile99 - percentile1)

    if channels == "CHW":
        im = im.transpose(2, 0, 1)

    return im


def standardize(img: np.ndarray) -> np.ndarray:
    """
    Standardize per image channel

    img (np.ndarray):
        Input image to be normalized. Shape (H, W, C)|(C, H, W)
    """
    pass



def overlays(im: np.ndarray, mask:np.ndarray) -> np.ndarray:
    """
    Overlay mask with original image where there is no background
    mask is assumed to have shape (HxW)

    Args:
        im (np.ndarray): 
            Original image of shape (H, W, C)
        mask (np.ndarray): 
            Instance or type mask of the nucleis in the image
    """
    return np.where(mask[..., None], im, 0)