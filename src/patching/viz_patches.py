import numpy as np
from typing import Tuple
from skimage.color import label2rgb

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib required. `pip install matplotlib`")


def viz_patches(patches: np.ndarray) -> Tuple[int]:
    """
    patches is assumed to be of shape (n_patches, H, W, n_channels)
    This function vizualizes those patches. Don't put too many patches 
    in or everything willl crash.

    Args:
    ----------
        patches (np.ndarray): 
            numpy array of stacked image patches.
            Shape: (n_patches, H, W, n_channels)

    Returns:
    ----------
        Tuple: Shape of the patches array
    """
    fignum = 200
    low=0
    high=len(patches)

    # Visualize
    fig_patches = plt.figure(fignum, figsize=(35,35))
    pmin, pmax = patches.min(), patches.max()
    dims = np.ceil(np.sqrt(high - low))
    for idx in range(high - low):
        spl = plt.subplot(dims, dims, idx + 1)
        ax = plt.axis("off")

        if len(patches[idx].shape) == 2:
            patch=label2rgb(patches[idx], bg_label=0)
        else:
            patch = patches[idx]
            if patch.dtype == np.dtype("float64"):
                patch = (patch * 255)
            patch = patch.astype(np.uint8)
            
        im = plt.imshow(patch)
        cl = plt.clim(pmin, pmax)
    plt.show()
    return patches.shape