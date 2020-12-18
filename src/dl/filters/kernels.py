import torch

def gaussian(window_size: int, sigma: float, device: torch.device = None) -> torch.Tensor:
    """
    Create a gaussian 1D tensor

    Args:
        window_size (int): number of elements in the tensor
        sigma (float): std of the gaussian distribution
        device (torch.device): device for the tensor

    Returns:
        1D torch.Tensor of length window_size
    """
    x = torch.arange(window_size, device=device).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()


def gaussian_kernel2d(window_size: int, 
                      sigma: float,
                      n_channels: int = 1,
                      device: torch.device = None) -> torch.Tensor:
    """
    Create 2D window_size**2 sized kernel a gaussial kernel

    Args:
        window_size (int): size of the window
        sigma (float): std of the gaussian distribution
        n_channel (int): number of channels in the image that will be convolved with this kernel
        device (torch.device): device for the kernel

    Returns:
        torch.Tensor of shape (1, 1, win_size, win_size)(_, channel, height, width) = yhat.size()
    """
    kernel_x = gaussian(window_size, sigma, device=device)
    kernel_y = gaussian(window_size, sigma, device=device)
    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    kernel_2d = kernel_2d.expand(n_channels, 1, window_size, window_size)
    return kernel_2d


def sobel_hv(window_size: int = 5, device: torch.device = None):
    """
    Creates a kernel that is used to compute 1st order derivatives 
    as in the HoVer-net paper.   

    Args:
        window_size (int): size of the convolution kernel
        direction (str): direction of the derivative. One of ("x", "y")

    Returns:
        torch.Tensor computed 1st order derivatives of the input tensor. Shape (B, 2, H, W)
    """
    assert window_size % 2 == 1, f"size must be odd. size: {window_size}"

    # Generate the sobel kernels
    range_h = torch.arange(-window_size//2+1, window_size//2+1, dtype=torch.float32, device=device)
    range_v = torch.arange(-window_size//2+1, window_size//2+1, dtype=torch.float32, device=device)
    h, v = torch.meshgrid(range_h, range_v)

    kernel_h = h / (h*h + v*v + 1e-6)
    kernel_h = kernel_h.unsqueeze(0).unsqueeze(0)
    
    kernel_v = v / (h*h + v*v + 1e-6)
    kernel_v = kernel_v.unsqueeze(0).unsqueeze(0)
    
    return torch.cat([kernel_h, kernel_v], dim=0)
