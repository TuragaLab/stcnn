"""Convolution helper functions
help setup various convolution kernels
"""

# standard library imports
import math

# third party imports
import torch
from torch import Tensor


def gcamp_filter(x: Tensor, tau: Tensor, dt: float, t_kernel_size: int) -> Tensor:
    """
    simulate gcamp low-pass filter with single exponential kernel

    Args:
        x (Tensor): shape (batch, n_frames) or (n_frames)
        tau (Tensor): exp filter time constant
        dt (float): time step in seconds
        t_kernel_size (int): temporal kernel # of time steps

    Returns:
        x_filt (tensor): shape (batch, n_frames)
    """

    # reshape input x into shape (batch=1, in_channel=1, num_frames)
    if len(x.size()) == 1:  # has shape (num_frames)
        x = x[None, None, :]
    else:  # assume has shape (batch, num_frames)
        x = x[:, None, :]  # (batch_size, 1, num_frames)

    #   t_kernel_size = x.shape[2]  # make it as large as the input trace, or num_frames

    t_weight = kernel_singleExp(tau, t_kernel_size, dt)

    x_filt = torch.nn.functional.conv1d(
        input=x, weight=t_weight.flip(2), bias=None, padding=t_kernel_size - 1
    )

    x_filt = x_filt[
        :, :, : -t_kernel_size + 1
    ]  # (out_channels=1, in_channels=1, n_frames)
    x_filt = x_filt.squeeze(1)  # shape (batch, n_frames)

    return x_filt


def kernel_singleExp(tau: Tensor, kernel_size: int, dt: float) -> Tensor:
    """
    generate weights for simple exp(-t/tau) kernel

    Args:
        tau (Tensor): exp time constant, shape (out_channels, in_channels/t_groups)
        kernel_size (int): # of time steps in kernel
        dt (float): time step in seconds

    Returns:
        weight (tensor): shape (out_channels, in_channels/t_groups, kernel_size)
    """

    # t = time has shape (1, 1, num_time_steps)
    # create tensor to be on the same device as input tau
    t = dt * (
        torch.arange(start=0, end=kernel_size, dtype=torch.float32, device=tau.device)[
            None, None, :
        ]
    )

    if tau.dim() == 1:
        tau = tau[:, None]  # unsqueeze tensor if it's singleton dim

    weight = torch.exp(-t / tau[:, :, None])

    # L2 normalization for each channel along the time dimension=2
    # w_normalized = w / sqrt( sum(w**2)*dt )
    weight = weight / torch.linalg.norm(weight, dim=2)[:, :, None] / dt**0.5

    return weight


def kernel_PowerExp(
    n_power: Tensor, tau: Tensor, kernel_size: int, dt: float
) -> Tensor:
    """generate weights for temporal kernel that has separate rise and decay
    time constants
    weight = (t/tau)^n exp(-n(t/tau-1))

    Args:
        n_power (Tensor): exponent
        tau (Tensor): exp time constant
        kernel_size (int): # steps in time
        dt (float): step size in sec

    Returns:
        Tensor: L2 normalized weights
    """

    t = dt * (
        torch.arange(start=0, end=kernel_size, dtype=torch.float32, device=tau.device)[
            None, None, :
        ]
    )

    if tau.dim() == 1:
        tau = tau[:, None]  # unsqueeze tensor if it's singleton dim

    if n_power.dim() == 1:
        n_power = n_power[:, None]

    # at t=0, weight=0
    # at t=tau, weight=1 and is the peak value
    weight = torch.pow(t / tau[:, :, None], n_power[:, :, None]) * torch.exp(
        -n_power[:, :, None] * (t / tau[:, :, None] - 1)
    )

    # L2 normalization for each channel along the time dimension=2
    # w_normalized = w / sqrt( sum(w**2)*dt )
    weight = weight / torch.linalg.norm(weight, dim=2)[:, :, None] / dt**0.5

    return weight


def kernel_gauss_photoreceptor(
    sigma_row: float,
    sigma_col: float,
    ds_row: float,
    ds_col: float,
    kernel_size_row: int,
    kernel_size_col: int,
) -> Tensor:
    """
    generate spatial weights for 2D gaussian sampling on a rectangular lattice
    kernel form is from Tuthill et al 2011 PNAS equation S1

    Note: kernel is defined with respect to the input movie lattice
    if kernel size is odd, gaussian peak is at the center pixel
    if kernel size is even, gaussian peak is not in the kernel, but the nearby
    pixels to center correspond to values nearby gaussian peak

    Args:
        sigma_row, sigma_col (float):
            gaussian standard deviation along row and col dimensions, for photoreceptor both should = 5 deg
        ds_row, ds_col (float):
            implicit input movie resolution
            visual angle (degrees) per pixel along row and col
        dt (float):
            time step in seconds
        kernel_size_row, kernel_size_col (int):
            size of spatial kernel

    Returns:
        weight (tensor):
            shape (out_channels=1, in_channels/groups=1, kernel_size_row, kernel_size_col)
    """

    # generate a meshgrid where the center position is 0,0
    y_row_grid, x_col_grid = _gen_xy_meshgrid(kernel_size_row, kernel_size_col)

    # convert meshgrid to values matching input movie x/y pixel resolution
    y_row_grid = y_row_grid * ds_row
    x_col_grid = x_col_grid * ds_col

    # gaussian exp contains angle^2/angle^2
    # so it's not important to have deg as the unit, can use radian also, but must be consistent
    weight = torch.exp(
        -4
        * math.log(2)
        * (torch.pow(y_row_grid, 2) + torch.pow(x_col_grid, 2))
        / (sigma_row * sigma_col)
    )
    weight = weight / torch.sum(weight)  # normalize sum = 1

    weight.unsqueeze_(0).unsqueeze_(0)  # shape (out_ch, in_ch, row_size, col_size)

    return weight


def _gen_xy_meshgrid(y_row_len: int, x_col_len: int) -> tuple[Tensor, Tensor]:
    """meshgrid for x and y separately, each centered at 0,0
    values are either integers or multiples of 0.5

    Args:
        y_row_len, x_col_len (int): length along row and col dimensions

    Returns:
        x_grid, y_grid (tensor): coordinates along x/row and y/col
    """

    # generate list of x, y coordinates separately first
    y_row_vec = torch.arange(start=0, end=y_row_len, step=1)
    x_col_vec = torch.arange(start=0, end=x_col_len, step=1)

    # now shift these coordinates to be centered around zero
    y_row_vec = _zero_center_vec(y_row_vec)
    x_col_vec = _zero_center_vec(x_col_vec)

    # now replicate these tensors along the opposite dim to generate meshgrids
    # dim0 = row, dim1 = col
    y_row_vec = y_row_vec.unsqueeze_(1)
    y_row_meshgrid = y_row_vec.repeat(1, x_col_len)  # tile along dim1

    x_col_vec = x_col_vec.unsqueeze_(0)
    x_col_meshgrid = x_col_vec.repeat(y_row_len, 1)  # tile along dim0

    return y_row_meshgrid, x_col_meshgrid


def _zero_center_vec(vec: Tensor) -> Tensor:
    """
    input vec is assumed to monotonically increment in value and evenly spaced
    function zero centers vec values

    Args:
        vec (tensor): should be one dimensional tensor

    Returns:
        vec (tensor): values are centered around zero now
    """

    vec = vec.squeeze() - vec.min()  # make vec[0] = 0
    max_val = vec.max()

    shift = 0
    if (
        len(vec) % 2 == 0
    ):  # even number of points in vec, then no point can = 0 exactly, have to shift by decimals
        shift = torch.div(max_val, 2.0)
    else:  # odd number of points in vec, the center index should be exactly = 0
        shift = torch.div(max_val, 2.0, rounding_mode="floor")

    vec = vec - shift

    return vec
