"""Functions for setting up neural networks

including parameter initializations for various types of layers

"""

# standard library imports
import math
from typing import Union, Callable
import sys

# third party imports
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# local application imports
from stcnn.util_conv import kernel_gauss_photoreceptor


def activation_factory(name: str) -> Callable[[Tensor], Tensor]:
    """
    shorthand for activation functions

    Args:
        name (str)

    Returns:
        activation function
    """

    d = {
        "identity": lambda x: x,
        "relu": F.relu,
        "tanh": F.tanh,
        "leaky_relu": F.leaky_relu,
        "elu": F.elu,
    }
    return d[name]


def layer_param_factory(
    config: dict, sampling_rate: Union[int, float], device: torch.device
) -> list[dict]:
    """
    generate config files for initializating network
    Kaiming fan-in initialization is used for spatial weights

    Args:
        config (dict): param initial settings
        sampling_rate: integer or float specified in Hz
        device (device)

    Returns:
        param_list (list of dict):
            each dict specifies a layer param values
    """

    # given config file, generate one param dict for each layer
    # save results into a param list containing dict
    param_list = []

    # define common param for spatial convolution
    x_groups = 1  # groups param for conv2d, want spatial to convolve all input channels to each output channel

    # define common param for temporal convolution
    dt = 1 / sampling_rate

    # custom param for different layer types
    if config["config_type"].lower() == "photoreceptor":
        param = {}

        # define number of input/output channels for the current layer
        in_channels = config["in_channels"][0]
        out_channels = config["out_channels"][0]
        t_in_channels = (
            out_channels  # input channel to temporal = output channel for spatial
        )
        t_out_channels = t_in_channels  # 1:1 mapping
        t_groups = out_channels

        # copy settings from config
        param["in_channels"] = in_channels
        param["out_channels"] = out_channels

        param["x_kernel_size"] = config["x_kernel_size"]
        param["x_ds"] = config["ds_input"]
        param["pr_sigma"] = config["pr_sigma"]
        param["x_stride"] = config[
            "x_stride"
        ]  # (int or list) single number for both row and col
        param["x_padding"] = config[
            "x_padding"
        ]  # (int or list) single number for both row and col
        param["x_num_groups"] = x_groups

        param["dt"] = dt

        # generate gaussian spatial kernel
        param["x_weight_init"] = kernel_gauss_photoreceptor(
            param["pr_sigma"][0],
            param["pr_sigma"][1],
            param["x_ds"][0],
            param["x_ds"][1],
            param["x_kernel_size"][0],
            param["x_kernel_size"][1],
        )

        param["tau_init"] = config["tau"] * torch.ones(
            t_out_channels, int(t_in_channels / t_groups)
        )
        param["t_kernel_size"] = config["t_kernel_size"]

        param["layer_name"] = config["layer_name"]

        param["activation_fn"] = activation_factory("identity")

        param_list.append(param)

    elif (
        config["config_type"].lower() == "spatiotemporal_exp"
        or config["config_type"].lower() == "spatiotemporal_pwrexp"
        or config["config_type"].lower() == "spatial"
    ):  # the default spatiotemporal conv layer param with uniform distrib initialization
        """
        spatiotemporal conv layer is
        output y = T * W * x + b
        T: temporal weights (only tau is learned, weights are not trainable)
        W: spatial weights (trainable)
        b: bias
        """

        num_layers = len(
            config["out_channels"]
        )  # total # of layers is determined by length of out_channels

        for k in range(num_layers):
            param = {}

            param["config_type"] = config["config_type"]

            # define # of input and output channels for the current layer
            if k == 0:
                in_channels = config["in_channels"][k]  # use user defined number
            else:
                in_channels = config["out_channels"][
                    k - 1
                ]  # output of previous layer is the input of current layer

            out_channels = config["out_channels"][k]
            t_in_channels = (
                out_channels  # input channel to temporal = output channel for spatial
            )
            t_out_channels = t_in_channels  # 1:1 mapping
            t_groups = out_channels

            # get spatial kernel size
            if type(config["x_kernel_size"][k]) is list:
                x_kernel_size_row = config["x_kernel_size"][k][0]
                x_kernel_size_col = config["x_kernel_size"][k][1]
            elif type(config["x_kernel_size"][k]) is int:
                x_kernel_size_row = config["x_kernel_size"][k]
                x_kernel_size_col = config["x_kernel_size"][k]

            # copy conv settings
            param["in_channels"] = in_channels
            param["out_channels"] = out_channels
            param["x_num_groups"] = x_groups
            param["x_stride"] = config["x_stride"][k]  # can be either int or list
            param["x_padding"] = config["x_padding"][k]  # can be either int or list
            param["dt"] = dt

            if not config["config_type"].lower() == "spatial":
                param["t_kernel_size"] = config["t_kernel_size"]

            # kaiming fan-in uniform initialiation
            param["x_weight_init"] = torch.empty(
                out_channels,
                int(in_channels / x_groups),
                x_kernel_size_row,
                x_kernel_size_col,
            )
            torch.nn.init.kaiming_uniform_(param["x_weight_init"], a=0.01)
            param["x_weight_init"] = (
                param["x_weight_init"] * config["init_amplitude_scale"]
            )

            # initialize bias to zero
            # not necessary to have bias with batchnorm, but need it for lc layer
            if "bias_init" in config:
                param["bias_init"] = torch.empty(t_out_channels)
                param["bias_init"].uniform_(
                    config["bias_init"][0],
                    config["bias_init"][1],
                )
            else:  # default fill with zeros
                param["bias_init"] = torch.zeros(t_out_channels)

            # additional power parameter for power-exp temporal kernel
            if config["config_type"].lower() == "spatiotemporal_pwrexp":
                param["n_power_init"] = torch.empty(
                    t_out_channels, int(t_in_channels / t_groups)
                )
                param["n_power_init"].uniform_(
                    config["n_power_init"][0], config["n_power_init"][1]
                )

            if not config["config_type"].lower() == "spatial":
                # generate log-uniformly distributed tau init values for temporal kernel
                base_ten = 10 * torch.ones(
                    t_out_channels, int(t_in_channels / t_groups)
                )
                exponent_init = torch.empty(
                    t_out_channels, int(t_in_channels / t_groups)
                )
                exponent_init.uniform_(config["tau_init"][0], config["tau_init"][1])
                param["tau_init"] = torch.pow(base_ten, exponent_init)

            if type(config["activation"]) is list:
                param["activation_fn"] = activation_factory(config["activation"][k])
            elif type(config["activation"]) is str:
                param["activation_fn"] = activation_factory(config["activation"])

            if num_layers == 1:
                param["layer_name"] = config["layer_name"]
            else:
                param["layer_name"] = config["layer_name"] + str(k)

            param_list.append(param)

    elif config["config_type"].lower() == "gcamp":  # calcium readout layer
        param = {}

        param["dt"] = dt
        param["t_kernel_size"] = config["t_kernel_size"]

        param["cell_types"] = config["cell_types"]
        param["cell_count"] = config["cell_count"]

        param["alpha_init"] = config["alpha_init"]
        param["beta_init"] = config["beta_init"]
        param["ca_tau_init"] = config["gcamp_tau"]
        param["use_global_ca_tau"] = (
            config["use_global_ca_tau"]
            if "use_global_ca_tau" in config.keys()
            else False
        )

        param["activation_fn"] = activation_factory(config["activation"])

        param_list.append(param)

    return param_list
