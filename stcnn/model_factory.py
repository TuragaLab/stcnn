"""Generate complete network model by composing various layers

"""

# standard library imports
import sys

# third party imports
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# local application imports
from stcnn.models import (
    STlayerPhotoreceptor,
    STlayerExpTemporal,
    STlayerPowerExpTemporal,
    STBatchNorm,
    LCModelBatchnorm,
)
from stcnn.util_setup import layer_param_factory


def model_stcnn_factory(
    pr_config: dict,
    network_config: dict,
    L_config: dict,
    ca_config: dict,
    sampling_rate: float,
    device: torch.device,
) -> torch.nn.Module:
    """
    generate full spatiotemporal conv network based on user config inputs
    this model stacks layers in the following order
        photoreceptor layer
        (spatiotemporal conv followed by batch norm) * # of hidden layers
        parallel readout layers, one for each unique cell type

    Args:
        pr_config (dict): photoreceptor layer config
        network_config (dict): hidden layer config
        L_config (dict): cell type specific voltage read out layers
        ca_config (dict): cell type specific calcium read out layers
        sampling_rate (float)
        device (device)

    Returns:
        model (torch.nn.Module): full network model
    """

    # generate the layers for the conv network
    pr = pr_factory(pr_config, sampling_rate, device)  # a single layer
    network = network_factory(network_config, sampling_rate, device)  # list of layers
    norm_layers = batchnorm_factory(network_config)
    lc_layers = L_factory(
        L_config, sampling_rate, device
    )  # parallel output layers, used to train on calcium imaging data

    gcamp_config_list = layer_param_factory(ca_config, sampling_rate, device)
    cfg = gcamp_config_list[0]

    model = LCModelBatchnorm(
        pr,
        network,
        norm_layers,
        lc_layers,
        device=device,
        **cfg,
    )

    return model


def pr_factory(
    init_config: dict, sampling_rate: float, device: torch.device
) -> STlayerPhotoreceptor:
    """
    generate photoreceptor layer

    Args:
        init_config (dict): param config
        sampling_rate (float): steps per sec, should match input movie
        device (torch.device): device

    Returns:
        pr (STlayerPhotoreceptor)
    """
    param_list = layer_param_factory(init_config, sampling_rate, device)

    cfg = param_list[0]  # assume only one layer

    pr = STlayerPhotoreceptor(
        device=device,
        **cfg,
    )

    return pr


def network_factory(
    init_config: dict, sampling_rate: float, device: torch.device
) -> list[torch.nn.Module]:
    """
    generate the hidden spatiotemporal conv layers

    Args:
        init_config (dict): param config
        sampling_rate (float): steps per sec, should match input movie
        device (torch.device): device

    Returns:
        network (list): spatiotemporal conv layers
    """

    network = []

    param_list = layer_param_factory(init_config, sampling_rate, device)

    for i in range(len(param_list)):
        cfg = param_list[i]

        if cfg["config_type"].lower() == "spatiotemporal_exp":
            layer = STlayerExpTemporal(
                device=device,
                **cfg,
            )
        elif cfg["config_type"].lower() == "spatiotemporal_pwrexp":
            layer = STlayerPowerExpTemporal(
                device=device,
                **cfg,
            )
        network.append(layer)

    return network


def L_factory(
    init_config: dict, sampling_rate: float, device: torch.device
) -> list[torch.nn.Module]:
    """
    generate voltage readout layers for each cell type

    Args:
        init_config (dict): param config
        sampling_rate (float): steps per sec, should match input movie
        device (torch.device): device

    Returns:
        L_layers (list): spatiotemporal conv layers
    """

    L_layers = []

    num_lc_types = len(init_config["lc_types"])

    # generate each LC layer specified in the init_config dict
    # these are parallel output layer used for training to calcium imaging data
    for i in range(num_lc_types):
        init_cfg = init_config.copy()

        init_cfg["x_kernel_size"] = init_config["x_kernel_size"][i]  # list
        init_cfg["x_stride"] = init_config["x_stride"][i]  # list
        init_cfg["x_padding"] = init_config["x_padding"][i]  # list

        init_cfg["layer_name"] = init_config["lc_types"][
            i
        ]  # used to identify individual L_layer for loss calculations

        param_list = layer_param_factory(init_cfg, sampling_rate, device)

        cfg = param_list[0]

        if cfg["config_type"].lower() == "spatiotemporal_exp":
            layer = STlayerExpTemporal(
                device=device,
                **cfg,
            )
        elif cfg["config_type"].lower() == "spatiotemporal_pwrexp":
            layer = STlayerPowerExpTemporal(
                device=device,
                **cfg,
            )

        L_layers.append(layer)

    return L_layers


def batchnorm_factory(init_config: dict) -> list[STBatchNorm]:
    """
    generate batchnorm layers

    Args:
        init_config (dict): param config

    Returns:
        norm_layers (list of STBatchNorm)
    """

    norm_layers = []

    num_layers = len(init_config["out_channels"])

    for i in range(num_layers):
        norm_layers.append(
            STBatchNorm(num_features=init_config["out_channels"][i])
        )  # batchnorm performed after conv layer, # features = out_channels of conv layer

    return norm_layers
