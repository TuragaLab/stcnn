"""Models for layers and networks

"""

# standard library imports
from typing import Union, Callable
import sys

# third party imports
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# local application imports
from stcnn.util_conv import gcamp_filter, kernel_singleExp, kernel_PowerExp


class _SpatiotemporalLayerBase(torch.nn.Module):
    """Spatiotemporal convolution layer base class"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        x_stride: int,
        x_padding: int,
        x_num_groups: int,
        x_weight_init: Tensor,
        dt: torch.float,
        t_kernel_size: int,
        layer_name: str,
        activation_fn: Callable[[Tensor], Tensor],
        device: torch.device,
        **kwargs,
    ) -> None:
        """initialize weights for one layer

        Args:
            in_channels (int): input channel #
            out_channels (int): output channel or # features
            x_stride (int): spatial conv
            x_padding (int): spatial conv
            x_num_groups (int): controls mapping from input to output
            x_weight_init (Tensor): user provided initial weights
            dt (torch.float): step size in sec
            t_kernel_size (int): # of time points
            layer_name (str): name used to access layer
            activation_fn (Callable[[Tensor], Tensor]): activation
            device (torch.device): device
        """

        super(_SpatiotemporalLayerBase, self).__init__()

        self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels

        # spatial conv parameters
        self.x_stride = x_stride
        self.x_padding = x_padding
        self.x_groups = (
            x_num_groups  # used in _spatial_conv method, expect this = 1 always
        )

        self.x_weight = torch.nn.Parameter(
            x_weight_init
        )  # shape (out_channels, in_channels/x_groups, row, col)

        # temporal conv parameters
        self.dt = dt
        self.t_kernel_size = t_kernel_size
        self.t_padding = self.t_kernel_size - 1
        self.t_groups = (
            self.out_channels
        )  # used in _temporal_conv method, defined this way to make sure only one input channel is conv to form one corresponding output channel

        # tensors have to be either param or buffers to be auto moved when using model.to(device)
        # do not make buffer persistent since t_weight size will change in the forward pass
        # difference in t_weight tensor size will throw error when loading state_dict
        self.register_buffer(
            "t_weight", torch.ones(out_channels, 1, 1), persistent=False
        )  # identity filter, shape (out_channels, in_channels/t_groups, n_frames)

        self.bias = None

        self.name = layer_name

        self.activation = activation_fn

    def forward(self, x: Tensor) -> Tensor:
        """
        template forward pass logic

        tensor has shape (batch, n_frames, channel, rows, cols)

        x and t weights can be updated based on user specified functional forms
        (e.g. gaussian kernel, exponential filter, etc...)

        activation is called before spatial + temporal convolution
        so that the layer output voltage is not rectified
        """

        x = self.activation(x)
        x = self._spatial_conv(x)
        x = self._temporal_conv(x)

        return x

    def _gen_t_weights(self):
        """
        update temporal kernel weights
        """
        pass

    def _spatial_conv(self, x: Tensor) -> Tensor:
        """
        2D convolution along spatial (rows, cols) dimensions
        convolves all input channels for each output channel
        uses kernel stored in self.x_weight
        no bias shifts at all

        Args:
            x (tensor): shape (batch, n_frames, channel, rows, cols)

        Returns:
            x (tensor): shape (batch, n_frames, channel, rows, cols)
        """

        batch, n_frames, in_channels, n_rows_in, n_columns_in = x.shape

        x = x.reshape(batch * n_frames, in_channels, n_rows_in, n_columns_in)
        x = F.conv2d(
            input=x,
            weight=self.x_weight,
            bias=None,
            stride=self.x_stride,
            padding=self.x_padding,
            groups=self.x_groups,
        )

        (
            _,
            _,
            n_rows_out,
            n_columns_out,
        ) = x.shape  # (batch*n_frames, out_channels, n_rows_out, n_columns_out)

        x = x.reshape(batch, n_frames, self.out_channels, n_rows_out, n_columns_out)

        return x

    def _temporal_conv(self, x: Tensor) -> Tensor:
        """
        1D convolution along time (n_frames) dimension for each channel
        uses kernel stored in self.t_weight
        and shifts output with self.bias

        Args:
            x (tensor): shape (batch, n_frames, channel, rows, cols)

        Returns:
            x (tensor): shape (batch, n_frames, channel, rows, cols)
        """
        batch, n_frames, in_channels, n_rows_in, n_columns_in = x.shape

        self._gen_t_weights()

        x = x.permute(0, 3, 4, 2, 1).reshape(
            -1, self.out_channels, n_frames
        )  # (batch*n_rows_in*n_columns_in, out_channels, n_frames)
        x = F.conv1d(
            input=x,
            weight=self.t_weight.flip(2),  # a real convolution, flipped in time
            bias=self.bias,
            padding=self.t_padding,
            groups=self.t_groups,
        )

        # need to clip output due to zero padding from conv1d
        # note that temporal conv does not change any spatial dimension
        # since spatial is folded into batch dimension
        x = x[
            :, :, : -self.t_kernel_size + 1
        ]  # (batch*n_rows_in*n_columns_in, out_channels, n_frames)

        x = x.reshape(batch, n_rows_in, n_columns_in, self.out_channels, n_frames)
        x = x.permute(
            0, 4, 3, 1, 2
        )  # (batch, n_frames, out_channels, n_rows_in, n_columns_in)

        return x

    def get_name(self) -> str:
        """
        name is used to identify layer

        can be used as key names in dict to link different layers
        (e.g. spatiotemporal layer with calcium readout layer)
        user has to enforce name uniqueness and consistency
        no internal checks implemented
        """
        return self.name

        """
        expect user to supply appropriate x weights and tau
        to simulate photoreceptor gaussian spatial sampling and temporal lowpass

        save additional parameter info as specified in
        util_conv, kernel_gauss_photoreceptor method
        """


class STlayerPhotoreceptor(_SpatiotemporalLayerBase):
    """photoreceptor spatiotemporal conv layer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        x_stride: Union[int, list[int]],
        x_padding: Union[int, list[int]],
        x_num_groups: int,
        x_weight_init: Tensor,
        dt: torch.float,
        t_kernel_size: int,
        layer_name: str,
        activation_fn: Callable[[Tensor], Tensor],
        x_ds: Union[int, list[int]],
        pr_sigma,
        device: torch.device,
        tau_init: Tensor,
        **kwargs,
    ) -> None:
        """gaussian spatial sampling + temporal lowpass layer

        Args:
            x_ds (Union[int, list[int]]): input movie neighboring pixels distance (degrees)
            pr_sigma (_type_): photoreceptor acceptance angle
            tau_init (Tensor): initial temporal kernel
        """

        super(STlayerPhotoreceptor, self).__init__(
            in_channels,
            out_channels,
            x_stride,
            x_padding,
            x_num_groups,
            x_weight_init,
            dt,
            t_kernel_size,
            layer_name,
            activation_fn,
            device,
        )

        # save spatial kernel info for internal records only
        self.x_ds = x_ds
        self.pr_sigma = pr_sigma

        # gaussian spatial kernel should not be trainable
        self.x_weight.requires_grad = False

        # temporal kernel is also not trainable
        self.register_buffer("tau", tau_init)

    def _gen_t_weights(self) -> None:
        """
        updates temporal weights using single exponential kernel
        """
        self.t_weight = kernel_singleExp(self.tau, self.t_kernel_size, self.dt)


class STlayerExpTemporal(_SpatiotemporalLayerBase):
    """unconstrained spatial conv, single exponential temporal conv layer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        x_stride: Union[int, list[int]],
        x_padding: Union[int, list[int]],
        x_num_groups: int,
        x_weight_init: Tensor,
        dt: torch.float,
        t_kernel_size: int,
        layer_name: str,
        activation_fn: Callable[[Tensor], Tensor],
        tau_init: Tensor,
        bias_init: Tensor,
        device: torch.device,
        **kwargs,
    ) -> None:
        """arbitrary spatial conv, exp(-t/tau) temporal conv layer
        adds trainable tau for temporal kernel
        adds a trainable bias, bias is added per channel after temporal conv

        Args:
            tau_init (Tensor): initial temporal kernel
            bias_init (Tensor): bias initialization, usually 0
        """

        super(STlayerExpTemporal, self).__init__(
            in_channels,
            out_channels,
            x_stride,
            x_padding,
            x_num_groups,
            x_weight_init,
            dt,
            t_kernel_size,
            layer_name,
            activation_fn,
            device,
        )

        # trainable temporal conv1d parameters
        self.tau = torch.nn.Parameter(tau_init)

        # trainable bias
        self.bias = torch.nn.Parameter(bias_init)

    def _gen_t_weights(self) -> None:
        """
        updates temporal weights using single exponential kernel
        """
        self.t_weight = kernel_singleExp(
            tau=self.tau, kernel_size=self.t_kernel_size, dt=self.dt
        )


class STlayerPowerExpTemporal(_SpatiotemporalLayerBase):
    """unconstrained spatial conv, power function * exp temporal conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        x_stride: Union[int, list[int]],
        x_padding: Union[int, list[int]],
        x_num_groups: int,
        x_weight_init: Tensor,
        dt: torch.float,
        t_kernel_size: int,
        layer_name: str,
        activation_fn: Callable[[Tensor], Tensor],
        tau_init: Tensor,
        n_power_init: Tensor,
        bias_init: Tensor,
        device: torch.device,
        **kwargs,
    ) -> None:
        """_summary_
        temporal kernel has two trainable parameters,
        time constant tau and exponent n

        Args:
            tau_init (Tensor): initial temporal kernel
            n_power_init (Tensor): power function exponent
            bias_init (Tensor): bias initialization, usually 0
        """

        super(STlayerPowerExpTemporal, self).__init__(
            in_channels,
            out_channels,
            x_stride,
            x_padding,
            x_num_groups,
            x_weight_init,
            dt,
            t_kernel_size,
            layer_name,
            activation_fn,
            device,
        )

        # trainable temporal conv1d parameters
        self.tau = torch.nn.Parameter(tau_init)
        self.n_power = torch.nn.Parameter(n_power_init)

        # trainable bias
        self.bias = torch.nn.Parameter(bias_init)

    def _gen_t_weights(self) -> None:
        """
        updates temporal weights using power function * exponential kernel
        """
        self.t_weight = kernel_PowerExp(
            n_power=self.n_power,
            tau=self.tau,
            kernel_size=self.t_kernel_size,
            dt=self.dt,
        )


class Slayer(_SpatiotemporalLayerBase):
    """spatial conv only layer
    use this to bypass temporal conv so that signals
    will be instantenous in time
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        x_stride: Union[int, list[int]],
        x_padding: Union[int, list[int]],
        x_num_groups: int,
        x_weight_init: Tensor,
        dt: torch.float,
        layer_name: str,
        activation_fn: Callable[[Tensor], Tensor],
        bias_init: Tensor,
        device: torch.device,
        **kwargs,
    ) -> None:

        super(Slayer, self).__init__(
            in_channels,
            out_channels,
            x_stride,
            x_padding,
            x_num_groups,
            x_weight_init,
            dt,
            1,
            layer_name,
            activation_fn,
            device,
        )

        # trainable bias
        self.bias = torch.nn.Parameter(bias_init)

    def _spatial_conv(self, x: Tensor) -> Tensor:
        """
        2D convolution along spatial (rows, cols) dimensions
        convolves all input channels for each output channel
        uses kernel stored in self.x_weight
        trainable bias

        Args:
            x (tensor): shape (batch, n_frames, channel, rows, cols)

        Returns:
            x (tensor): shape (batch, n_frames, channel, rows, cols)
        """

        batch, n_frames, in_channels, n_rows_in, n_columns_in = x.shape

        x = x.reshape(batch * n_frames, in_channels, n_rows_in, n_columns_in)
        x = F.conv2d(
            input=x,
            weight=self.x_weight,
            bias=self.bias,
            stride=self.x_stride,
            padding=self.x_padding,
            groups=self.x_groups,
        )

        (
            _,
            _,
            n_rows_out,
            n_columns_out,
        ) = x.shape  # (batch*n_frames, out_channels, n_rows_out, n_columns_out)

        x = x.reshape(batch, n_frames, self.out_channels, n_rows_out, n_columns_out)

        return x

    def _temporal_conv(self, x: Tensor) -> None:
        pass


class STBatchNorm(torch.nn.BatchNorm3d):
    """wrapper for BatchNorm3d"""

    def forward(self, input: Tensor) -> Tensor:
        """
        custom _Spatiotemporal_Layer_Base class uses different tensor
        index order than BatchNorm3d

        method permutes tensor to match BatchNorm3d, which expects
        tensor shape(batch, channel, *)

        Args:
            input (tensor): shape (batch, n_frames, channel, rows, cols)

        Returns:
            input (tensor): shape (batch, n_frames, channel, rows, cols)
        """

        input = input.permute(0, 2, 1, 3, 4)
        input = super(STBatchNorm, self).forward(input)
        input = input.permute(0, 2, 1, 3, 4)

        return input


class _CalciumLayerBase(torch.nn.Module):
    """calcium readout layer base class"""

    def __init__(
        self,
        dt: float,
        activation_fn: Callable[[Tensor], Tensor],
        cell_types: list[str],
        cell_count: list[int],
        alpha_init: tuple[float, float],
        beta_init: tuple[float, float],
        ca_tau_init: tuple[float, float],
        t_kernel_size: int,
        device: torch.device,
        **kwargs,
    ):
        """initialize calcium readout layer parameters
        default mode is to train separate calcium parameters
        for every neuron in the dataset

        activation is applied before voltage-to-ca conversion

        Args:
            dt (float): step size in sec
            activation_fn (Callable[[Tensor], Tensor]): activation for lc_layers
            cell_types (list[str]): list of celltypes in dataset
            cell_count (list[int]): count of unique neurons for each celltype
            alpha_init (tuple[float, float]): affine transform scaling
            beta_init (tuple[float, float]): affine transform shift
            ca_tau_init (tuple[float, float]): initial calcium indicator time constant
            t_kernel_size (int): # of time points
            device (torch.device): device
        """

        super().__init__()

        self.device = device

        self.dt = dt
        self.t_kernel_size = t_kernel_size

        self.activation = activation_fn

        # shift 'voltage' output after elu activation by +1 so
        # that the output is strictly positive, >= 0
        # this is not applicable to other activation functions
        self.offset = 1.0 if self.activation.__name__ == "elu" else 0

        self.cell_types = cell_types

        # create nested pytorch dict to store trainable calcium param for each unique neuron
        # each neuron is uniquely identified by cell type name, and integer # (assigned by dataloader)
        self.ca_param = torch.nn.ModuleDict()
        for i in range(len(self.cell_types)):
            cell_name = self.cell_types[i]

            n_neurons = cell_count[
                i
            ]  # number of neurons per currently selected cell type

            # allow each neuron have separate affine transform parameter
            alpha = torch.empty(n_neurons).uniform_(alpha_init[0], alpha_init[1])
            beta = torch.empty(n_neurons).uniform_(beta_init[0], beta_init[1])

            # one tau per cell type, assuming the same ca indcator was used
            ca_tau = torch.empty(1).uniform_(ca_tau_init[0], ca_tau_init[1])

            new_pdict = torch.nn.ParameterDict()
            new_pdict["alpha"] = torch.nn.Parameter(alpha)
            new_pdict["beta"] = torch.nn.Parameter(beta)
            new_pdict["ca_tau"] = torch.nn.Parameter(ca_tau)

            self.ca_param[cell_name] = new_pdict

    def voltage_to_ca(
        self, v_trace: Tensor, alpha: Tensor, beta: Tensor, tau: Tensor
    ) -> Tensor:
        """
        converts voltage to calcium using single exponential kernel followed
        by affine transform

        ca = beta + softplus(alpha) * gcamp_filter( activation(v_trace) )

        input v_trace is activated to remove negative voltage dynamics
        expect calcium indicators (e.g. gcamp) to be highly rectified

        use softplus to make alpha > 0 for interpretable calcium output

        Args:
            v_trace (Tensor): input volage trace with shape (batch, n_frames)
            alpha (Tensor): affine transform scaling
            beta (Tensor): affine transform shift
            tau (Tensor): time constant for calcium indicator

        Returns:
            ca (Tensor): output calcium trace with shape (batch, n_frames)
        """

        v_trace = self.activation(v_trace) + self.offset

        # softplus forces alpha to be > 0
        ca = beta.unsqueeze(0) + F.softplus(alpha.unsqueeze(0)) * gcamp_filter(
            x=v_trace,
            tau=tau,
            dt=self.dt,
            t_kernel_size=self.t_kernel_size,
        )
        return ca


class LCModelBatchnorm(_CalciumLayerBase):
    """full model from input movie to output voltage and calcium"""

    def __init__(
        self,
        pr: STlayerPhotoreceptor,
        network: list[STlayerExpTemporal],
        norm_layers: list[STBatchNorm],
        lc_layers: list[STlayerExpTemporal],
        dt: float,
        activation_fn: Callable[[Tensor], Tensor],
        cell_types: list[str],
        cell_count: list[int],
        alpha_init: tuple[float, float],
        beta_init: tuple[float, float],
        ca_tau_init: tuple[float, float],
        t_kernel_size: int,
        device: torch.device,
        use_global_ca_tau: bool = False,
        **kwargs,
    ):
        """generate the full network model

        Args:
            pr (STlayerPhotoreceptor): photoreceptor layer
            network (list[STlayerExpTemporal]): hidden layers
            norm_layers (list[STBatchNorm]): normalization layers
            lc_layers (list[STlayerExpTemporal]): celltype specific readout layers
            use_global_ca_tau (bool, optional): train a single calcium indicator time constant for all neurons. Defaults to False.
        """

        super(LCModelBatchnorm, self).__init__(
            dt,
            activation_fn,
            cell_types,
            cell_count,
            alpha_init,
            beta_init,
            ca_tau_init,
            t_kernel_size,
            device,
        )

        self.pr = pr
        self.network = torch.nn.ModuleList(network)
        self.norm_layers = torch.nn.ModuleList(norm_layers)
        self.lc_layers = torch.nn.ModuleList(lc_layers)

        # option to train a single ca tau across all celltypes
        self.use_global_ca_tau = use_global_ca_tau
        self.global_ca_tau = torch.nn.Parameter(
            torch.empty(1).uniform_(ca_tau_init[0], ca_tau_init[1])
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        for each input movie, generate a voltage output for each cell type
        voltage trace is taken from one central neuron

        Args:
            x (Tensor): (batch, n_frames, channel, rows, cols)

        Returns:
            lc_center_out (dict):
                keys are cell type name from self.lc_layers.get_name()
                values are voltage (not rectified) from central neuron
        """

        x = self.pr(x)
        for i, layer in enumerate(
            self.network
        ):  # forward pass sequentially through network layers
            x = layer(x)  # activation + spatiotemporal conv
            x = self.norm_layers[i](x)  # normalization

        # save lc_output to dict, only export the voltage trace from the center neuron
        lc_center_out = {}
        for i, layer in enumerate(self.lc_layers):
            fmap = layer(
                x
            )  # shape (batch, n_frames, out_channels, n_rows_out, n_columns_out)
            lc_center_out[layer.get_name()] = fmap[
                :, :, 0, fmap.shape[3] // 2, fmap.shape[4] // 2
            ]  # shape (batch, n_frames)

        return lc_center_out

    def loss(
        self,
        lc_cntr_out: dict,
        lc_names: list,
        ca_traces: Tensor,
        neuron: Tensor,
        rec_inds: Tensor,
    ) -> tuple[dict, list]:
        """
        for supervised training to user provided movie input and calcium traces
        loss is calculated over user defined time intervals as mean-squared-error (MSE)

        inputs are implicitly linked by index #
        i.e. lc_cntr_out entry 3 corresponds to lc_names entry 3, ca_traces entry 3, etc..


        Args:
            lc_cntr_out (dict):
                dict is identical to the output from self.forward
                key lc_name, value (tensor) voltage trace with shape (batch, n_frames)
            lc_names (list):
                generated by dataloader for the current batch
                this is not a unique list, it's just the lc_names by batch #
                e.g. ['lc18', 'lc18', 'lc4', 'lc12', 'lc4']
            ca_traces (Tensor):
                shape (batch, n_frames)
                ground truth calcium data from dataloader
            neuron (Tensor):
                shape (batch, int)
                neuron id associated with each unique neuron within
                a cell type (integer, range 0 to (num_uniq_neurons-1))
                generated by the dataloader
            rec_inds (Tensor):
                shape (batch, (start time, stop time))
                user defined time intervals to calculate loss
                note that interval is not fixed, but depends on movie input
                generated by dataloader

        Returns:
            avg_loss (dict):
                keys cell type names, values (Tensor) average loss
                mean-squared-error (MSE) loss is normalized by neuorn count
                e.g. if cell type A has N ground truth traces in this minibatch
                MSE loss for each trace is weighted by 1/N
            list_ca_pred (list):
                inferred calcium outputs, ordered by input
        """

        # for each minibatch, there can only be recording from one lc_type at a time given how the dataloader current works
        avg_loss = (
            {}
        )  # store loss tensor for every cell type in dict, whether there was any training examples for that cell type or not doesn't matter, just initialize as zero
        data_count = (
            {}
        )  # store how many ground truth recordings for each lc type, used to normalize loss by neuron count

        for name in lc_names:
            if name in data_count:
                data_count[name] += 1
            else:  # key does not exist in dict yet
                avg_loss[name] = torch.zeros(
                    1, device=self.device
                )  # initialize avg loss as zeros
                data_count[name] = 1  # count=1, this is the first data

        list_ca_pred = []  # save inferred ca also

        for i in range(len(lc_names)):  # iterate over individual batch
            # dataloader can only provide one real recording per data batch
            # the LC_model can generate predictions for all lc cell types in every data batch
            # need to pull out the right cell type, and the right batch

            true_lc_name = lc_names[
                i
            ]  # ground truth data came from this cell type, for the current data batch
            ca_true = ca_traces[
                i, :
            ]  # ground truth calcium data, for the current data batch. ca_true has shape (n_frames)

            # lc_cntr_out[true_lc_name] contains predicted voltage in the form of shape (batch, n_frames)
            v_pred = lc_cntr_out[true_lc_name][
                i, :
            ]  # grab only the current batch, shape (1, n_frames)

            neuron_num = neuron[
                i
            ]  # (int) each unique neuron within each lc cell type is assigned an integer value, ranging from 0 to (total num neurons-1)

            cur_ca_tau = (
                self.global_ca_tau
                if self.use_global_ca_tau
                else self.ca_param[true_lc_name]["ca_tau"]
            )

            ca_pred = self.voltage_to_ca(
                v_pred,
                self.ca_param[true_lc_name]["beta"][neuron_num],
                self.ca_param[true_lc_name]["alpha"][neuron_num],
                cur_ca_tau,
            )

            ca_pred.squeeze_()  # ca_pred should have shape (batch=1, n_frames), collapse it into shape (n_frames)

            # measure mean-squared-error loss
            # trim ca_pred or ca_true in time to the user specified range
            # dataloader currently trims from t=-1 to end of original trace (non-padded), where vstim movie starts at t=0
            ia = rec_inds[i, 0]  # start index for calculating loss
            ib = rec_inds[i, 1]  # stop index for calculating loss
            loss = torch.mean(
                (ca_true[ia:ib] - ca_pred[ia:ib]) ** 2
            )  # mean-squared-error loss

            avg_loss[true_lc_name] = avg_loss[true_lc_name] + torch.true_divide(
                loss, data_count[true_lc_name]
            )  # add loss to total average loss, with 1/neuron_count weight

            # save inferred calcium
            list_ca_pred.append(ca_pred)

        return avg_loss, list_ca_pred

    def calcium_inference(self, x: Tensor, sel_lc: str) -> list[Tensor]:
        """
        generate inferred calcium output for the given movie input and
        user specified cell type

        Args:
            x (tensor):
                batch of movie inputs
                shape (batch, n_frames, channel, rows, cols)
            sel_lc (string):
                keyname of the cell type to run the calcium inference on

        Returns:
            list_ca_pred (list of Tensor):
                predicated calcium dynamics
                shape (batch, n_frames)
        """

        list_ca_pred = []

        with torch.no_grad():
            lc_center_out = self.forward(x)

            # affine transform param were trained for each unique neuron
            # but predict average neuron response by averaging over affine param
            avg_beta = torch.mean(self.ca_param[sel_lc]["beta"])
            avg_alpha = torch.mean(self.ca_param[sel_lc]["alpha"])

            num_minibatch = x.size()[0]

            for i in range(num_minibatch):
                # compute calcium predicted for each minibatch for the selected lc

                v_pred = lc_center_out[sel_lc][
                    i, :
                ]  # grab only the current batch, shape (1, n_frames)

                ca_pred = self.voltage_to_ca(
                    v_pred, avg_beta, avg_alpha, self.ca_param[sel_lc]["ca_tau"]
                )

                list_ca_pred.append(ca_pred.squeeze_())

        return list_ca_pred

    def generate_feature_maps(self, x: Tensor) -> dict[str, Tensor]:
        """
        generate voltage dynamics (pre-activation) from each layer in the network

        Args:
            x (tensor):
                batch of movie inputs
                shape (batch, n_frames, channel, rows, cols)

        Returns:
            fmaps (dict):
                voltage dynamics from all intermediate feature maps
        """

        fmaps = {}

        with torch.no_grad():
            x = self.pr(x)
            fmaps["pr"] = x

            for i, layer in enumerate(
                self.network
            ):  # forward pass sequentially throught network layers
                x = layer(x)  # activation + spatiotemporal conv
                fmaps[layer.get_name()] = x

                x = self.norm_layers[i](x)  # normalization
                fmaps[f"norm_layer_{i}"] = x

            for i, layer in enumerate(
                self.lc_layers
            ):  # cell type specific read out layers
                fmaps[layer.get_name()] = layer(x)

        return fmaps
