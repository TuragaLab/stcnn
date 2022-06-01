"""Functions to plot model outputs and model training

"""

import torch
from torch.utils.data import Dataset
from torch import Tensor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera
from pathlib import Path


def get_data_by_vstim(
    dataset: Dataset,
    vstim_list: list[int],
    lc_type: str,
    uid: int = None,
) -> tuple[dict, int]:
    """given a map-style dataset, return calcium traces matching user
    specified criteria. this function does not return movie in order to
    save space.

    Args:
        dataset (Dataset): map-style dataset
        vstim_list (list[int]): visual stimulus barcode #
        lc_type (str): cell type
        uid (int, optional): neuron unique ID #. Defaults to None.

    Returns:
        data (dict): key=vstim, value=dict of calcium traces by neuron recording
        n_frames (int): length of movie
    """
    # return calcium data only, don't save movie, that's way too much space

    n_frames = -1

    # key=vstim barcode, value=dataset indices matching celltype and barcode
    vstim2idx = {}

    for k in vstim_list:
        vstim2idx[k] = dataset.get_idx_by_criteria(
            lc_type=lc_type, vstim="vstim_" + str(k), uid=uid
        )

    data = {}  # key=vstim barcode, value=dict

    for vstim in vstim2idx.keys():
        data[vstim] = {}  # key=vstim, value=dict

        for idx in vstim2idx[vstim]:
            tmp = dataset[idx]

            data[vstim][idx] = {}
            data[vstim][idx]["ca_trace"] = (
                tmp["ca_trace"].squeeze().cpu().detach().numpy()
            )

            n_frames = max(n_frames, tmp["movie"].shape[0])

    return data, n_frames


def get_one_vstim_mov(dataset: Dataset, vstim: int) -> Tensor:
    """grab a single movie based on barcode #

    Args:
        dataset (Dataset): map-style dataset
        vstim (int): a single visual stimulus barcode #

    Returns:
        mov (Tensor): shape (n_frames, channel, row, col)
    """
    idx_list = dataset.get_idx_by_criteria(
        lc_type="any", vstim="vstim_" + str(vstim), uid=None
    )

    mov = dataset[idx_list[0]]["movie"]

    return mov


def animate_vstim_ca(
    dataset: Dataset,
    vstim_list: list[int],
    lc_type: str,
    out_fname: str,
    fig_width: float = 20,
    fig_height: float = 5,
    uid: str = None,
    dilute_factor: float = 6,
    fps: int = 180,
    ax_y_min: float = -0.5,
    ax_y_max: float = 2,
    val_min: float = -0.5,
    val_max: float = 0.5,
) -> None:
    """animate the provided vstim_list in a row
    visual stimulus movie at top, and groundtruth calcium traces at bottom

    Args:
        dataset (Dataset): map-style dataset
        vstim_list (list[int]): visual stimulus barcode #
        lc_type (str): cell type
        out_fname (str): full path of output file name
        fig_width (float, optional): Defaults to 20.
        fig_height (float, optional): Defaults to 5.
        uid (str, optional): neuron unique ID #. Defaults to None.
        dilute_factor (int, optional): downsampling factor. Defaults to 6.
        fps (int, optional): frames per sec. Defaults to 180.
        ax_y_min (float, optional): y-axis min. Defaults to -0.5.
        ax_y_max (float, optional): y-axis max. Defaults to 2.
        val_min (float, optional): movie intensity min val. Defaults to -0.5.
        val_max (float, optional): movie intensity max val. Defaults to 0.5.
    """

    effective_fps = fps / dilute_factor

    data2plot, n_frames = get_data_by_vstim(
        dataset, vstim_list, lc_type=lc_type, uid=uid
    )

    # plot frame by frame for all movies/traces
    fig, ax = plt.subplots(
        2, len(vstim_list), squeeze=False
    )  # num rows, num cols in figure, squeeze=False allows you to index by row/col when specifying which subplot
    fig.set_size_inches(fig_width, fig_height)  # figure width, height
    fig.tight_layout()

    camera = Camera(fig)

    # set titles
    col_idx = 0
    for p in data2plot.keys():
        ax[0, col_idx].title.set_text(p)
        col_idx += 1

    # now plot frame by frame
    fnum_list = []
    t_axis = []
    dt = 1 / fps  # step in seconds

    for fnum in np.arange(
        0, n_frames, dilute_factor
    ):  # for each diluted time point or frame number
        # build indices for plotting calcium traces
        fnum_list.append(fnum)
        t_axis.append(dt * fnum)

        col_idx = 0
        for v in data2plot.keys():  # for each vstim barcode
            tmp_mov = get_one_vstim_mov(dataset, v)
            cur_img = tmp_mov[fnum, 0, :, :].squeeze().cpu().detach().numpy()

            # first row of plot are vstim movies
            ax[0, col_idx].imshow(cur_img, cmap="gray", vmin=val_min, vmax=val_max)

            # second row of plot are calcium traces
            for k in data2plot[v].keys():
                ca_val = np.take(data2plot[v][k]["ca_trace"], fnum_list)
                ax[1, col_idx].plot(t_axis, ca_val)

            ax[1, col_idx].set_ylim(ax_y_min, ax_y_max)  # same range for ca fluor
            ax[1, col_idx].set_prop_cycle(
                None
            )  # reset color cycle so that colors will be stable for each neuron trace

            col_idx += 1

        camera.snap()

    # render and save movie
    animation = camera.animate(interval=1 / effective_fps * 1e3, blit=True)
    animation.save(str(out_fname), fps=effective_fps)


def animate_grid_vstim_ca(
    dataset: Dataset,
    vstim_list: list[int],
    lc_type: str,
    out_fname: str,
    n_rows: int = 7,
    n_cols: int = 9,
    fig_width: float = 22.5,
    fig_height: float = 35,
    uid: str = None,
    dilute_factor: float = 6,
    fps: int = 180,
    ax_y_min: float = -0.5,
    ax_y_max: float = 2,
    val_min: float = -0.5,
    val_max: float = 0.5,
) -> None:
    """animate the provided vstim_list in a grid
    movies are in the top half, calcium traces in the bottom half

    Args:
        dataset (Dataset): map-style dataset
        vstim_list (list[int]): visual stimulus barcode #
        lc_type (str): cell type
        out_fname (str): full path of output file name
        n_rows (int, optional): # of rows in plot. Defaults to 7.
        n_cols (int, optional): # of cols in plot. Defaults to 9.
        fig_width (float, optional): Defaults to 20.
        fig_height (float, optional): Defaults to 5.
        uid (str, optional): neuron unique ID #. Defaults to None.
        dilute_factor (int, optional): downsampling factor. Defaults to 6.
        fps (int, optional): frames per sec. Defaults to 180.
        ax_y_min (float, optional): y-axis min. Defaults to -0.5.
        ax_y_max (float, optional): y-axis max. Defaults to 2.
        val_min (float, optional): movie intensity min val. Defaults to -0.5.
        val_max (float, optional): movie intensity max val. Defaults to 0.5.
    """

    effective_fps = fps / dilute_factor

    data2plot, n_frames = get_data_by_vstim(
        dataset, vstim_list, lc_type=lc_type, uid=uid
    )

    # plot frame by frame for all movies/traces
    fig, ax = plt.subplots(
        2 * n_rows, n_cols, squeeze=False
    )  # num rows, num cols in figure, squeeze=False allows you to index by row/col when specifying which subplot
    fig.set_size_inches(fig_width, fig_height)  # figure width, height
    fig.tight_layout()

    camera = Camera(fig)

    # now plot frame by frame
    fnum_list = []
    t_axis = []
    dt = 1 / fps  # step in seconds

    mov_r_idx_offset = 0  # plot movie together at top
    ca_r_idx_offset = n_rows  # plot traces together at bottom

    for fnum in np.arange(
        0, n_frames, dilute_factor
    ):  # for each diluted time point or frame number
        # build indices for plotting calcium traces
        fnum_list.append(fnum)
        t_axis.append(dt * fnum)

        data_iter = iter(data2plot)  # assumes in appropriate order already

        for r in np.arange(n_rows):
            for c in np.arange(n_cols):

                v = next(data_iter)
                tmp_idx = next(iter(data2plot[v]))
                ca_val = np.take(data2plot[v][tmp_idx]["ca_trace"], fnum_list)

                tmp_mov = get_one_vstim_mov(dataset, v)
                cur_img = tmp_mov[fnum, 0, :, :].squeeze().cpu().detach().numpy()

                # top plots are vstim movies
                ax[r + mov_r_idx_offset, c].imshow(
                    cur_img, cmap="gray", vmin=val_min, vmax=val_max
                )
                ax[r + ca_r_idx_offset, c].plot(t_axis, ca_val)

                ax[r + ca_r_idx_offset, c].set_ylim(
                    ax_y_min, ax_y_max
                )  # same range for ca fluor
                ax[r + ca_r_idx_offset, c].set_prop_cycle(
                    None
                )  # reset color cycle so that colors will be stable for each neuron trace
        # done plotting movie and traces for current frame, save plot to camera
        camera.snap()

    # render and save movie
    animation = camera.animate(interval=1 / effective_fps * 1e3, blit=True)
    animation.save(out_fname, fps=effective_fps)


def plot_spatial_kernels(
    layers: torch.nn.ModuleList,
    out_path: Path,
    fname_prefix: str,
    fig_scale: float = 2.5,
) -> None:
    """plot the spatial x_weights in a grid for every convolution layer

    Args:
        layers (torch.nn.ModuleList): list containing convolution layer
        out_path (Path): output directory
        fname_prefix (str): output file prefix
        fig_scale (float, optional): Defaults to 2.5.
    """

    for k in layers:
        cur_kernel = k.x_weight

        (
            num_out_ch,
            num_in_ch,
            _,
            _,
        ) = cur_kernel.shape  # (out_channel, in_channel, row, col)

        fig, ax = plt.subplots(num_out_ch, num_in_ch, squeeze=False)
        fig.set_size_inches(
            num_in_ch * fig_scale, num_out_ch * fig_scale
        )  # figure width, height
        fig.tight_layout()

        for r in range(num_out_ch):
            for c in range(num_in_ch):
                img = cur_kernel[r, c, :, :].cpu().detach().numpy()

                ax[r, c].imshow(img, cmap="gray")

                ax[r, c].set_title(k.get_name() + "_" + str(r) + "_" + str(c))
                ax[r, c].axis("off")

        out_fname = out_path / (fname_prefix + "_" + k.get_name() + ".png")
        plt.savefig(str(out_fname))


def plot_weight_hist(
    model: torch.nn.Module,
    ptn_match: list[str],
    out_path: Path,
    fname_prefix: str,
    num_bins: int = 20,
    fig_scale: float = 2.5,
) -> None:
    """aggregate and plot parameters as histograms
    with each string pattern in ptn_match list as its own histogram

    Args:
        model (torch.nn.Module): network model
        ptn_match (list[str]): string pattern to aggregate parameter names
        out_path (Path): output directory
        fname_prefix (str): output file prefix
        num_bins (int, optional): Defaults to 20.
        fig_scale (float, optional): Defaults to 2.5.
    """
    num_col = len(ptn_match)

    fig, ax = plt.subplots(1, num_col, squeeze=False)
    fig.set_size_inches(num_col * fig_scale * 2, fig_scale)  # figure width, height
    fig.tight_layout()

    plot_idx = 0
    for ptn in ptn_match:
        values = np.zeros(1)

        for name, param in model.named_parameters():
            if ptn in name:
                tmp_val = torch.flatten(param).cpu().detach().numpy()

                values = np.append(values, tmp_val, axis=0)

        ax[0, plot_idx].hist(values[1:], bins=num_bins)
        ax[0, plot_idx].set_title(ptn)
        plot_idx += 1

    out_fname = out_path / (fname_prefix + "_parameter_hist.png")
    plt.savefig(str(out_fname))


def plot_loss_history(
    loss_hist: dict,
    num_epochs: int,
    out_path: Path,
    fname_prefix: str,
    rel_window_size: int = 4,
    fig_width: float = 12,
    fig_height: float = 8,
) -> None:
    """plot losses by cell type
    each cell type can have different # of data samples
    this code assumes all cell types were trained for
    the same # of epochs


    Args:
        loss_hist (dict): key=celltype, value=loss list
        num_epochs (int): number of epochs
        out_path (Path): output directory
        fname_prefix (str): output file prefix
        rel_window_size (int, optional): 1=an epoch, 4=1/4 of epoch. Defaults to 4.
        fig_width (float, optional): Defaults to 12.
        fig_height (float, optional): Defaults to 8.
    """
    fig, ax = plt.subplots(2, 1, squeeze=False)
    fig.set_size_inches(fig_width, fig_height)  # figure width, height
    fig.tight_layout()

    for lc in loss_hist.keys():
        loss_val = loss_hist[lc]
        iter_per_epoch = len(loss_val) // num_epochs

        epoch_axis = np.arange(0, len(loss_val)) / iter_per_epoch

        ax[0, 0].plot(epoch_axis, loss_val, label=lc)

        ax[0, 0].set_yscale("log")
        ax[0, 0].legend()

        # compute rolling average using window size defined as a fraction of epoch
        window_size = iter_per_epoch // rel_window_size
        loss_val_avg = np.convolve(
            loss_val, np.ones(window_size) / window_size, mode="valid"
        )

        ax[1, 0].plot(epoch_axis[window_size - 1 :], loss_val_avg, label=lc)

        ax[1, 0].set_xlabel("epoch #")
        ax[1, 0].set_yscale("log")
        ax[1, 0].legend()

    out_fname = out_path / (fname_prefix + "_loss_hist.png")
    plt.savefig(str(out_fname))


def plot_traces_in_grid(
    dataset: Dataset,
    model: torch.nn.Module,
    vstim_list: list[int],
    lc_type: str,
    out_path: Path,
    suptitle: str = None,
    uid: str = None,
    fps: int = 180,
    num_rows: int = 7,
    num_cols: int = 7,
    fig_scale: float = 1.5,
    x_left: float = 2,
    x_right: float = 8,
    y_bottom: float = -0.5,
    y_top: float = 1.5,
    rev_row: bool = True,
    rev_col: bool = False,
    hide_axis: bool = False,
    show_title: bool = False,
    ca_true_color: str = "#ffcfa1",
) -> None:
    """plot both calcium inference and ground truth (if available) traces in a grid

    Args:
        dataset (Dataset): map-style dataset
        model (torch.nn.Module): model
        vstim_list (list[int]): visual stimulus barcode #
        lc_type (str): cell type
        out_path (Path): output directory
        suptitle (str, optional): super title, above all subplots. Defaults to None.
        uid (str, optional): neuron unique id #. Defaults to None.
        fps (int, optional): frames per sec. Defaults to 180.
        num_rows (int, optional): grid plot # rows. Defaults to 7.
        num_cols (int, optional): grid plot # cols. Defaults to 7.
        fig_scale (float, optional): Defaults to 1.5.
        x_left (float, optional): x-axis limit. Defaults to 2.
        x_right (float, optional): x-axis limit. Defaults to 8.
        y_bottom (float, optional): y-axis limit. Defaults to -0.5.
        y_top (float, optional): y-axis limit. Defaults to 1.5.
        rev_row (bool, optional): flip row order. Defaults to True.
        rev_col (bool, optional): flip col order. Defaults to False.
        hide_axis (bool, optional): Defaults to False.
        show_title (bool, optional): title above each subplot. Defaults to False.
        ca_true_color (str, optional): Defaults to "#ffcfa1".
    """

    data2plot, _ = get_data_by_vstim(dataset, vstim_list, lc_type=lc_type, uid=uid)

    tmp_mov = get_one_vstim_mov(
        dataset, vstim_list[0]
    )  # shape (n_frames, channel, row, col)
    n_frames = tmp_mov.shape[0]

    fig, ax = plt.subplots(num_rows, num_cols, squeeze=False)
    suptitle_y_space = 1 if suptitle else 0
    fig.set_size_inches(
        num_cols * fig_scale, num_rows * fig_scale + suptitle_y_space
    )  # figure width, height

    r_idx_ar = np.flip(np.arange(num_rows)) if rev_row else np.arange(num_rows)
    c_idx_ar = np.flip(np.arange(num_cols)) if rev_col else np.arange(num_cols)

    dt = 1 / fps  # seconds
    t_axis = np.arange(n_frames) * dt

    idx = 0  # index through each barcode in vstim_list

    model.eval()

    for r in r_idx_ar:
        for c in c_idx_ar:
            v = vstim_list[idx]  # keyname=barcode #

            # plot groundtruth ca data
            for k in data2plot[v].keys():  # each groundtruth recording
                ax[r, c].plot(t_axis, data2plot[v][k]["ca_trace"], color=ca_true_color)

            # generate predicted calcium based on movie selection
            mov = get_one_vstim_mov(dataset, vstim_list[idx])

            ca_pred_list = model.calcium_inference(mov.unsqueeze_(0), lc_type)
            ca_pred = ca_pred_list[0].squeeze().cpu().detach().numpy()

            ax[r, c].plot(t_axis, ca_pred, linewidth=2)

            ax[r, c].set_xlim(left=x_left, right=x_right)
            ax[r, c].set_ylim(bottom=y_bottom, top=y_top)

            if hide_axis:
                ax[r, c].axis("off")

            if show_title:
                ax[r, c].set_title(v)

            idx += 1

    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()

    out_fname = out_path / (suptitle + ".png")
    plt.savefig(str(out_fname))


def plot_feature_maps(
    fmaps: Tensor, out_fname: str, fps: int = 180, dilute_factor: float = 6
) -> plt.figure:
    """plot feature map responses (to visual stimulus) as movies

    Args:
        fmaps (Tensor): layer response to visual stimulus
        out_fname (str): full path for output file
        fps (int, optional): frames per sec. Defaults to 180.
        dilute_factor (float, optional): downsampling factor. Defaults to 6.

    Returns:
        plt.figure: figure handle
    """

    _, n_frames, n_channels, _, _ = fmaps.size()

    # plot frame by frame for all movies/traces
    fig, ax = plt.subplots(
        1, n_channels, squeeze=False
    )  # num rows, num cols in figure, squeeze=False allows you to index by row/col when specifying which subplot
    fig.set_size_inches(20 / 7 * n_channels, 5)  # figure width, height
    fig.tight_layout()

    camera = Camera(fig)

    # set titles
    for i in np.arange(n_channels):
        ax[0, i].title.set_text("channel " + str(i))

    # dump tensor into a list of numpy array movie for each channel
    fmap_list = []
    for i in range(n_channels):
        fmap_list.append(fmaps[0, :, i, :, :].squeeze().cpu().detach().numpy())

    # now plot frame by frame
    list_fnum = []
    t_axis = []
    dt = 1 / fps  # step in seconds

    for fnum in np.arange(
        0, n_frames, dilute_factor
    ):  # for each diluted time point or frame number
        # build indices for plotting calcium traces
        list_fnum.append(fnum)
        t_axis.append(dt * fnum)

        for i in range(n_channels):  # for each movie file
            cur_img = fmap_list[i][fnum, :, :]

            # first row of plot are vstim movies
            ax[0, i].imshow(
                cur_img, cmap="gray", vmin=fmap_list[i].min(), vmax=fmap_list[i].max()
            )

        camera.snap()

    # render and save movie
    animation = camera.animate(interval=1 / fps * dilute_factor * 1e3, blit=True)
    animation.save(out_fname, fps=fps / dilute_factor)

    return fig
