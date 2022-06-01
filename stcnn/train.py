"""Train model

"""

# standard library imports
import os, logging, sys, pickle, json, time, argparse
from pathlib import Path
from datetime import datetime

sys.path.append("..")

# third party imports
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np

# local application imports
from stcnn.ca_dataloader import LCDataLoader, load_ca_data, load_vstim_mov
from stcnn.model_factory import model_stcnn_factory

log = logging.getLogger(__name__)  # logger for this file


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    avg_loss_hist: dict[str, Tensor],
    print_freq: int,
    clip_max_norm: float,
    device: torch.device,
    log: logging.Logger,
    clip_param: dict,
) -> None:
    """train model for one epoch
    losses are calculated separately for each cell type
    losses are normalized by number of cell type data instances within
    each mini batch.

    Args:
        model (torch.nn.Module): model
        train_loader (torch.utils.data.DataLoader): training data loader
        optimizer (torch.optim.Optimizer): training optimizer
        avg_loss_hist (dict[str, Tensor]): stores minibatch avg_loss for each cell type
        print_freq (int): print every # of iterations
        clip_max_norm (float): clip gradient if not None
        device (torch.device): cpu or gpu
        log (logging.Logger): console logging
        clip_param (dict): patterns for clipping param values
    """

    model.train()  # set model to be in training mode, need this for batchnorm

    for i, data_batch in enumerate(train_loader):
        v_pred = model(data_batch["movie"])

        avg_loss, list_ca_pred = model.loss(
            lc_cntr_out=v_pred,
            lc_names=data_batch["lc_type"],
            ca_traces=data_batch["ca_trace"],
            rec_inds=data_batch["recording_indices"],
            neuron=data_batch["neuron_num"],
        )

        # sum up avg_losses across all cell types
        loss = torch.zeros(1, device=device)
        for k in avg_loss.keys():  # iter over cell types
            if avg_loss[k]:  # value is not zero
                loss += avg_loss[k]
                avg_loss_hist[k].append(avg_loss[k].item())

        # zeros gradient and backprop to accumulate gradient
        optimizer.zero_grad()
        loss.backward()

        if clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=clip_max_norm
            )

        optimizer.step()

        # print losses
        if (i % print_freq == 0) or (i == len(train_loader) - 1):
            s = f"iter: {i}/{len(train_loader)} "

            for k in avg_loss_hist.keys():
                if avg_loss_hist[k]:  # if it's not empty
                    s = s + f"{k}: {avg_loss_hist[k][-1]:.2e}, "

            log.info(s)

        # clip parameters
        clip_param_value(model, clip_param)


def clip_param_value(model: torch.nn.Module, clip_param: dict) -> None:
    """clip trainable parameter values using user specified config

    Args:
        model (torch.nn.Module): model
        clip_param (dict): dict of patterns for matching param name and clip values
                            assumes each param can only match one pattern
    """

    with torch.no_grad():
        for name, param in model.named_parameters():
            for ptn in clip_param.values():  # iter param clip patterns
                if ptn["name"] in name:
                    param.data.clamp_(min=ptn["min"], max=ptn["max"])
                    break  # assumes param name can only match to one user specified pattern


def save_model_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, out_path: str
) -> None:
    """save model and optimizer state_dict

    Args:
        model (torch.nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        out_path (str): complete file path + filename
    """

    torch.save(
        {"model": model.state_dict(), "optim": optimizer.state_dict()},
        out_path,
    )


def count_celltype_instances(
    dataset: torch.utils.data.Dataset, ca_traces: dict
) -> dict[str, int]:
    """count # of celltype instances for one data batch

    Args:
        dataset (torch.utils.data.Dataset): assumes map-style dataset
        ca_traces (dict): used to generate celltype names

    Returns:
        data_count dict[str,int]: [celltype name, count]
    """

    data_count = {}
    for celltype in ca_traces.keys():
        data_count[celltype] = 0  # initialize count = 0 for each cell type

    # count # instances for each cell type by iterating through the whole dataset
    for i in range(len(dataset)):
        celltype = dataset[i]["lc_type"]
        data_count[celltype] += 1

    return data_count


@hydra.main(config_path="conf", config_name="config")
def main(
    cfg: DictConfig,
) -> None:

    # --- setup experiment dir ---
    now = datetime.now()
    expt_name = now.strftime("%Y%m%d_%H%M%S_") + cfg["expt_name"]
    expt_path = Path(cfg["expt_path"]) / expt_name

    if not expt_path.exists():
        expt_path.mkdir()

    log.info("Experiment saved at: %s", expt_path)

    # --- setup torch ---
    #    seed = 0  # deterministic seed
    #    torch.manual_seed(seed)

    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    #       torch.cuda.manual_seed(seed)
    else:
        my_device = torch.device("cpu")

    log.info("Device: %s", my_device)

    torch.autograd.set_detect_anomaly(True)  # useful for debugging

    # --- setup dataloader ---
    # load onto CPU to save space, dataloader will handle GPU transfer
    ca_traces, num_t_pts = load_ca_data(
        input_path=cfg["dataset"]["ca_path"], my_device=torch.device("cpu")
    )
    vstim_mov = load_vstim_mov(
        input_path=cfg["dataset"]["mov_path"], my_device=torch.device("cpu")
    )
    train_dataset = LCDataLoader(
        vstim_mov,
        ca_traces,
        num_t_pts,
        avg_xTrials=cfg["dataset"]["avg_xTrials"],
        my_device=my_device,
        px_res=cfg["dataset"]["mov_res"],
        mov_bg_int=cfg["dataset"]["mov_bg_int"],
        opts_meas_t_start=cfg["dataset"]["loss_meas_t_start"],
        num_t_pts_prepad=cfg["dataset"]["num_t_pts_prepad"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=cfg["shuffle_dataset"],
        drop_last=False,
    )

    data_count = count_celltype_instances(train_dataset, ca_traces)

    # --- collate celltype specific config ---
    pr_config = cfg["model"]["pr"]
    network_config = cfg["model"]["network"]

    L_config = cfg["model"]["lobula"]
    ca_config = cfg["model"]["ca"]
    for lc in ca_traces.keys():
        L_config["lc_types"].append(lc)
        L_config["lc_count"].append(train_dataset.get_num_of_neurons(lc))
        L_config["x_kernel_size"].append(cfg["model"][lc]["x_kernel_size"])
        L_config["x_stride"].append(cfg["model"][lc]["x_stride"])
        L_config["x_padding"].append(cfg["model"][lc]["x_padding"])

        ca_config["cell_types"].append(lc)
        ca_config["cell_count"].append(train_dataset.get_num_of_neurons(lc))

    L_config["in_channels"] = [network_config["out_channels"][-1]]

    # --- save config ---
    my_config = {
        "experiment_path": str(expt_path),
        "pr_config": OmegaConf.to_container(pr_config),
        "network_config": OmegaConf.to_container(network_config),
        "L_config": OmegaConf.to_container(L_config),
        "ca_config": OmegaConf.to_container(ca_config),
        "data_count": data_count,
        "sampling_rate": cfg["dataset"]["sampling_rate"],
        "dataset_config": OmegaConf.to_container(cfg["dataset"]),
    }
    with open(expt_path / "config.json", "w") as f:
        json.dump(my_config, f)

    # --- setup model ---
    model = model_stcnn_factory(
        pr_config=OmegaConf.to_container(pr_config),
        network_config=OmegaConf.to_container(network_config),
        L_config=OmegaConf.to_container(L_config),
        ca_config=OmegaConf.to_container(ca_config),
        sampling_rate=cfg["dataset"]["sampling_rate"],
        device=my_device,
    )
    model.to(my_device)

    # --- setup optimizer ---
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # --- training loop ---
    avg_loss_hist = {}  # a dict of avg_loss by cell type
    for lc in ca_traces.keys():
        avg_loss_hist[lc] = []

    for epoch in range(cfg["n_epochs"]):
        fname = "model_chkpnt_" + str(epoch) + ".pt"
        save_model_state(model=model, optimizer=optim, out_path=str(expt_path / fname))

        logging.info("Epoch {}/{}".format(epoch + 1, cfg["n_epochs"]))

        train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optim,
            avg_loss_hist=avg_loss_hist,
            print_freq=cfg["print_freq"],
            clip_max_norm=cfg["clip_max_norm"],
            device=my_device,
            log=log,
            clip_param=OmegaConf.to_container(cfg["clip_param"]),
        )

        fname = expt_path / "avg_loss_hist.pkl"
        with open(fname, "wb") as f:
            pickle.dump(avg_loss_hist, f)

    fname = "model_chkpnt_" + str(cfg["n_epochs"]) + ".pt"
    save_model_state(model=model, optimizer=optim, out_path=str(expt_path / fname))

    log.info("Completed training model")


if __name__ == "__main__":
    main()
