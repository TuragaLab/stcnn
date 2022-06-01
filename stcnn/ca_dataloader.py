"""Dataloader for calcium traces and visual stimulus movies

"""

import torch
from torch import Tensor
import os


def load_ca_data(
    input_path: str,
    my_dtype: torch.dtype = torch.float,
    my_device: torch.device = torch.device("cpu"),
) -> tuple[dict, int]:
    """
    load calcium imaging traces from user defined input_path
    expect input_path to contain one dir for each cell type,
    and arbitrary # of pt files within each cell type dir

    each visual stimulus movie should have a unique vstim_barcode
    each recorded neuron must have a unique UID

    lc_dir_name must be consistent with convnet readout layer naming, since it
    will be used to match cell type to layers

    Note: dataloader implicitly assumes ca traces are formatted identically
    with t=[-2 sec, variable endtime), and stimulus always start at t=0
    and that all traces have the same sample rate (180Hz)

    Args:
        input_path (str): Path to data files.
        device (device)

    Returns:
        ca_traces (dict):
            ca_traces[lc_dir_name][pt_filename][vstim_barcode][t_axis]
            ca_traces[lc_dir_name][pt_filename][vstim_barcode][Hz]
            ca_traces[lc_dir_name][pt_filename][vstim_barcode][neuron][uid][traces]
            ca_traces[lc_dir_name][pt_filename][vstim_barcode][neuron][uid][azi_shift]
            ca_traces[lc_dir_name][pt_filename][vstim_barcode][neuron][uid][ele_shift]

        max_num_t_pts (int):
            number of data points in the longest calcium trace
            this will be used to pad all traces to have the same length later on
            in the dataloader
    """
    ca_traces = {}

    # import and save each ca data .pt file as it's own dict
    with os.scandir(input_path) as it:
        for lc_type in it:
            if lc_type.is_dir() and not lc_type.name.startswith("."):
                ca_traces[
                    lc_type.name.lower()
                ] = (
                    {}
                )  # initialize empty dict here, will only traverse each lc_type folder once

                with os.scandir(lc_type.path) as iit:
                    for ca_data_file in iit:
                        if (
                            ca_data_file.is_file()
                            and not ca_data_file.name.startswith(".")
                            and ca_data_file.name.endswith(".pt")
                        ):
                            # print( ca_data_file.path ) # this should be the full filepath
                            ca_traces[lc_type.name.lower()][
                                ca_data_file.name
                            ] = torch.load(
                                ca_data_file.path, map_location=torch.device("cpu")
                            )  # load dict onto cpu, manually move tensors to gpu if avail later

    # figure out the max # of time points across all lc_type
    max_num_t_pts = -1
    for lc_type in ca_traces.keys():
        for data_file in ca_traces[lc_type].keys():
            for vstim in ca_traces[lc_type][data_file].keys():
                #    print(vstim)
                #    print(len(ca_traces[lc_type][vstim]['t_axis']))

                num_time_pts = len(ca_traces[lc_type][data_file][vstim]["t_axis"])
                if num_time_pts > max_num_t_pts:
                    max_num_t_pts = num_time_pts

                # copy traces to user specified device
                for uid in ca_traces[lc_type][data_file][vstim]["neuron"].keys():
                    ca_traces[lc_type][data_file][vstim]["neuron"][uid]["traces"].to(
                        device=my_device, dtype=my_dtype
                    )

    return ca_traces, max_num_t_pts


def load_vstim_mov(
    input_path: str,
    file_prefix: str = "vstim_",
    my_dtype: torch.dtype = torch.float,
    my_device: torch.device = torch.device("cpu"),
    mov_key: str = "rect_mov",
) -> dict:
    """
    load visual stimulus movies from input_path with file_prefix in name


    expect one movie per pytorch tensor .pt file
    expect movie pixel values (float) to range from 0 (black) to 1 (white)

    Args:
        input_path (str): Path to movie files
        device (device)
        my_dtype: expect pixel values to be in float
        mov_key: should be either 'rect_mov' or 'hex_mov' depending on
            whether user sampled movie on a rectangular or hexagonal lattice

    Returns:
        vstim_mov (dict):
            keynames are vstim_barcode, should match ca_traces dict in
            load_ca_data method
    """
    vstim_mov = {}

    # import and save each ca data .pt file as it's own dict
    with os.scandir(input_path) as it:
        for mov in it:
            if (
                mov.is_file()
                and mov.name.startswith(file_prefix)
                and mov.name.endswith(".pt")
            ):
                tmp_dict = torch.load(
                    mov.path, map_location=torch.device("cpu")
                )  # load dict onto cpu, manually move tensors to gpu if available later
                for key in tmp_dict.keys():
                    vstim_mov[key] = tmp_dict[key]

    for mov in vstim_mov.keys():
        vstim_mov[mov][mov_key].to(device=my_device, dtype=my_dtype)

    return vstim_mov


class LCDataLoader(torch.utils.data.Dataset):
    """
    dataloader for all recorded cell types
    no dataset balancing or train/validation dataset split functions are implemented
    assume calcium traces and movies provided are matched in sampling freq (180Hz)

    currently only map-style dataloader is implemented
    movies are always padded to the same length in time to match the longest movie
    """

    def __init__(
        self,
        movies: dict,
        ca_traces: dict,
        max_num_t_pts: int,
        avg_xTrials=True,
        my_device=torch.device("cuda"),
        mov_bg_int=0.498,
        mov_key="rect_mov",
        num_t_pts_prepad=720,
        px_res=1,
        opts_meas_t_start=-1,
    ):
        """
        create dataloader using the provided movies (input) and ca_traces (output)

        Args:
            movies (dict):
                keynames are vstim_barcode, use mov_key to access actual tensor
                data, see load_vstim_mov method
            ca_traces (dict):
                calcium time traces, see load_ca_data method for data structure
            max_num_t_pts (int):
                generated from load_ca_data method
                info used for padding traces
            avg_xTrials (bool):
                dataloader config option
                True = average over individual trials
                False = single trial
                most ca imaging measurements have 2-3 trial repeats
            device (device)
            movie_background_intensity (float):
                used to generate blank frames for padding movies
            mov_key (str):
                should be either 'rect_mov' or 'hex_mov' depending on
                whether user sampled movie on a rectangular or hexagonal lattice
            num_t_pts_prepad (int):
                num of frames to pad before movie and traces
            opts_meas_t_start (float):
                start time for loss calculation
        """

        super().__init__()

        # save config settings
        self.device = my_device
        self.max_num_t_pts = max_num_t_pts
        self.mov_key = mov_key
        self.bg_intensity = mov_bg_int
        self.px_res = px_res  # movie's implicit degree/pixel resolution

        # actual datasets are stored in self.movies, self.ca_traces
        self.movies = movies
        self.ca_traces = ca_traces

        # map integer indices into the actual datasets (self.movies, self.ca_traces)
        self.idx2dict = {}  # store int index to (movies, ca_traces) dict key mapping
        self.uniq_neuron_ref = {}  # store unique lc_type, and uid in this dictionary
        self._map_data2ind(avg_xTrials)
        self._map_uid2neuronNum()

        # pre-pad trace and movie to let the model reach steady state
        self.pre_pad_t_pts = (
            num_t_pts_prepad  # number of frames to pad at the beginning of trace
        )

        # stimulus onset is at t=0, define time (sec) for when to start
        # loss calculation measurements
        # eg t = -1.5 means start loss meas 1.5 second before stimulus onset
        self.opts_meas_t_start = opts_meas_t_start

    def __getitem__(self, i: int) -> dict:
        """
        retrieve a minibatch based on the input index i

        calcium traces and movies are padded in time and shifted in xy as needed
        """

        # get keynames for the i_th index
        lc_type = self.idx2dict[i]["lc_type"]
        data_file = self.idx2dict[i]["data_file"]
        vstim = self.idx2dict[i]["vstim"]
        uid = self.idx2dict[i]["uid"]
        trace_ind = self.idx2dict[i]["trace_ind"]
        neuron_num = self.idx2dict[i]["neuron_num"]

        # -------------------- process calcium recording -----------------------
        # select ca trace and t-axis
        sel_ca_t_axis = self.ca_traces[lc_type][data_file][vstim][
            "t_axis"
        ]  # this is the original time axis, not padded
        sel_trace = torch.index_select(
            self.ca_traces[lc_type][data_file][vstim]["neuron"][uid]["traces"].to(
                self.device
            ),
            0,
            trace_ind.to(self.device),
        )

        if (
            sel_trace.size()[0] > 1
        ):  # if more than one trial selected, average across trials
            sel_trace = torch.mean(sel_trace, 0).to(self.device)

        # pad trace so that all traces have the same length
        orig_trace_len = len(sel_trace.squeeze())  # save trace length pre-padding
        sel_trace = self._pad_trace(sel_trace).squeeze()

        # define start index for loss calculation
        # t=0 stimulus onset, useful to set loss calc to begin at t<0 to constrain
        # activity (to be flat) before stimulus onset
        tstart_idx = self._find_time_idx(
            t_axis=sel_ca_t_axis, time_pt=self.opts_meas_t_start
        )

        # shift indices to account for pre-padding and trace length
        rec_inds = torch.tensor(
            [tstart_idx + self.pre_pad_t_pts, orig_trace_len - 1 + self.pre_pad_t_pts]
        )  # [start index, end index]

        # -------------------- process vstim movie -----------------------
        if vstim in self.movies:  # check if movie exists
            sel_mov = self.movies[vstim][self.mov_key].to(self.device)
            sel_mov_t_axis = self.movies[vstim]["t_axis"].to(self.device)
        else:
            print(vstim + " is not found")

        # pad movie by adding grey blank screen frames before/after movie
        # along time dimension to match the calcium recording
        sel_mov = self._pad_movie(sel_mov, sel_ca_t_axis, sel_mov_t_axis)

        # flip movie for hex_mov types
        # it depends on how movies are pre-processed, should always check
        # visually to make sure movies look fine
        if self.mov_key == "hex_mov":
            sel_mov = torch.flip(sel_mov, [1])  # dim1 = row or y-axis

        # coordinate system used in vstim generation is azi: -180 to +180 deg, ele: 0 (north pole) to 180 deg (south pole)
        # ele=0 is the equator
        frontward_shift = self.ca_traces[lc_type][data_file][vstim]["neuron"][uid][
            "azi_shift"
        ].to(
            dtype=torch.float
        )  # relative shift (deg) = neuron_center - vstim_center, positive # means vstim is closer to midline than neuron
        upward_shift = self.ca_traces[lc_type][data_file][vstim]["neuron"][uid][
            "ele_shift"
        ].to(
            dtype=torch.float
        )  # relative shift (deg) = neuron_center - vstim_center, positive # means vstim is closer to north pole than neuron
        sel_mov = self._shift_movie(
            sel_mov, azi_shift=frontward_shift, ele_shift=upward_shift
        )  # shape (n_frames, row, col)

        sel_mov.unsqueeze_(
            1
        )  # shape (n_frames, channels=1, rows, cols). note that the dataloader will create the batch dimension at dim=0, do not add it manually

        # -------------------- return minibatch as dict -----------------------

        return {
            "ca_trace": sel_trace,
            "movie": sel_mov - self.bg_intensity,
            "lc_type": lc_type,
            "recording_indices": rec_inds,
            "uid": torch.tensor(uid, dtype=torch.int64),
            "neuron_num": neuron_num,
            "vstim_barcode": vstim,
        }

    def __len__(self) -> int:
        return len(self.idx2dict)

    def count_num_mov(self) -> int:
        return len(self.movies)

    def _map_data2ind(self, avg_xTrials) -> None:
        """
        generate indices to map into self.ca_traces and self.movies
        each index entry corresponds to a minibatch, see __getitem__ method

        this method does not copy data or references
        it's just a look-up table to index into the calcium and movie dict
        data structure in a comprehensive (all data entries) and linear fashion

        store mapping in self.idx2dict
            self.idx2dict[#]['lc_type']
                cell type
            self.idx2dict[#]['data_file']
                original ca_trace filename
            self.idx2dict[#]['vstim']
                vstim barcode #
            self.idx2dict[#]['uid']
                unique ID for each recorded neuron
            self.idx2dict[#]['trace_ind']
                index for which trial to select, if avg_xTrials = True,
                then multiple indices are selected correspond to all repeats

        store cell type names in self.uniq_neuron_ref

        Args:
            avg_xTrials (bool):
                whether to select a single trial or combine all repeated trials
                in self.idx2dict[#]['trace_ind']
        """

        cur_idx = 0  # counter for the new index scheme

        for lc_type in self.ca_traces.keys():
            if (
                lc_type not in self.uniq_neuron_ref
            ):  # save unique lc_type into uniq_neuron_ref dict
                self.uniq_neuron_ref[lc_type] = {}

            for data_file in self.ca_traces[lc_type].keys():
                for vstim in self.ca_traces[lc_type][data_file].keys():
                    for uid in self.ca_traces[lc_type][data_file][vstim][
                        "neuron"
                    ].keys():
                        if (
                            uid not in self.uniq_neuron_ref[lc_type]
                        ):  # within each lc_type, save the unique neuron id (UID).
                            self.uniq_neuron_ref[lc_type][
                                uid
                            ] = None  # will assign a number for each uid later in another function
                            # print( str(lc_type) + ' has a new neuron with uid: ' + str(uid)) # for debugging, to make sure new neurons are imported

                        if avg_xTrials:
                            self.idx2dict[cur_idx] = {}

                            self.idx2dict[cur_idx]["lc_type"] = lc_type
                            self.idx2dict[cur_idx]["data_file"] = data_file
                            self.idx2dict[cur_idx]["vstim"] = vstim
                            self.idx2dict[cur_idx]["uid"] = uid

                            num_trials = self.ca_traces[lc_type][data_file][vstim][
                                "neuron"
                            ][uid]["traces"].size()[
                                0
                            ]  # dim0 = trial #, dim1 = time pts
                            self.idx2dict[cur_idx]["trace_ind"] = torch.arange(
                                0, num_trials, 1
                            )  # start num, end num, step size. generated num are always < end num, so does not include it

                            cur_idx += 1
                        else:
                            num_trials = self.ca_traces[lc_type][data_file][vstim][
                                "neuron"
                            ][uid]["traces"].size()[0]

                            # print( str(data_file) + '  has  ' +  str(num_trials) + ' traces') # for debugging, make sure correct # of traces are parsed for each neuron

                            for i in range(num_trials):
                                self.idx2dict[cur_idx] = {}

                                self.idx2dict[cur_idx]["lc_type"] = lc_type
                                self.idx2dict[cur_idx]["data_file"] = data_file
                                self.idx2dict[cur_idx]["vstim"] = vstim
                                self.idx2dict[cur_idx]["uid"] = uid

                                self.idx2dict[cur_idx]["trace_ind"] = torch.tensor(
                                    [i]
                                )  # select a single trial

                                cur_idx += 1

    def _pad_movie(self, mov: Tensor, ca_t_axis: Tensor, mov_t_axis: Tensor) -> Tensor:
        """
        pad movie along time dimension with gray (blank) frames

        Args:
            mov (Tensor):
                movie, assume to start at t=0 and ends after stimuls ends
            ca_t_axis, mov_t_axis (Tensor):
                calcium and movie time axis

        Returns:
            padded_mov (Tensor)

        """

        # find t=0 in ca t-axis to figure out how many padding frames pre-movie onset
        ca_t_zero_idx = self._find_time_idx(ca_t_axis, time_pt=0)

        mov_num_t_pts, mov_num_rows, mov_num_cols = mov.size()

        # say if index at 5 correspond to zero, then there are 5 frames before time = zero (with indices 0,1,2,3,4)
        num_frames_pre = ca_t_zero_idx
        num_frames_post = self.max_num_t_pts - mov_num_t_pts - num_frames_pre

        # generate blank frame template
        blank_frame = self.bg_intensity * torch.ones(
            1, mov_num_rows, mov_num_cols, device=self.device
        )

        pre_blank_frames = blank_frame.repeat(
            num_frames_pre + self.pre_pad_t_pts, 1, 1
        )  # replicate along dim0=time
        post_blank_frames = blank_frame.repeat(
            num_frames_post, 1, 1
        )  # replicate along dim0=time

        padded_mov = torch.cat(
            (
                pre_blank_frames.to(self.device),
                mov.to(self.device),
                post_blank_frames.to(self.device),
            ),
            0,
        )  # may be concatenating empty tensors, but torch.cat method can handle it properly
        padded_mov.to(self.device)

        return padded_mov

    def _shift_movie(
        self, mov: Tensor, azi_shift: Tensor = 0, ele_shift: Tensor = 0
    ) -> Tensor:
        """
        shift movie along xy (row, col) as needed

        Args:
            mov (Tensor)
            azi_shift, ele_shift (Tensor): amount to shift in degrees

        Returns:
            mov (Tensor)

        """

        # need to transform angular shifts in degrees into # of cols or rows
        # ele_shift and azi_shift are in degrees, need to divide by the movie's pixel resolution
        row_shift = int(
            torch.round(ele_shift / self.px_res)
        )  # must be integer, not float
        col_shift = int(
            torch.round(azi_shift / self.px_res)
        )  # must be integer, not float

        mov_n_t_pts = mov.size()[0]
        mov_n_rows = mov.size()[1]
        mov_n_cols = mov.size()[2]

        # perform spatial shifts by filling in new xy regions with self.bg_intensity
        # this is not a cyclic shift, since movie may only cover a small visual area (i.e. not 360deg coverage)
        if (self.mov_key == "hex_mov") or (self.mov_key == "rect_mov"):
            row_shift *= -1  # invert shift direction

        if row_shift >= 0:
            pad_mov = (
                torch.ones(mov_n_t_pts, row_shift, mov_n_cols, device=self.device)
                * self.bg_intensity
            )
            mov = torch.cat((mov[:, row_shift:, :], pad_mov), dim=1)  # original
        else:
            pad_mov = (
                torch.ones(mov_n_t_pts, -1 * row_shift, mov_n_cols, device=self.device)
                * self.bg_intensity
            )
            mov = torch.cat((pad_mov, mov[:, :row_shift, :]), dim=1)  # original

        if col_shift >= 0:
            pad_mov = (
                torch.ones(mov_n_t_pts, mov_n_rows, col_shift, device=self.device)
                * self.bg_intensity
            )
            mov = torch.cat((mov[:, :, col_shift:], pad_mov), dim=2)
        else:
            pad_mov = (
                torch.ones(mov_n_t_pts, mov_n_rows, -1 * col_shift, device=self.device)
                * self.bg_intensity
            )
            mov = torch.cat((pad_mov, mov[:, :, :col_shift]), dim=2)

        return mov

    def _pad_trace(self, sel_trace: Tensor) -> Tensor:
        """
        pad before and after the sel_trace using average values at the beginning
        and end of the trace

        Note: the __getitem__ method explicitly enforces time indices used for
        loss calculation later on will never include padded time points
        since they're not real
        """

        sel_trace = (
            sel_trace.squeeze()
        )  # assume trace is already averaged across trials if needed

        diff_t_pts = self.max_num_t_pts - len(sel_trace)

        # pad at the end of the trace
        if diff_t_pts != 0:  # need to pad trace
            pad_val_post = torch.mean(
                sel_trace[-18:-1]
            )  # take the average of the last X time points, assuming 180Hz, this is averaging the last 100ms in the trace
            t_end_pad = pad_val_post * torch.ones(diff_t_pts, device=self.device)
            sel_trace = torch.cat((sel_trace, t_end_pad), 0)

        # pad at the beginning of the trace
        pad_val_pre = torch.mean(sel_trace[0:18])
        t_start_pad = pad_val_pre * torch.ones(self.pre_pad_t_pts, device=self.device)
        sel_trace = torch.cat((t_start_pad, sel_trace), 0)

        return sel_trace

    def _find_time_idx(self, t_axis: Tensor, time_pt) -> int:
        """
        return index value for t=time_pt in input t_axis (Tensor)
        """
        boolean_mask = t_axis <= time_pt
        inds = torch.nonzero(boolean_mask)
        time_idx = torch.max(inds)

        return time_idx

    def _map_uid2neuronNum(self) -> None:
        """
        assign int value to each unique neuron within a given cell type
        (range 0 to # neurons -1 for each cell type)

        self.idx2dict[#]['uid'] contains ID that is unique across all cell types
        but this is a very long integer and not easy to use

        much easier to use a combination of lc_type + neuron_num to uniquely
        identify neurons (for loss calculations)
        """

        # assign neuron number
        for lc_type in self.uniq_neuron_ref.keys():
            neuron_counter = 0  # reset counter for each cell type, number refers to uniq neuron within a cell type
            for uid in self.uniq_neuron_ref[lc_type].keys():
                self.uniq_neuron_ref[lc_type][uid] = torch.tensor(
                    [neuron_counter], device=self.device
                )
                neuron_counter += 1

        # now walk through the full dataset index dict to assign neuron #
        for n in self.idx2dict.keys():
            lc_type = self.idx2dict[n]["lc_type"]
            uid = self.idx2dict[n]["uid"]
            self.idx2dict[n]["neuron_num"] = self.uniq_neuron_ref[lc_type][uid]

        return

    def get_num_of_neurons(self, lc_type: str) -> int:
        """
        return # of neurons within a cell type
        """

        return len(self.uniq_neuron_ref[lc_type])

    def get_idx_by_criteria(self, lc_type: str, vstim: str, uid: int = None) -> list:
        """find ind by celltype and vstim barcode

        Args:
            lc_type (str): cell type name, special case is 'any' where all types considered
            vstim (str): visual stimulus barcode, expect "vstim_#"
            uid (int): neuron ID

        Returns:
            list: dataset indices that match celltype and vstim
        """
        idx_match = []

        if uid is not None:  # prioritize searching by uid
            for i in range(len(self.idx2dict)):
                if (
                    uid == int(self.idx2dict[i]["uid"])
                    and vstim == self.idx2dict[i]["vstim"]
                ):  # grab data from a single neuron
                    idx_match.append(i)
        elif lc_type == "any":  # ignore cell type
            for i in range(len(self.idx2dict)):
                if vstim == self.idx2dict[i]["vstim"]:
                    idx_match.append(i)
        else:  # grab all neuron of the specified type
            for i in range(len(self.idx2dict)):
                if (
                    lc_type == self.idx2dict[i]["lc_type"]
                    and vstim == self.idx2dict[i]["vstim"]
                ):  # grab data from all neurons of the same type
                    idx_match.append(i)

        return idx_match
