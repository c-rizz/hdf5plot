#!/usr/bin/env python3 
from __future__ import annotations

import io
import matplotlib.backend_bases
import h5py
import argparse
import matplotlib.axes
import matplotlib.widgets
import matplotlib.pyplot as plt
import numpy as np
import readline # enables better input() features (arrow keys, history)
# import seaborn as sns
import os
from typing import TypeVar
import shutil
import math 
import traceback
import mplcursors
import time
import itertools
import zipfile

_K = TypeVar("_K")
_V = TypeVar("_V")


def exc_to_str(exception):
    # return '\n'.join(traceback.format_exception(etype=type(exception), value=exception, tb=exception.__traceback__))
    return '\n'.join(traceback.format_exception(exception, value=exception, tb=exception.__traceback__))

def recdict_access(rdict : dict[_K,_V], keylist : list[_K]) -> dict[_K,_V]:
    if len(keylist)==0:
        return rdict
    return recdict_access(rdict[keylist[0]], keylist[1:])

# def multiplot(n_cols_rows, plotnames, datas : dict, filename : dict, labels : dict, titles : dict):

plot_count = 0
def plot(data, labels = None, title : str = "HDF5Plot", xlims=None, print_raw : bool = False):
    """ Plots a data array of shape (K,N), where K is the number of data points per series and
    N is the number of data series. I.e. 12 joint positions evolving over 100 timesteps
    would be a data array of shape (100,12).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (K,N) (or (K,) for a single series) with K points for each of the N series.
    labels : list[str], optional
        One label per series, used in the legend. If None, series are labeled by their index.
    title : str, optional
        Title of the plot window and axes, by default "HDF5Plot".
    xlims : tuple[float, float], optional
        (min, max) limits for the x axis, by default None (auto).
    print_raw : bool, optional
        If True, also print each series' raw values to stdout, by default False.
    """
    print(f"plotting data with shape {data.shape}")

    global plot_count
    plot_count += 1
    ax : matplotlib.axes.Axes
    fig, ax = plt.subplots(num=title+str(plot_count))
    ax.grid(True, linestyle=":")
    ax.set_title(title)
    if len(data.shape)==1:
        data = np.expand_dims(data,1)
    series_num = data.shape[1]
    if labels is None:
        labels = [f"{i}" for i in range(series_num)]
        if len(labels)==1:
            labels = labels[0]
    linewidth = 1.5
    lines = ax.plot(data, label=labels, linewidth=linewidth)
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    legend.set_draggable(True)
    ax.set_xlim(xlims)

    if print_raw:
        np.set_printoptions(precision=3, suppress=True)
        print("raw data:")
        for i in range(data.shape[1]):
            print(f"series {i} : {labels[i] if labels is not None else ''}")
            print(data[:,i])

    map_legend_to_ax = {}  # Will map legend lines to original lines.
    for legend_line, ax_line in zip(legend.get_lines(), lines):
        legend_line.set_picker(5)  # Enable picking on the legend line. (radius at 5pt)
        map_legend_to_ax[legend_line] = ax_line
    def on_pick(event : matplotlib.backend_bases.PickEvent):
        if event.mouseevent.button == matplotlib.backend_bases.MouseButton.LEFT:
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legend_line = event.artist
            if legend_line in map_legend_to_ax:
                ax_line = map_legend_to_ax[legend_line]
                visible = not ax_line.get_visible()
                ax_line.set_visible(visible)
                # Change the alpha on the line in the legend, so we can see what lines
                # have been toggled.
                legend_line.set_alpha(1.0 if visible else 0.2)
        elif event.mouseevent.button == matplotlib.backend_bases.MouseButton.RIGHT:
            # On the pick event, find the original line corresponding to the legend
            # proxy line, and toggle its visibility.
            legend_line = event.artist
            if legend_line in map_legend_to_ax:
                # Always make the line visible
                ax_line = map_legend_to_ax[legend_line]
                ax_line.set_visible(True)
                alpha = legend_line.get_alpha()
                # When we hide all others we set alpha to 0.99, just to then recognize the situation
                hide_all_others = alpha is None or alpha >= 1.0
                if hide_all_others:
                    legend_line.set_alpha(0.99)
                else:
                    legend_line.set_alpha(1.0)
                for ll,al in map_legend_to_ax.items():
                    if ll != legend_line:
                        ll.set_alpha(0.2 if hide_all_others else 1.0)
                        al.set_visible(not hide_all_others)
        # elif event.mouseevent.button == matplotlib.backend_bases.MouseButton.RIGHT:
        #     for legend_line,ax_line in map_legend_to_ax.items():
        #         ax_line.set_visible(False)
        #         legend_line.set_alpha(0.2)
        fig.canvas.draw()
    fig.canvas.mpl_connect('pick_event', on_pick)
    mplcursors.cursor(lines)
    # def hover(event):
    #     t0 = time.monotonic()
    #     for line in lines:
    #         if line.contains(event):
    #             line.set_linewidth(linewidth*2)
    #             # print(f"line {line.get_label()} enlarged")
    #         else:
    #             line.set_linewidth(linewidth)
    #             # print(f"line {line.get_label()} not enlarged")
    #     fig.canvas.draw()
    #     print(f"hovercallback t = {time.monotonic()-t0}")
    # fig.canvas.mpl_connect("motion_notify_event", hover)
    # matplotlib.use('TkAgg')
    def on_resize(event):
        fig.set_layout_engine('constrained')
        fig.canvas.draw()
    fig.canvas.mpl_connect('resize_event', on_resize)
    fig.show()
    on_resize(None)

hist_count = 0
def hist(data, title : str = "HDF5Plot", bins : int = 50, xlims=None, print_raw : bool = False):
    """ Plots a histogram of a data array, in the wandb style: for a 2D input of shape (K,N)
    (K timesteps, N values per timestep), a histogram over the N values is computed at each
    of the K timesteps and the result is displayed as a heatmap, with time on the x axis,
    the histogram bins on the y axis, and color indicating the count in each bin. This shows
    how the distribution of the N values evolves over time.

    If there is only a single timestep (K==1, or a 1D input of shape (N,)), there is no time
    axis to build a heatmap over, so a normal histogram of the N values is plotted instead.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (K,N) (or (N,) for a single sample) with K timesteps and N values per timestep.
    title : str, optional
        Title of the plot window and axes, by default "HDF5Plot".
    bins : int, optional
        Number of histogram bins, by default 50.
    xlims : tuple[float, float], optional
        (min, max) limits for the value axis the histogram bins span, by default None (auto).
    print_raw : bool, optional
        If True, also print the raw data to stdout, by default False.
    """
    print(f"plotting histogram of data with shape {data.shape}")

    global hist_count
    hist_count += 1
    ax : matplotlib.axes.Axes
    fig, ax = plt.subplots(num=title+"_hist"+str(hist_count))
    ax.grid(True, linestyle=":")
    ax.set_title(title)

    if len(data.shape) == 1:
        data = np.expand_dims(data, 0)
    timesteps_num = data.shape[0]
    vmin, vmax = xlims if xlims is not None else (float(np.min(data)), float(np.max(data)))

    if timesteps_num == 1:
        ax.hist(data[0], bins=bins, range=(vmin, vmax))
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    else:
        bin_edges = np.linspace(vmin, vmax, bins + 1)
        heatmap = np.stack([np.histogram(data[t], bins=bin_edges)[0] for t in range(timesteps_num)], axis=1)
        im = ax.imshow(heatmap, origin="lower", aspect="auto",
                        extent=(0, timesteps_num, vmin, vmax), cmap="viridis")
        fig.colorbar(im, ax=ax, label="count")
        ax.set_xlabel("timestep")
        ax.set_ylabel("value")

    if print_raw:
        np.set_printoptions(precision=3, suppress=True)
        print("raw data:")
        print(data)

    def on_resize(event):
        fig.set_layout_engine('constrained')
        fig.canvas.draw()
    fig.canvas.mpl_connect('resize_event', on_resize)
    fig.show()
    on_resize(None)

def cmd_cd(file, current_path, *args, **kwargs):
    """ Move into a the dataset structure as if it was a folder structure. \
        E.g. 'cd data' moves into the 'data' dict and 'cd ..' moves back \
        up the hierarchy."""
    k = recdict_access(file, current_path).keys()
    if len(args) == 1:
        current_path = []                
    elif args[1] == "..":
        current_path = current_path[:-1]
    else:
        if args[1] in k:
            new_path = current_path + [args[1]]
            if isinstance(recdict_access(file, current_path), dict):
                current_path = new_path
            else:
                print(f"{args[1]} is not dict-like")
        else:
            print(f"{args[1]} not found")
    return current_path, True

def cmd_ls(file, current_path, *args, **kwargs):
    prefix = args[0] if len(args)>0 else ""
    ks = recdict_access(file, current_path).keys()
    max_k_len = max([len(k) for k in ks]) 
    ks = [(str(k)).rjust(max_k_len) for k in ks]
    ks = [k for k in ks if k.strip().startswith(prefix.strip())]
    elements_per_row = int(shutil.get_terminal_size().columns/max_k_len)
    print('\n'.join([''.join(ks[p:p+elements_per_row]) for p in range(0,len(ks), elements_per_row)]))
    return current_path, True

def cmd_quit(file, current_path, *args, **kwargs):
    return current_path, False


def cmd_info(file, current_path, *args, **kwargs):
    """ Show information about a data element. E.g. 'info state_robot' prints its shape. """
    if len(args) < 1:
        print(f"Argument missing for info.")
        return current_path, True
    available_fields = recdict_access(file, current_path).keys()
    field = args[0]
    if field not in available_fields:
        matches = [af for af in available_fields if af.startswith(field)]
        if len(matches) == 1:
            field = matches[0]
        else:
            print(f"Possible fields = " + (" ; ".join(matches)))
            return current_path, True
    element = recdict_access(file, current_path + [field])
    if hasattr(element, "shape"):
        print(f"{field} shape: {element.shape}")
    else:
        print(f"{field} has no shape (not a dataset).")
    return current_path, True


def _select_field_data(file, current_path, argument_groups, lims_arg_name="--xlims="):
    """ Shared field/column selection and slicing logic used by cmd_plot and cmd_hist.
    Resolves each argument group to a field (by exact match or unambiguous prefix), applies
    the requested column slicing, and resolves the field's labels if available.
    Returns a dict field -> (data, labels, lims), or None if a field could not be resolved
    (an error has already been printed in that case, and the caller should just return). """
    fields_tbd = {}
    for args in argument_groups:
        available_fields = recdict_access(file, current_path).keys()
        field : str = ""
        if args[0] in available_fields:
            field = args[0]
        else:
            matches = []
            for af in recdict_access(file, current_path).keys():
                if  af.startswith(args[0]) and not af.endswith("_labels"):
                    matches.append(af)
            if len(matches)==1:
                field = matches[0]
            else:
                print(f"Possible fields = "+(" ; ".join(matches)))
                return None
        print(f"selected {field}")
        data = np.array(recdict_access(file, current_path+[field]))
        if len(data.shape) == 1:
            data = np.expand_dims(data,1)
        cols_num = data.shape[1]
        columns = None
        lims = None
        if len(args)>=2:
            columns = []
            for arg in args[1:]:
                if arg.startswith("--"):
                    if arg.startswith(lims_arg_name):
                        lims = [int(l) for l in arg[len(lims_arg_name):].split(",")]
                    else:
                        print(f"Unrecognized arg {arg}")
                else:
                    groups = arg.split(",") # e.g. "1:4,7:9,11,12" gets split in ["1:4","7:9","11","12"]
                    for g in groups:
                        if ":" in g:
                            slice_offset = g.split("+")
                            if len(slice_offset) == 1:
                                slice_offset.append("0")
                            slice,offset = slice_offset
                            e = slice.split(":")
                            if len(e)>3:
                                raise RuntimeError(f"Invalid slice '{g}'")
                            if len(e)==2:
                                e.append("")
                            if e[0] == "": e[0] = 0
                            if e[1] == "": e[1] = cols_num
                            if e[2] == "": e[2] = 1
                            e = [int(es) for es in e]
                            columns += [c+int(offset) for c in list(range(cols_num))[e[0]:e[1]:e[2]]]
                        else:
                            columns.append(int(g))
        if columns is not None:
            data = data[:,columns]
        if columns is None:
            columns = list(range(data.shape[1]))
        maybe_labels_name = field+"_labels"
        default_columns = [f"{field}_{c}" for c in columns]
        if maybe_labels_name in recdict_access(file, current_path).keys():
            try:
                labels = np.array(recdict_access(file, current_path+[maybe_labels_name]))[0]
                labels = [a.tobytes().decode("utf-8").strip() for a in list(labels)]
                # print(f"Found {len(labels)} labels {labels}")
                if columns is not None:
                    labels = [labels[i] if i<len(labels) else str(i) for i in columns]
                n = "\n"
                print("using labels\n"+f"{n.join([f'{i} : {l}' for i,l in zip(columns,labels)])}")
            except Exception as e:
                print(f"Failed to read labels from {maybe_labels_name} with exception {e.__class__.__name__}: {e}")
                labels = default_columns
        else:
            labels = default_columns
        fields_tbd[field] = (data, labels, lims)
    return fields_tbd

def cmd_plot(file, current_path, *args, **kwargs):
    """ Plot a data element. For example 'plot state_robot 0:96:8+2 --xlims=-1,30' plots from state_robot a
        slice from 0 to 96 with stride 8 and an offset of 2 (i.e. 2,10,18,...), with x axis limits -1 and 30.
        You can plot multiple data from multiple fields at once, e.g. 'plot state_robot 0:96:8+2 ; state_goal 0'. """
    if len(args) < 1:
        print(f"Argument missing for plot.")

    argument_groups = [list(y) for x, y in itertools.groupby(args, lambda z: z.strip() == ";") if not x]
    plots_tbd = _select_field_data(file, current_path, argument_groups, lims_arg_name="--xlims=")
    if plots_tbd is None:
        return current_path, True

    all_data = None
    all_fields = []
    all_labels = []
    all_xlims = None
    for field, plot_tbd in plots_tbd.items():
        all_fields.append(field)
        if all_data is None:
            all_data = plot_tbd[0]
            all_labels = plot_tbd[1]
            all_xlims = plot_tbd[2]
        else:
            all_data = np.hstack((all_data, plot_tbd[0]))
            all_labels = all_labels + plot_tbd[1]
            all_xlims = plot_tbd[2]
    plot(all_data,
        labels=all_labels,
        title = os.path.basename(kwargs["filename"])+"/["+",".join(current_path+all_fields)+"]",
        xlims=all_xlims,
        print_raw = False)
    return current_path, True

def cmd_hist(file, current_path, *args, **kwargs):
    """ Plot a data element as a histogram, wandb-style: if the selected data spans multiple
        timesteps, a histogram is computed at each timestep and shown as a heatmap over time;
        otherwise a normal histogram is shown. For example 'hist state_robot 0:96:8+2 --xlims=-1,30'
        histograms state_robot's slice from 0 to 96 with stride 8 and an offset of 2 (i.e. 2,10,18,...),
        with value axis limits -1 and 30. You can combine data from multiple fields into the same
        histogram at once, e.g. 'hist state_robot 0:96:8+2 ; state_goal 0'. """
    if len(args) < 1:
        print(f"Argument missing for hist.")

    argument_groups = [list(y) for x, y in itertools.groupby(args, lambda z: z.strip() == ";") if not x]
    hists_tbd = _select_field_data(file, current_path, argument_groups, lims_arg_name="--xlims=")
    if hists_tbd is None:
        return current_path, True

    all_data = None
    all_fields = []
    all_xlims = None
    for field, hist_tbd in hists_tbd.items():
        all_fields.append(field)
        if all_data is None:
            all_data = hist_tbd[0]
        else:
            all_data = np.hstack((all_data, hist_tbd[0]))
        all_xlims = hist_tbd[2]
    hist(all_data,
        title = os.path.basename(kwargs["filename"])+"/["+",".join(current_path+all_fields)+"]",
        xlims=all_xlims,
        print_raw = False)
    return current_path, True

img_count = 0
def show_frames(frames, title : str = "HDF5Plot", vmin : float = None, vmax : float = None):
    print(f"showing {frames.shape[0]} frames of shape {frames.shape[1:]}")
    global img_count
    img_count += 1
    n_frames, c, h, w = frames.shape

    def to_image(idx):
        frame = frames[idx]
        if c == 1:
            return frame[0]
        if c in (3, 4):
            return np.transpose(frame, (1, 2, 0))
        print(f"Cannot display {c} channels as an image, showing channel 0 only.")
        return frame[0]

    print(f"pixel values: dtype={frames.dtype}, min={frames.min()}, max={frames.max()}")

    fig, ax = plt.subplots(num=title+str(img_count))
    fig.subplots_adjust(bottom=0.2)
    ax.set_title(f"{title} - frame 0/{n_frames-1}")
    ax.set_xticks([])
    ax.set_yticks([])
    if vmin is None:
        vmin = float(frames.min())
    if vmax is None:
        vmax = float(frames.max())
    print(f"scaling pixel values with vmin={vmin}, vmax={vmax}")
    im = ax.imshow(to_image(0), cmap="gray" if c == 1 else None, vmin=vmin, vmax=vmax)

    def format_coord(x, y):
        col, row = int(round(x)), int(round(y))
        img = im.get_array()
        if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
            return f"x={col}, y={row}, value={img[row, col]}"
        return f"x={x:.1f}, y={y:.1f}"
    ax.format_coord = format_coord

    cursor = mplcursors.cursor(im, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        col, row = int(round(sel.target[0])), int(round(sel.target[1]))
        img = im.get_array()
        if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
            sel.annotation.set_text(f"({col}, {row}): {img[row, col]}")

    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = matplotlib.widgets.Slider(slider_ax, "Frame", 0, n_frames-1, valinit=0, valstep=1)
    def update(val):
        idx = int(slider.val)
        im.set_data(to_image(idx))
        ax.set_title(f"{title} - frame {idx}/{n_frames-1}")
        fig.canvas.draw_idle()
    slider.on_changed(update)
    fig._hdf5plot_slider = slider  # keep a reference alive, otherwise the slider stops responding
    fig.show()

def cmd_img(file, current_path, *args, **kwargs):
    """ Draw a data element as a sequence of images, with a slider to scroll through frames. \
        E.g. 'img camera_obs 1x64x64' interprets each row of camera_obs as a CxHxW image \
        with 1 channel, 64 height and 64 width. \
        By default pixel values are scaled using the data's own min/max. Use --range=0,1 or \
        --range=0,255 (or any other pair of values) to set the scaling explicitly. """
    if len(args) < 2:
        print(f"Usage: img <field> <channels>x<height>x<width> [--range=<min>,<max>]")
        return current_path, True
    available_fields = recdict_access(file, current_path).keys()
    field = args[0]
    if field not in available_fields:
        matches = [af for af in available_fields if af.startswith(field)]
        if len(matches) == 1:
            field = matches[0]
        else:
            print(f"Possible fields = " + (" ; ".join(matches)))
            return current_path, True
    sep = "x" if "x" in args[1] else ","
    try:
        c, h, w = [int(s) for s in args[1].split(sep)]
    except ValueError:
        print(f"Invalid shape '{args[1]}', expected e.g. 1x64x64")
        return current_path, True

    vmin, vmax = None, None
    for arg in args[2:]:
        if arg.startswith("--range="):
            vrange = arg[len("--range="):].split(",")
            if len(vrange) != 2:
                print(f"Invalid --range '{arg}', expected e.g. --range=0,1")
                return current_path, True
            vmin, vmax = float(vrange[0]), float(vrange[1])
        else:
            print(f"Unrecognized arg {arg}")

    data = np.array(recdict_access(file, current_path+[field]))
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    frame_size = c*h*w
    if data.shape[-1] != frame_size:
        print(f"Field '{field}' last dim is {data.shape[-1]}, doesn't match {c}x{h}x{w} = {frame_size}")
        return current_path, True
    frames = data.reshape(data.shape[0], c, h, w)
    show_frames(frames,
        title=os.path.basename(kwargs["filename"])+"/["+",".join(current_path+[field])+"]",
        vmin=vmin, vmax=vmax)
    return current_path, True

from collections import defaultdict
def cmd_help(file, current_path, *args, **kwargs):
    """ This help command. """
    cmds = kwargs["cmds"]
    cmds_by_func = defaultdict(list)
    for key, value in sorted(cmds.items()):
        cmds_by_func[value].append(key)
    print(f"Available commands:")
    n = "\n"
    for func,cmd_names in cmds_by_func.items():
        doc = func.__doc__
        if doc is None:
            doc = "No documentation."
        doc = doc.replace(n,' ')
        doc = ' '.join([k for k in doc.split(" ") if k])
        print(f" - {', '.join(cmd_names)} :\n"
              f"    {doc}")
    return current_path, True

def print_progress(current : int, total : int, prefix : str = "Loading", bar_len : int = 30):
    frac = min(current/total, 1.0) if total > 0 else 1.0
    filled = int(bar_len*frac)
    bar = "#"*filled + "-"*(bar_len-filled)
    mb_current, mb_total = current/(1024*1024), total/(1024*1024)
    print(f"\r{prefix} [{bar}] {frac*100:5.1f}% ({mb_current:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    if current >= total:
        print()

def read_zip_member_with_progress(zf : zipfile.ZipFile, member_name : str, chunk_size : int = 4*1024*1024) -> bytes:
    total = zf.getinfo(member_name).file_size
    read = 0
    chunks = []
    print_progress(read, total)
    with zf.open(member_name) as member:
        while True:
            chunk = member.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            read += len(chunk)
            print_progress(read, total)
    return b"".join(chunks)

history_file = os.path.abspath(os.path.expanduser("~/.hdf5plot/.cmd_history.txt"))
def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--file", default = None, type=str, help="File to open")
        ap.add_argument("file", nargs='?', default = None, type=str, help="File to open")

        ap.set_defaults(feature=True)
        args = vars(ap.parse_args())

        fname = args["file"]
        if fname is None:
            print(f"Not input file provided.")
            input("Press ENTER to exit.")
            exit(0)
        print("\33]0;HDF5 Plot - "+fname.split("/")[-1]+"\a")
        current_path = []
        running = True
        cmds = {"cd" : cmd_cd,
                "ls" : cmd_ls,
                "quit" : cmd_quit,
                "exit" : cmd_quit,
                "q" : cmd_quit,
                "plot" : cmd_plot,
                "p" : cmd_plot,
                "hist" : cmd_hist,
                "h" : cmd_hist,
                "img" : cmd_img,
                "info" : cmd_info,
                "help" : cmd_help}

        file_obj = fname
        inner_name = None
        driver = None
        if zipfile.is_zipfile(fname):
            with zipfile.ZipFile(fname, "r") as zf:
                data = read_zip_member_with_progress(zf, "data.hdf5")
            file_obj = io.BytesIO(data)
            driver = "fileobj"


        with h5py.File(file_obj, "r", driver=driver) as f:
            opened_msg = fname if inner_name is None else f"{fname} (inner: {inner_name})"
            print(f"Opened file {opened_msg}")
            print(f"Content:")
            print(list(recdict_access(f, current_path).keys()))
            cmd_help(f,current_path,cmds = cmds)

            def completer(text, state):
                buffer = readline.get_line_buffer()
                tokens = buffer.split()
                completing_first_token = len(tokens) == 0 or (len(tokens) == 1 and not buffer.endswith(" "))
                if completing_first_token:
                    options = sorted(c for c in cmds.keys() if c.startswith(text))
                else:
                    try:
                        options = [str(k) for k in recdict_access(f, current_path).keys()]
                    except Exception:
                        options = []
                    if tokens[0] == "cd":
                        options.append("..")
                    options = sorted(o for o in options if o.startswith(text))
                return options[state] if state < len(options) else None
            readline.set_completer(completer)
            readline.set_completer_delims(" \t\n")
            if readline.__doc__ and "libedit" in readline.__doc__:
                readline.parse_and_bind("bind ^I rl_complete")  # libedit (e.g. macOS) syntax
            else:
                readline.parse_and_bind("tab: complete")  # GNU readline syntax

            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            readline.set_history_length(100)
            # load history once, up front: read_history_file appends into the in-memory
            # history rather than replacing it, so calling it on every loop iteration (as
            # this used to) re-appends the whole file every command, making it grow without
            # bound until the readline backend can no longer parse it back
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass
            except OSError as e:
                # the history file can end up in a format the current readline backend
                # (e.g. libedit vs GNU readline) can't parse; drop it rather than crash
                print(f"History file unreadable ({e}), resetting it.")
                try:
                    os.remove(history_file)
                except OSError:
                    pass

            while running:
                cmd = input("/"+"/".join(current_path)+"> ")
                try:
                    readline.write_history_file(history_file)
                except OSError as e:
                    print(f"Could not save command history ({e}).")

                cmd = " ".join(cmd.split()) # remove repeated spaces
                cmd = cmd.split(" ")
                if len(cmd) == 0:
                    continue

                cmd_name = cmd[0]
                cmd_args = cmd[1:]
                cmd_func = cmds.get(cmd[0],None)
                if cmd_func != None:
                    kwargs = {}
                    kwargs["cmds"] = cmds
                    kwargs["filename"] = fname
                    try:
                        current_path, running = cmd_func(f,current_path, *cmd_args, **kwargs)
                    except Exception as e:
                        print(f"Command failed with exception {e.__class__.__name__}: {e}")
                        print(exc_to_str(e))

                else:
                    print(f"Command {cmd[0]} not found.")
    except Exception as e:
        print(f"Failed with exception: {exc_to_str(e)}")
        input("Press ENTER to close")


if __name__ == "__main__":
    main()
