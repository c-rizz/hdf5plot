#!/usr/bin/env python3 
from __future__ import annotations

import io
import matplotlib.backend_bases
import h5py
import argparse
import matplotlib.axes
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
def plot(data, labels = None, title : str = "HDF5Plot", xlims=None):
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
    ks = recdict_access(file, current_path).keys()
    max_k_len = max([len(k) for k in ks]) 
    ks = [(str(k)+" ").rjust(max_k_len) for k in ks]
    elements_per_row = int(shutil.get_terminal_size().columns/max_k_len)
    print('\n'.join([''.join(ks[p:p+elements_per_row]) for p in range(0,len(ks), elements_per_row)]))
    return current_path, True

def cmd_quit(file, current_path, *args, **kwargs):
    return current_path, False


def cmd_plot(file, current_path, *args, **kwargs):
    """ Plot a data element. For example 'plot state_robot 0:96:8+2 --xlims=-1,30' plots from state_robot a
        slice from 0 to 96 with stride 8 and an offset of 2 (i.e. 2,10,18,...), with x axis limits -1 and 30.
        You can plot multiple data from multiple fields at once, e.g. 'plot state_robot 0:96:8+2 ; state_goal 0'. """
    if len(args) < 1:
        print(f"Argument missing for plot.")

    argument_groups = [list(y) for x, y in itertools.groupby(args, lambda z: z.strip() == ";") if not x]
    plots_tbd = {}
    for args in argument_groups:
        # print(f"cmd_plot({args})")
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
                print(f"Possible fields = "+(",".join(matches)))
                return current_path, True
        print(f"plotting {field}")
        data = np.array(recdict_access(file, current_path+[field]))
        if len(data.shape) == 1:
            data = np.expand_dims(data,1)
        cols_num = data.shape[1]
        columns = None
        xlims = None
        if len(args)>=2:
            columns = []
            for arg in args[1:]:
                if arg.startswith("--"):
                    if arg.startswith("--xlims="):
                        xlims = [int(l) for l in arg[8:].split(",")]
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
        if maybe_labels_name in recdict_access(file, current_path).keys():
            try:
                labels = np.array(recdict_access(file, current_path+[maybe_labels_name]))[0]
                labels = [a.tobytes().decode("utf-8").strip() for a in list(labels)]
                # print(f"Found {len(labels)} labels {labels}")
                if columns is not None:
                    labels = [labels[i] if i<len(labels) else str(i) for i in columns]
                else:
                    columns = list(range(cols_num))
                n = "\n"
                print("using labels\n"+f"{n.join([f'{i} : {l}' for i,l in zip(columns,labels)])}")
            except Exception as e:
                print(f"Failed to get labels from {maybe_labels_name} with exception {e.__class__.__name__}: {e}")
                labels = columns
        else:
            labels = columns
        plots_tbd[field] = (data, labels, xlims)

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
        xlims=all_xlims)
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
                "help" : cmd_help}

        file_obj = fname
        inner_name = None
        driver = None
        if zipfile.is_zipfile(fname):
            with zipfile.ZipFile(fname, "r") as zf:
                data = zf.read("data.hdf5")
            file_obj = io.BytesIO(data)
            driver = "fileobj"


        with h5py.File(file_obj, "r", driver=driver) as f:
            opened_msg = fname if inner_name is None else f"{fname} (inner: {inner_name})"
            print(f"Opened file {opened_msg}")
            print(f"Content:")
            print(list(recdict_access(f, current_path).keys()))
            cmd_help(f,current_path,cmds = cmds)
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            while running:
                try:
                    readline.read_history_file(history_file)
                except FileNotFoundError as e:
                    pass
                cmd = input("/"+"/".join(current_path)+"> ")
                readline.set_history_length(100)
                readline.write_history_file(history_file)
                
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
        print(f"Failed with exception: {e}")
        input("Press ENTER to close")


if __name__ == "__main__":
    main()
