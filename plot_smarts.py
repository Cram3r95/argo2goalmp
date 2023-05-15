#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon May 15 13:47:58 2023
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import pdb
import sys
import csv
import pandas as pd
import git
import os

from typing import Sequence, Tuple, Optional, List

# DL & Math imports

# Plot imports

import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import utils

from av2.utils.typing import NDArrayFloat

# Global variables

_PlotBounds = Tuple[float, float, float, float]
SCENARIO_ROUTE = os.path.join(BASE_DIR,"smarts_simulator_scenarios/scenario_poses_v1")
ADDITIONAL_STRING = "poses_"

SAVE_DIR = os.path.join(BASE_DIR,"smarts_simulator_scenarios/scenario_poses_v1_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

#######################################

def plot_actor_tracks(ax: plt.Axes, 
                      agents_trajectories: List[NDArrayFloat],
                      agents_ids: List[int]) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.


    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    
    for agent_id, agent_trajectory in zip(agents_ids, agents_trajectories):
        if agent_id == 1:
            color = "#F39C12"
        else:
            color = "#27AE60"
            
        plot_polylines(agent_trajectory, agent_id, color=color, line_width=2, zorder=10)
        
def plot_polylines(
    polyline: NDArrayFloat,
    agent_id: int,
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
    zorder: int = 200,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """

    plt.plot(polyline[:, 0], polyline[:, 1], style, linewidth=line_width, color=color, alpha=alpha, zorder=zorder)
    plt.plot(polyline[-1, 0], polyline[-1, 1], "D", color=color, alpha=alpha, zorder=zorder, markersize=9)
    plt.text(polyline[-1, 0] + 1,
            polyline[-1, 1] + 1,
            str(agent_id),
            zorder=zorder+1,
            fontsize=10
            )
        
def main():
    """
    """
    
    ## Get files and sort
    
    files, num_files = utils.load_list_from_folder(SCENARIO_ROUTE)
    file_id_list, _ = utils.get_sorted_file_id_list(files,additional_string=ADDITIONAL_STRING) 

    for file_id in file_id_list:
        if file_id != 73: continue
        
        # Plots
                        
        fig = plt.figure(1, figsize=(8, 7))
        ax = fig.add_subplot(111)
        
        csv_file = os.path.join(SCENARIO_ROUTE, ADDITIONAL_STRING+str(file_id)+".csv")      
        df = pd.read_csv(csv_file, delim_whitespace=True)
        
        # Add column names if it was not previously specified

        df.columns = ["agent_id", "timestamp", "x", "y", "padding"]
        
        # Get agents of the scene (we assume the first one represents our ego-vehicle)
        
        agents_id = df["agent_id"].unique()

        agents_trajectories = []
         
        for agent_id in agents_id:
            agent_data = df[df["agent_id"] == agent_id]
            
            agent_trajectory = agent_data[["x","y"]].to_numpy()
            agents_trajectories.append(agent_trajectory)
            
        plot_actor_tracks(ax, agents_trajectories, agents_id)   
        
        filename = os.path.join(SAVE_DIR, ADDITIONAL_STRING+str(file_id)+".pdf")
        plt.savefig(filename, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)
        plt.close('all')
                                   
if __name__ == "__main__":
    main()