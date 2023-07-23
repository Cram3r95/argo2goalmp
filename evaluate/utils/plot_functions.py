#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon May 15 13:47:58 2023
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

from typing import Tuple, Optional, List

# DL & Math imports

import numpy as np

# Plot imports

import matplotlib.pyplot as plt

# Custom imports

from av2.utils.typing import NDArrayFloat

# Global variables

_PlotBounds = Tuple[float, float, float, float]

#######################################

def plot_actor_tracks(ax: plt.Axes, 
                      agents_info: List[NDArrayFloat],
                      ego_vehicle_id: int) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.


    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    
    for agent_info in agents_info:
        agent_id = int(agent_info[0,0])
        # Filter agent to avoid plotting padded information
        agent_trajectory = agent_info[agent_info[:,-1] == 1][:,2:4] 
        
        if agent_id == ego_vehicle_id:
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
    zorder: int = 1,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 3) polylines to plot. If the third column (padded) is 1, avoid plotting that row
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
            fontsize=10,
            clip_on=True
            )

def plot_predictions(output_predictions, 
                      output_confidences, 
                      prediction_type="multimodal",
                      *,
                      style: str = "-",
                      line_width: float = 1.0,
                      alpha: float = 1.0,
                      color: str = "r",):
    
    num_modes = output_predictions.shape[0]
    
    if prediction_type == "multimodal":
        sorted_confidences = np.sort(output_confidences)
        slope = (1 - 1/num_modes) / (num_modes - 1)
    
    for num_mode in range(num_modes):
        
        if prediction_type == "multimodal":
            conf_index = np.where(output_confidences[num_mode] == sorted_confidences)[0].item()
            transparency = round(slope * (conf_index + 1),2)
        else:
            transparency = 1.0
        
        # Prediction
        plt.plot(
            output_predictions[num_mode, :, 0],
            output_predictions[num_mode, :, 1],
            color="#008000",
            label="Output",
            alpha=transparency,
            linewidth=3,
            zorder=11,
        )
        # Prediction endpoint
        plt.plot(
            output_predictions[num_mode, -1, 0],
            output_predictions[num_mode, -1, 1],
            "*",
            color="#008000",
            label="Output",
            alpha=transparency,
            linewidth=3,
            zorder=11,
            markersize=9,
        )
        
        if prediction_type == "multimodal":
            # Confidences
            plt.text(
                    output_predictions[num_mode, -1, 0] + 1,
                    output_predictions[num_mode, -1, 1] + 1,
                    str(round(output_confidences[num_mode],2)),
                    zorder=12,
                    fontsize=9,
                    clip_on=True
                    )