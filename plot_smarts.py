#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon May 15 13:47:58 2023
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import pdb
import sys
import pandas as pd
import git
import os
import argparse

from typing import Tuple, Optional, List, Set
from importlib import import_module

# DL & Math imports

import torch
import numpy as np

# Plot imports

import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL import Image

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import utils

from av2.utils.typing import NDArrayFloat
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory

from data import from_numpy

# Global variables

PREDICT_AGENTS = True

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}

OBS_LEN = 50

_PlotBounds = Tuple[float, float, float, float]

RELATIVE_ROUTE = "smarts_simulator_scenarios/scenario_poses_v1"
SCENARIO_ROUTE = os.path.join(BASE_DIR,RELATIVE_ROUTE)
ADDITIONAL_STRING = "poses_"

SAVE_DIR = os.path.join(BASE_DIR,RELATIVE_ROUTE+"_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="CGHNet in Pytorch")
parser.add_argument(
    "-m", "--model", default="CGHNet", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")

parser.add_argument(
    "--weight", default="/home/robesafe/argo2goalmp/stable_ckpts/results_student/50.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "--exp_name", default="results_student", type=str)

parser.add_argument(
    "--use_map", default=False, type=bool)
parser.add_argument(
    "--use_goal_areas", default=False, type=bool)
parser.add_argument(
    "--map_in_decoder", default=False, type=bool)
parser.add_argument(
    "--motion_refinement", default=False, type=bool)
parser.add_argument(
    "--distill", default=False, type=bool)

#######################################

def plot_actor_tracks(ax: plt.Axes, 
                      agents_info: List[NDArrayFloat]) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.


    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    
    for agent_info in agents_info:
        agent_id = int(agent_info[0,0])
        agent_trajectory = agent_info[agent_info[:,-1] == 0][:,2:4] # Filter agent to avoid plotting padded information
        
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
                      purpose="multimodal",
                      *,
                      style: str = "-",
                      line_width: float = 1.0,
                      alpha: float = 1.0,
                      color: str = "r",):
    
    num_modes = output_predictions.shape[0]
    
    if purpose == "multimodal":
        sorted_confidences = np.sort(output_confidences)
        slope = (1 - 1/num_modes) / (num_modes - 1)
    
    for num_mode in range(num_modes):
        
        if purpose == "multimodal":
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
        
        if purpose == "multimodal":
            # Confidences
            plt.text(
                    output_predictions[num_mode, -1, 0] + 1,
                    output_predictions[num_mode, -1, 1] + 1,
                    str(round(output_confidences[num_mode],2)),
                    zorder=12,
                    fontsize=9,
                    clip_on=True
                    )

def get_object_type(object_type):
        x = np.zeros(3, np.float32)
        if object_type == ObjectType.STATIC or object_type == ObjectType.BACKGROUND or object_type == ObjectType.CONSTRUCTION or object_type == ObjectType.RIDERLESS_BICYCLE:
            x[:] = 0
        elif object_type == ObjectType.PEDESTRIAN:
            x[2] = 1
        elif object_type == ObjectType.CYCLIST:
            x[1] = 1
        elif object_type == ObjectType.MOTORCYCLIST:
            x[1] = 1
            x[2] = 1
        elif object_type == ObjectType.BUS:
            x[0] = 1
        elif object_type == ObjectType.VEHICLE:
            x[0] = 1
            x[2] = 1
        elif object_type == ObjectType.UNKNOWN:
            x[0] = 1
            x[1] = 1
            x[2] = 1
        return x
                  
def main():
    """
    """
    
    if PREDICT_AGENTS:
        # Import all settings for experiment
        
        args = parser.parse_args()
        model = import_module("src.%s" % args.model)
        config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model(exp_name=args.exp_name,
                                                                                    distill=args.distill,
                                                                                    use_map=args.use_map,
                                                                                    use_goal_areas=args.use_goal_areas,
                                                                                    map_in_decoder=args.map_in_decoder)
        
        ## Load pretrained model
        
        ckpt_path = args.weight
        
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
            
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        utils.load_pretrain(net, ckpt["state_dict"])
        net.eval()
    
    # Get files from Smarts simulator and sort
    
    files, num_files = utils.load_list_from_folder(SCENARIO_ROUTE)
    file_id_list, _ = utils.get_sorted_file_id_list(files,additional_string=ADDITIONAL_STRING) 

    for file_id in file_id_list:
        
        ## Create figure
                        
        fig = plt.figure(1, figsize=(8, 7))
        ax = fig.add_subplot(111)
        
        ## Read .csv file
        
        csv_file = os.path.join(SCENARIO_ROUTE, ADDITIONAL_STRING+str(file_id)+".csv")      
        df = pd.read_csv(csv_file, delim_whitespace=True,
                         names=["agent_id", "timestamp", "x", "y", "padding"], header=None)
        
        ## Get agents of the scene (we assume the first one represents our ego-vehicle)
        
        agents_id = df["agent_id"].unique()

        valid_agents_info = []
          
        for agent_id in agents_id:
            # agent_data = df[(df["agent_id"] == agent_id) & (df["padding"] == 0.0)]
            agent_data = df[(df["agent_id"] == agent_id)]

            agent_info = agent_data.to_numpy()
            
            if not (agent_info[:,-1] == 1).all(): # Avoid storing full-padded agents 
                valid_agents_info.append(agent_info)

        plot_actor_tracks(ax, valid_agents_info)   
        
        if PREDICT_AGENTS:
            ## Preprocess agents (relative displacements, orientation, etc.)
            
            trajs, steps, track_ids, object_types = [], [], [], []
            
            object_type = ObjectType.VEHICLE # All agents in the Smarts simulator are assumed to be vehicles
            
            # agent_info: 0 = track_id, 1 = timestep, 2 = x, 3 = y, 4 = padding
            
            for agent_info in valid_agents_info:
                non_padded_agent_info = agent_info[agent_info[:,-1] == 0]

                trajs.append(non_padded_agent_info[:,2:4])
                steps.append(non_padded_agent_info[:,1].astype(np.int64))
                track_ids.append(non_padded_agent_info[0,0]) # Our ego-vehicle is always the first agent of the scenario
                object_types.append(get_object_type(object_type))
            
            if trajs[0].shape[0] > 1: # Our ego-vehicle must have at least two observations
                current_step_index = steps[0].tolist().index(OBS_LEN-1)    
                pre_current_step_index = current_step_index-1 
                    
                orig = trajs[0][current_step_index][:2].copy().astype(np.float32)
                pre = trajs[0][pre_current_step_index][:2] - orig
                theta = np.arctan2(pre[1], pre[0])
        
                rot = np.asarray([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]], np.float32)
                
                feats, ctrs, valid_track_ids, valid_object_types = [], [], [], []
                for traj, step, track_id, object_type in zip(trajs, steps, track_ids, object_types):
                    if OBS_LEN-1 not in step:
                        continue

                    valid_track_ids.append(track_id)
                    valid_object_types.append(object_type)

                    obs_mask = step < 50
                    step = step[obs_mask]
                    traj = traj[obs_mask]
                    idcs = step.argsort()
                    step = step[idcs]
                    traj = traj[idcs]

                    feat = np.zeros((50, 3), np.float32)
                    feat[step, :2] = np.matmul(rot, (traj[:, :2] - orig.reshape(-1, 2)).T).T
                    feat[step, 2] = 1.0
                    
                    ctrs.append(feat[-1, :2].copy())
                    feat[1:, :2] -= feat[:-1, :2]
                    feat[step[0], :2] = 0
                    feats.append(feat)

                feats = np.asarray(feats, np.float32)
                ctrs = np.asarray(ctrs, np.float32)

                data = dict()
                
                # OBS: Our network must receive a list (batch) per value of the dictionary. In this case, we only want
                # to analyze a single scenario, so the values must be introduced as lists of 1 element, indicating 
                # batch_size = 1
                
                data['scenario_id'] = [file_id]
                data['track_ids'] = [valid_track_ids]
                data['object_types'] = [np.asarray(valid_object_types, np.float32)]
                data['feats'] = [feats]
                data['ctrs'] = [ctrs]
                data['orig'] = [orig]
                data['theta'] = [theta]
                data['rot'] = [rot]
                
                data = from_numpy(data) # Recursively transform numpy.ndarray to torch.Tensor
                output = net(data)
  
                for agent_index in range(feats.shape[0]):
                    focal_agent_mm_pred = output["reg"][0][agent_index,:,:,:].cpu().data.numpy()
                    focal_agent_cls = output["cls"][0][agent_index,:].cpu().data.numpy()
                
                    plot_predictions(focal_agent_mm_pred, focal_agent_cls)
    
            else:
                pass

        plt.xlim([-30, 30], auto=False)
        plt.ylim([0, 120], auto=False)
    
        filename = os.path.join(SAVE_DIR, ADDITIONAL_STRING+str(file_id)+".png")
        plt.savefig(filename, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)
        plt.close('all')
    
    # Create .gif

    output_file = os.path.join(SAVE_DIR, "output.gif")
    
    ## Remove previous .gif
    
    if os.path.isfile(output_file): os.remove(output_file, dir_fd = None)
    
    ## Get files and sort
    
    files, num_files = utils.load_list_from_folder(SAVE_DIR)
    file_id_list, _ = utils.get_sorted_file_id_list(files,additional_string=ADDITIONAL_STRING) 
 
    frames = []
    
    for file_id in file_id_list:
        filename = os.path.join(SAVE_DIR, ADDITIONAL_STRING+str(file_id)+".png")
        new_frame = Image.open(filename)
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    
    frames[0].save(output_file, format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)
                             
if __name__ == "__main__":
    main()