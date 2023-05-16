#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon May 15 13:47:58 2023
@author: Carlos Gómez-Huélamo
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' # '0,1,2,3,4,5,6,7'
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
import sys
import time
import shutil
import pandas as pd
from importlib import import_module
from utils import gpu

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd
import pdb

from torch.utils.data.distributed import DistributedSampler

from utils import Logger, load_pretrain

from mpi4py import MPI

comm = MPI.COMM_WORLD
hvd.init()
torch.cuda.set_device(hvd.local_rank())

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

#######################################################################################

import os
import pandas as pd
from typing import Final, Sequence, Tuple, Union, Optional, Set, List
import click
import matplotlib.pyplot as plt
import numpy as np
import pdb
import git
import math

from torch import Tensor

from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import av2.geometry.polyline_utils as polyline_utils
import av2.geometry.interpolate as interpolate
import av2.rendering.vector as vector_plotting_utils

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneSegment
from av2.utils.typing import NDArrayFloat, NDArrayInt
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType, TrackCategory

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
SAVE_DIR = os.path.join(BASE_DIR,"qualitative_results")
os.makedirs(SAVE_DIR, exist_ok=True)

OBS_LEN = 50
PRED_LEN = 60

# scaled to [0,1] for matplotlib
PURPLE_RGB: Final[Tuple[int, int, int]] = (201, 71, 245)
PURPLE_RGB_MPL: Final[Tuple[float, float, float]] = (PURPLE_RGB[0] / 255, PURPLE_RGB[1] / 255, PURPLE_RGB[2] / 255)

# DARK_GRAY_RGB: Final[Tuple[int, int, int]] = (40, 39, 38)
# DARK_GRAY_RGB: Final[Tuple[int, int, int]] = (100, 99, 98)
DARK_GRAY_RGB: Final[Tuple[int, int, int]] = (200, 199, 198)
DARK_GRAY_RGB_MPL: Final[Tuple[float, float, float]] = (
    DARK_GRAY_RGB[0] / 255,
    DARK_GRAY_RGB[1] / 255,
    DARK_GRAY_RGB[2] / 255,
)

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7

_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_UNSCORED_AGENT_COLOR: Final[str] = "#808080"
_SCORED_AGENT_COLOR: Final[str] = "#9A0EEA"
_FOCAL_AGENT_COLOR: Final[str] = "#F97306"
_AV_COLOR: Final[str] = "#069AF3"
# _FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"
# _AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[int] = 100  # Ensure actor bounding boxes are plotted on top of all map elements

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}

_PlotBounds = Tuple[float, float, float, float]

OVERLAID_MAPS_ALPHA: Final[float] = 0.1

#######################################################################################

parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="GANet", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
# parser.add_argument(
#     "--weight", default="/home/robesafe/GANet-unofficial/exp_agent_gnn_dim_6_latent_128/GANet/50.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
# )
# parser.add_argument(
#     "--exp_name", default="exp_agent_gnn_dim_6_latent_128", type=str)

parser.add_argument(
    "--weight", default="/home/robesafe/GANet-unofficial/results_student/GANet/50.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "--exp_name", default="results_student", type=str)

parser.add_argument(
    "--use_map", default=False, type=bool)
parser.add_argument(
    "--use_goal_areas", default=True, type=bool)
parser.add_argument(
    "--map_in_decoder", default=True, type=bool)
parser.add_argument(
    "--motion_refinement", default=False, type=bool)
parser.add_argument(
    "--distill", default=False, type=bool)

def _plot_actor_tracks(ax: plt.Axes, 
                       scenario: ArgoverseScenario, 
                       timestep: int,
                       plot_gt: bool = False) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Avoid track fragments or static
        if track.category == TrackCategory.TRACK_FRAGMENT or track.object_type in _STATIC_OBJECT_TYPES:
            continue
        
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [object_state.timestep for object_state in track.object_states if object_state.timestep <= timestep]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep <= timestep]
        )
        actor_headings: NDArrayFloat = np.array(
            [object_state.heading for object_state in track.object_states if object_state.timestep <= timestep]
        )

        # Plot polyline for focal agent location history
        ZORDER_GT = 201
        if track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            track_color = _FOCAL_AGENT_COLOR
            _plot_polylines([actor_trajectory], color=track_color, line_width=2, zorder=299) # Past observation
            
            gt_actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep > timestep]
            )
            if plot_gt:
                _plot_polylines([gt_actor_trajectory], color="#FF0000", line_width=2, zorder=300) # Ground-truth
            
        elif track.track_id == "AV":
            track_color = _AV_COLOR
            
            _plot_polylines([actor_trajectory], color=track_color, line_width=2) # Past observation
            
            gt_actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep > timestep]
            )
            if plot_gt:
                _plot_polylines([gt_actor_trajectory], color="#7BC8F6", line_width=2, zorder=ZORDER_GT) # Ground-truth
                
        elif track.category == TrackCategory.UNSCORED_TRACK:
            track_color = _UNSCORED_AGENT_COLOR
            
            _plot_polylines([actor_trajectory], color=track_color, line_width=2) # Past observation
            
            gt_actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep > timestep]
            )
            if plot_gt:
                _plot_polylines([gt_actor_trajectory], color="#C0C0C0", line_width=2, zorder=ZORDER_GT) # Ground-truth
                
        elif track.category == TrackCategory.SCORED_TRACK:
            track_color = _SCORED_AGENT_COLOR
            
            _plot_polylines([actor_trajectory], color=track_color, line_width=2) # Past observation

            gt_actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep > timestep]
            )
            if plot_gt:
                _plot_polylines([gt_actor_trajectory], color="#EE82EE", line_width=2, zorder=ZORDER_GT) # Ground-truth
                
        # Plot bounding boxes for all vehicles and cyclists
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(actor_trajectory[-1, 0], actor_trajectory[-1, 1], "o", color=track_color, markersize=4)

    return track_bounds

def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
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
    for polyline in polylines:
        plt.plot(polyline[:, 0], polyline[:, 1], style, linewidth=line_width, color=color, alpha=alpha, zorder=zorder)

def _plot_predictions(output_predictions, 
                      output_confidences, purpose="multimodal",
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
            zorder=15,
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
            zorder=15,
            markersize=9,
        )
        
        if purpose == "multimodal":
            # Confidences
            plt.text(
                    output_predictions[num_mode, -1, 0] + 1,
                    output_predictions[num_mode, -1, 1] + 1,
                    str(round(output_confidences[num_mode],2)),
                    zorder=400,
                    fontsize=9,
                    # color="#008000",
                    )
            
def _plot_actor_bounding_box(
    ax: plt.Axes, cur_location: NDArrayFloat, heading: float, color: str, bbox_size: Tuple[float, float]
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width, np.degrees(heading), color=color, zorder=_BOUNDING_BOX_ZORDER
    )
    ax.add_patch(vehicle_bounding_box)
    
def _plot_lane_segments(
    ax: Axes, lane_segments: Sequence[LaneSegment], lane_color: Tuple[float, float, float] = DARK_GRAY_RGB_MPL
) -> None:
    """Plot lane segments onto a Matplotlib canvas, according to their lane marking boundary type/color.

    Note: we use an approximation for SOLID_DASHED and other mixed pattern/color marking types.

    Args:
        ax: matplotlib figure to use as drawing canvas.
        lane_segments: lane segment objects. The lane markings along their boundaries will be rendered.
        lane_color: Color of the lane.
    """
    for ls in lane_segments:
        pts_city = ls.polygon_boundary
        ALPHA = 1.0  # 0.1
        vector_plotting_utils.plot_polygon_patch_mpl(
            polygon_pts=pts_city, ax=ax, color=lane_color, alpha=ALPHA, zorder=1
        )

        mark_color: str = ""
        linestyle: Union[str, Tuple[int, Tuple[int, int]]] = ""
        for bound_type, bound_city in zip(
            [ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]
        ):
            if "YELLOW" in bound_type:
                mark_color = "y"
            elif "WHITE" in bound_type:
                mark_color = "w"
            elif "BLUE" in bound_type:
                mark_color = "b"
            else:
                mark_color = "grey"
                # mark_color = "silver"

            LOOSELY_DASHED = (0, (5, 10))

            if "DASHED" in bound_type:
                linestyle = LOOSELY_DASHED
            else:
                linestyle = "solid"

            if "DOUBLE" in bound_type:
                left, right = polyline_utils.get_double_polylines(
                    polyline=bound_city.xyz[:, :2], width_scaling_factor=0.1
                )
                ax.plot(left[:, 0], left[:, 1], color=mark_color, alpha=ALPHA, linestyle=linestyle, zorder=2)
                ax.plot(right[:, 0], right[:, 1], color=mark_color, alpha=ALPHA, linestyle=linestyle, zorder=2)
            else:
                ax.plot(
                    bound_city.xyz[:, 0],
                    bound_city.xyz[:, 1],
                    mark_color,
                    alpha=ALPHA,
                    linestyle=linestyle,
                    zorder=2,
                )
                
def main():
    """
    """
    
    # Import all settings for experiment
    
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model(exp_name=args.exp_name,
                                                                                distill=args.distill,
                                                                                use_map=args.use_map,
                                                                                use_goal_areas=True,
                                                                                map_in_decoder=True)
    
    if config["horovod"]:
        opt.opt = hvd.DistributedOptimizer(
            opt.opt, named_parameters=net.named_parameters()
        )
        
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    # config["epoch"] = ckpt["epoch"]
    # opt.load_state_dict(ckpt["opt_state"])
            
    # Data loader for evaluation
    
    # """Dataset"""
    # data_path = "/home/robesafe/shared_home/benchmarks/argoverse2/motion-forecasting"

    # config["preprocess_val"] = os.path.join(
    #     data_path, "preprocess_c", "val_crs_dist6_angle90_with_scenario_id_tiny.p"
    # )
    split = "val"
    dataset = Dataset(config["val_split"], config, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    PLOT_QUALITATIVE_RESULTS = True
    PLOT_MAP = True
    PLOT_AGENTS = False
    PLOT_PREDICTION = False
    modes = ["input_data"] # "input_data","mm_prediction"
    
    OBS_LEN = 50
    PRED_LEN = 60
    
    for ii, data in tqdm(enumerate(val_loader)):
        data = dict(data)

        with torch.no_grad():
            start = time.time()
            # output = net(data)
            end = time.time()

            print("Inference time: ", end-start)
            if ii == 10:
                pdb.set_trace()

            if PLOT_QUALITATIVE_RESULTS:
                for jj in range(len(data["scenario_id"])):
                    for mode in modes:
                        if mode == "input_data":
                            PLOT_PREDICTION = False
                        elif mode == "mm_prediction":
                            PLOT_PREDICTION = True
                            
                        scenario_id = data["scenario_id"][jj]
                        
                        # Read scenario
                        
                        scenario_path_filename = os.path.join(BASE_DIR,f"dataset/argoverse2/{split}",f"{scenario_id}/scenario_{scenario_id}.parquet")
                        scenario_path = Path(scenario_path_filename)
                        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                        df = pd.read_parquet(scenario_path)
                        target_rows = df.loc[df["track_id"] == scenario.focal_track_id]

                        last_obs_x, last_obs_y = target_rows["position_x"].iloc[OBS_LEN], target_rows["position_y"].iloc[OBS_LEN]
                        query_center: NDArrayFloat = np.array([last_obs_x, last_obs_y])

                        # Static map

                        static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
                        static_map = ArgoverseStaticMap.from_json(static_map_path)
    
                        search_radius = 50 # m
                        nearby_lane_segments = static_map.get_scenario_lane_segments() # All lane segments
                        # nearby_lane_segments = static_map.get_nearby_lane_segments(query_center,search_radius) # Nearby segments

                        # Plots
                        
                        fig = plt.figure(1, figsize=(8, 7))
                        ax = fig.add_subplot(111)

                        ## Map

                        if PLOT_MAP:
                            crosswalk_color = PURPLE_RGB_MPL
                            CROSSWALK_ALPHA = 0.6
                            for pc in static_map.get_scenario_ped_crossings():
                                vector_plotting_utils.plot_polygon_patch_mpl(
                                    polygon_pts=pc.polygon[:, :2],
                                    ax=ax,
                                    color=crosswalk_color,
                                    alpha=CROSSWALK_ALPHA,
                                    zorder=3,
                                )
                                    
                            _plot_lane_segments(ax=ax, lane_segments=nearby_lane_segments) # Nearby lane segments
                        
                        ## Plot past trajectories

                        if PLOT_AGENTS:
                            _plot_actor_tracks(ax, scenario, OBS_LEN, plot_gt=PLOT_PREDICTION)
                        
                        ## Save multimodal prediction (only focal agent) 
                        
                        if PLOT_PREDICTION:
                            AGENT_INDEX = 0
                            focal_agent_mm_pred = output["reg"][jj][AGENT_INDEX,:,:,:].cpu().data.numpy()
                            focal_agent_cls = output["cls"][jj][AGENT_INDEX,:].cpu().data.numpy()
                            
                            _plot_predictions(focal_agent_mm_pred, focal_agent_cls)
                        
                        ## Save plot
                        
                        _PLOT_BOUNDS_BUFFER_M = search_radius
                        plt.xlim(last_obs_x - _PLOT_BOUNDS_BUFFER_M, last_obs_x + _PLOT_BOUNDS_BUFFER_M)
                        plt.ylim(last_obs_y - _PLOT_BOUNDS_BUFFER_M, last_obs_y + _PLOT_BOUNDS_BUFFER_M)
                        plt.axis("off")
                        # plt.title(f"Split = {split} ; Scenario ID = {scenario_id}")                 
                        # plt.gca().set_aspect("equal", adjustable="box")
                        
                        if PLOT_MAP and PLOT_AGENTS and PLOT_PREDICTION: filename_agent = os.path.join(SAVE_DIR,f"{scenario_id}_mm_prediction.pdf") # .png, .pdf
                        elif PLOT_MAP and PLOT_AGENTS: filename_agent = os.path.join(SAVE_DIR,f"{scenario_id}_input_data.pdf") # .png, .pdf
                        elif PLOT_AGENTS: filename_agent = os.path.join(SAVE_DIR,f"{scenario_id}_only_agents_input_data.pdf") # .png, .pdf
                        elif PLOT_MAP: filename_agent = os.path.join(SAVE_DIR,f"{scenario_id}_only_map_input_data.pdf") # .png, .pdf
                        print("filename: ", filename_agent)
                        plt.savefig(filename_agent, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)
                        plt.close('all')
                        
                        # pdb.set_trace()
            
if __name__ == "__main__":
    main()