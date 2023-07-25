#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon July 03 17:46:19 2023
@author: Carlos Gómez Huélamo

Scenarios of interest: 

<route id="25" town="Town05"> Test
"""

# General purpose imports

import argparse
import os
import git
import sys
import pdb
import os
import csv

from importlib import import_module
from types import SimpleNamespace

# DL & Math imports

import torch
import numpy as np

# ROS imports

import rospy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Custom imports

from av2.datasets.motion_forecasting.data_schema import ObjectType

repo = git.Repo(__file__, search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from model.utils.utils import load_pretrain
from preprocess.data import from_numpy, get_object_type

# Global variables

#######################################

# Main class

class Motion_Predictor():
    def __init__(self, get_model=False):
        """
        """
        
        self.NUM_MODES = 6
        self.OBS_LEN = 50
        self.PRED_LEN = 60
        self.required_variables = 5 # id, obs_num, x, y, padding
        
        self.VISUALIZE_EGO_TRAJECTORY = False
        self.PREDICTION_TYPE = "multimodal"
        self.TINY_VISUALIZATION = False # Represent only few future steps for visualization purposes
        self.THRESHOLD_NUM_OBSERVATIONS = 5 # Minimum number of observations out of self.OBS_LEN (e.g. 20 out of 50),
                                  # to start predicting an agent
        self.NUM_STEPS = 10 # To obtain predictions every n-th STEP
        self.NUM_PREDICTED_POSES = 4 # e.g. t+0, t+STEP, t+2*STEP, t+3*STEP
        
        if get_model: self.prediction_network = self.get_prediction_model()

        self.CONFIDENCE_THRESHOLD = 0.2 # The mode must have at least 0.2 out of 1.0 to be plotted
        self.pub_predictions_marker = rospy.Publisher("/t4ac/perception/prediction/prediction_markers", MarkerArray, queue_size=10)
        
    def get_prediction_model(self):
        """
        """

        # Motion Prediction network configuration (Note, since this script
        # is an auxiliary module, we avoid the use of ArgParse. Its use is preferred
        # for main scripts, not for imported modules)

        args = {}
        args["model"] = "CGHNet"
        args["exp_name"] = "results_student"
        args["weight_dir"] = "stable_ckpts"
        args["ckpt"] = "50.000.ckpt"
        args["use_map"] = False
        args["use_goal_areas"] = False
        args["map_in_decoder"] = False
        args["motion_refinement"] = False
        args["distill"] = False

        # Transform from dictionary to dot notation

        args = SimpleNamespace(**args)

        model = import_module("model.models.%s" % args.model)
        config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model(exp_name=args.exp_name,
                                                                                    distill=args.distill,
                                                                                    use_map=args.use_map,
                                                                                    use_goal_areas=args.use_goal_areas,
                                                                                    map_in_decoder=args.map_in_decoder)

        # Load pretrained model

        ckpt_path = os.path.join(BASE_DIR, args.weight_dir,
                                args.exp_name, args.ckpt)

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(net, ckpt["state_dict"])
        net.eval()

        return net

    def preprocess_trackers(self, trajectories):
        """
        """
        
        agents_info_array = np.zeros([0, self.required_variables])

        for key, value in trajectories.items():
            for num_obs in range(self.OBS_LEN):
                agent_info = np.array([key,
                                        num_obs,
                                        value[num_obs,0],
                                        value[num_obs,1],
                                        value[num_obs,2]])

                agents_info_array = np.vstack((agents_info_array, agent_info))

        # Avoid storing full-padded agents
        
        agents_id = np.unique(agents_info_array[:, 0], axis=0)

        valid_agents_info = []
        valid_agents_id = []
        
        for agent_index, agent_id in enumerate(agents_id):
            agent_info = agents_info_array[agents_info_array[:, 0] == agent_id]

            # if not (agent_info[:, -1] == 0).all():  # Avoid storing full-padded agents
            if np.sum(agent_info[:,-1] == 1) >= self.THRESHOLD_NUM_OBSERVATIONS: # Only consider those agents that have 
                                                                                 # at least self.THRESHOLD_STEPS of observations
                valid_agents_info.append(agent_info)
                valid_agents_id.append(int(agent_info[0,0]))
                      
        return valid_agents_info, valid_agents_id

    def predict_agents(self, valid_agents_info, file_id):
        """
        """
             
        # Get agents of the scene (we assume the first one represents our ego-vehicle)
        # (N agents * 50) x 5 (track_id, timestep, x, y, padding)
        # Preprocess agents (relative displacements, orientation, etc.)

        trajs, steps, track_ids, object_types = [], [], [], []
        final_predictions, final_confidences = [], []

        # TODO: Write here the corresponding object type, though at this moment it is not used
        
        object_type = ObjectType.VEHICLE

        # agent_info: 0 = track_id, 1 = timestep, 2 = x, 3 = y, 4 = padding

        for agent_info in valid_agents_info:
            non_padded_agent_info = agent_info[agent_info[:, -1] == 1]
            
            trajs.append(non_padded_agent_info[:, 2:4])
            steps.append(non_padded_agent_info[:, 1].astype(np.int64))
            
            track_ids.append(non_padded_agent_info[0, 0])
            object_types.append(get_object_type(object_type))

        # Our ego-vehicle is always the first agent of the scenario
        
        if len(trajs) > 0 and trajs[0].shape[0] > 1:

            current_step_index = steps[0].tolist().index(self.OBS_LEN-1)
            pre_current_step_index = current_step_index-1

            orig = trajs[0][current_step_index][:2].copy().astype(np.float32)
            
            curr_pos = trajs[0][current_step_index][:2]
            pre_pos = trajs[0][pre_current_step_index][:2]
            theta = np.arctan2(curr_pos[1] - pre_pos[1], 
                               curr_pos[0] - pre_pos[0])
            
            rot = np.asarray([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], np.float32)

            feats, ctrs, valid_track_ids, valid_object_types = [], [], [], []

            for traj, step, track_id, object_type in zip(trajs, steps, track_ids, object_types):

                if self.OBS_LEN-1 not in step:

                    continue

                valid_track_ids.append(track_id)
                valid_object_types.append(object_type)

                obs_mask = step < self.OBS_LEN
                
                step = step[obs_mask]
                traj = traj[obs_mask]
                idcs = step.argsort()
                step = step[idcs]
                traj = traj[idcs]
                feat = np.zeros((self.OBS_LEN, 3), np.float32)

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

            # Recursively transform numpy.ndarray to torch.Tensor
            data = from_numpy(data)

            output = self.prediction_network(data)

            for agent_index in range(feats.shape[0]):
                agent_mm_pred = output["reg"][0][agent_index,:,:,:].cpu().data.numpy()
                agent_cls = output["cls"][0][agent_index,:].cpu().data.numpy()

                if self.PREDICTION_TYPE == "unimodal": # Unimodal prediction (60 x 2)
                    most_probable_mode_index = np.argmax(agent_cls)
                    agent_um_pred = agent_mm_pred[most_probable_mode_index, :, :]
                    agent_um_pred = agent_um_pred[::self.NUM_STEPS,:][:self.NUM_PREDICTED_POSES, :]
                    final_predictions.append(agent_um_pred)

                else:
                    final_predictions.append(agent_mm_pred)
                    final_confidences.append(agent_cls)

        return final_predictions, final_confidences
                        
    def plot_predictions_ros_markers(self, predictions, confidences, valid_agents_id, timestamp, colours_palette, apply_colour=False, lifetime=0.5):
        """
        """
        
        lifetime = 0.2
        
        predictions_markers_list = MarkerArray()
        
        assert self.NUM_MODES == len(confidences[0]) 
        
        slope = (1 - 1/self.NUM_MODES) / (self.NUM_MODES - 1)
            
        for num_agent, agent_predictions in enumerate(predictions):
            # Avoid plotting the predictions of the ego-vehicle
            
            if num_agent == 0 and not self.VISUALIZE_EGO_TRAJECTORY: 
                continue 
            
            sorted_confidences = np.sort(confidences[num_agent])
            
            for num_mode in range(agent_predictions.shape[0]):
                if confidences[num_agent][num_mode] < self.CONFIDENCE_THRESHOLD:
                    continue
                
                agent_predictions_marker = Marker()
                agent_predictions_marker.header.frame_id = "/map"
                agent_predictions_marker.header.stamp = timestamp
            
                agent_predictions_marker.ns = f"agent_{valid_agents_id[num_agent]}_predictions"
                
                agent_predictions_marker.action = agent_predictions_marker.ADD
                agent_predictions_marker.lifetime = rospy.Duration.from_sec(lifetime)

                agent_predictions_marker.id = num_agent * agent_predictions.shape[0] + num_mode
                agent_predictions_marker.type = Marker.LINE_STRIP
                # agent_predictions_marker.type = Marker.POINTS
                
                conf_index = np.where(confidences[num_agent][num_mode] == sorted_confidences)[0].item()
                agent_predictions_marker.color.a = round(slope * (conf_index + 1),2)
                
                if apply_colour:
                    colour = colours_palette[valid_agents_id[num_agent]%colours_palette.shape[0]]
                    agent_predictions_marker.color.r = colour[0]
                    agent_predictions_marker.color.g = colour[1]
                    agent_predictions_marker.color.b = colour[2]
                else: # Green
                    agent_predictions_marker.color.r = 0.2
                    agent_predictions_marker.color.g = 0.4
                    agent_predictions_marker.color.b = 0.0
                
                agent_predictions_marker.scale.x = 0.5
                agent_predictions_marker.pose.orientation.w = 1.0
                
                assert self.PRED_LEN == agent_predictions.shape[1]
                
                if self.TINY_VISUALIZATION:
                    curr_agent_predictions = agent_predictions[num_mode,::self.NUM_STEPS,:]
                else:
                    curr_agent_predictions = agent_predictions[num_mode,:,:]
                    
                for num_pred in range(curr_agent_predictions.shape[0]):
                    point = Point()

                    point.x = curr_agent_predictions[num_pred,0]
                    point.y = curr_agent_predictions[num_pred,1]
                    point.z = 0

                    agent_predictions_marker.points.append(point)

                predictions_markers_list.markers.append(agent_predictions_marker)
                
                # End-Point
                
                agent_prediction_end_point_marker = Marker()
                agent_prediction_end_point_marker.header.frame_id = "/map"
                agent_prediction_end_point_marker.header.stamp = timestamp
            
                agent_prediction_end_point_marker.ns = f"agent_{valid_agents_id[num_agent]}_predictions_end_points"
                
                agent_prediction_end_point_marker.action = agent_prediction_end_point_marker.ADD
                agent_prediction_end_point_marker.lifetime = rospy.Duration.from_sec(lifetime)

                agent_prediction_end_point_marker.id = num_agent * agent_predictions.shape[0] + num_mode
                agent_prediction_end_point_marker.type = Marker.SPHERE
                
                conf_index = np.where(confidences[num_agent][num_mode] == sorted_confidences)[0].item()
                agent_prediction_end_point_marker.color.a = round(slope * (conf_index + 1),2)
                
                if apply_colour:
                    colour = colours_palette[valid_agents_id[num_agent]%colours_palette.shape[0]]
                    agent_prediction_end_point_marker.color.r = colour[0]
                    agent_prediction_end_point_marker.color.g = colour[1]
                    agent_prediction_end_point_marker.color.b = colour[2]
                else:
                    agent_prediction_end_point_marker.color.r = 0.2
                    agent_prediction_end_point_marker.color.g = 0.4
                    agent_prediction_end_point_marker.color.b = 0.0
                
                agent_prediction_end_point_marker.scale.x = 1.0
                agent_prediction_end_point_marker.scale.y = 1.0
                agent_prediction_end_point_marker.scale.z = 1.0
                
                agent_prediction_end_point_marker.pose.orientation.w = 1.0

                agent_prediction_end_point_marker.pose.position.x = curr_agent_predictions[-1,0]
                agent_prediction_end_point_marker.pose.position.y = curr_agent_predictions[-1,1]
                agent_prediction_end_point_marker.pose.position.z = 0
                
                predictions_markers_list.markers.append(agent_prediction_end_point_marker)
                
                # Confidence text
                
                # agent_prediction_confidence_marker = Marker()
                # agent_prediction_confidence_marker.header.frame_id = "/map"
                # agent_prediction_confidence_marker.header.stamp = timestamp
            
                # agent_prediction_confidence_marker.ns = f"agent_{valid_agents_id[num_agent]}_predictions_confidences"
                
                # agent_prediction_confidence_marker.action = agent_prediction_confidence_marker.ADD
                # agent_prediction_confidence_marker.lifetime = rospy.Duration.from_sec(lifetime)

                # agent_prediction_confidence_marker.id = num_agent * agent_predictions.shape[0] + num_mode
                # agent_prediction_confidence_marker.type = Marker.TEXT_VIEW_FACING
                
                # agent_prediction_confidence_marker.text = str(round(confidences[num_agent][num_mode],2))
                
                # agent_prediction_confidence_marker.color.a = 1.0

                # agent_prediction_confidence_marker.color.r = 0
                # agent_prediction_confidence_marker.color.g = 0
                # agent_prediction_confidence_marker.color.b = 0
                
                # agent_prediction_confidence_marker.scale.z = 2.0
                
                # agent_prediction_confidence_marker.pose.orientation.w = 1.0

                # agent_prediction_confidence_marker.pose.position.x = curr_agent_predictions[-1,0] + 1
                # agent_prediction_confidence_marker.pose.position.y = curr_agent_predictions[-1,1] + 1
                # agent_prediction_confidence_marker.pose.position.z = 0
                
                # predictions_markers_list.markers.append(agent_prediction_confidence_marker)
                
        self.pub_predictions_marker.publish(predictions_markers_list)