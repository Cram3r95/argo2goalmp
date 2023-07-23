#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon May 15 13:47:58 2023
@author: Carlos Gómez-Huélamo
"""

# General purpose imports

import sys
import git
import csv
import os
import copy
import pdb

# DL & Math imports

import numpy as np
import pandas as pd

# from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

# Plot imports

import matplotlib.pyplot as plt

from PIL import Image

# Custom imports

from av2.datasets.motion_forecasting.data_schema import ObjectType

repo = git.Repo(__file__, search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from motion_predictor import Motion_Predictor 
from model.utils.utils import load_list_from_folder, get_sorted_file_id_list 
from evaluate.utils.plot_functions import plot_actor_tracks, plot_polylines, plot_predictions
    
# Global variables

PLOT_SCENARIO = False
PREDICT_AGENTS = True
COMPUTE_METRICS = True
INTERPOLATE_METRICS = True

RELATIVE_ROUTE = "data/datasets/CARLA/scenario_route22_town03_training"
BASE_STRING = "poses"
ADDITIONAL_STRING = "poses_"
OBSERVATIONS_DIR = os.path.join(BASE_DIR,RELATIVE_ROUTE,BASE_STRING)
GROUNDTRUTH_PATH = os.path.join(BASE_DIR,RELATIVE_ROUTE,"groundtruth.csv") 
METRICS_PATH = os.path.join(BASE_DIR,RELATIVE_ROUTE,"metrics_2.pdf") 

PREDICTIONS_DIR = os.path.join(BASE_DIR,RELATIVE_ROUTE,"predictions") # To save .csv
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

if PREDICT_AGENTS: # To save .png
    SAVE_DIR = os.path.join(BASE_DIR,RELATIVE_ROUTE,f"{BASE_STRING}_with_predictions_plots") 
else:
    SAVE_DIR = os.path.join(BASE_DIR,RELATIVE_ROUTE,f"{BASE_STRING}_plots") 
os.makedirs(SAVE_DIR, exist_ok=True)

#######################################

def generate_gif():
    """
    """
    
    # Create .gif

    output_file = os.path.join(SAVE_DIR, "output.gif")
    
    ## Remove previous .gif
    
    if os.path.isfile(output_file): os.remove(output_file, dir_fd = None)
    
    ## Get files and sort
    
    files, num_files = load_list_from_folder(SAVE_DIR)
    file_id_list, _ = get_sorted_file_id_list(files,additional_string=ADDITIONAL_STRING) 
 
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

def compute_predictions(motion_predictor):
    """
    TODO: Fix data acquisition. The number of the observation should be correct. We cannot store the 48-th and 49th observation of an agent
    if it has been observed in the frame 2 and 104 respectively. Two many steps in the middle!!!
    """
    
    print("Compute predictions ...")
    
    # Get files and sort
    
    files, num_files = load_list_from_folder(OBSERVATIONS_DIR)
    file_id_list, _ = get_sorted_file_id_list(files,additional_string=ADDITIONAL_STRING) 

    gt = np.zeros(())
    try:
        gt = pd.read_csv(GROUNDTRUTH_PATH, sep=" ", header=None).to_numpy()
    except:
        print("GT not ready yet. Please, generate it using the corresponding function.")
    
    ade_k_1 = [] # Metrics over time in the traffic scenario
    ade_k_6 = []
    fde_k_1 = []
    fde_k_6 = []
    valid_file_id_pred = []
       
    for file_id in file_id_list: 
        ## Read .csv file
        
        csv_file = os.path.join(OBSERVATIONS_DIR, ADDITIONAL_STRING+str(file_id)+".csv")      
        df = pd.read_csv(csv_file, delim_whitespace=True,
                         names=["agent_id", "timestamp", "x", "y", "padding"], header=None)
        
        ## Get agents of the scene (we assume the first one represents our ego-vehicle)
                
        agents_id = df["agent_id"].unique()

        valid_agents_info = []
        valid_agents_id = []

        for agent_id in agents_id:
            agent_data = df[(df["agent_id"] == agent_id)]
            agent_info = agent_data.to_numpy()

            if not (agent_info[:, -1] == 0).all():  # Avoid storing full-padded agents
            # if np.sum(agent_info[:,-1] == 1) >= motion_predictor.THRESHOLD_NUM_OBSERVATIONS: # Only consider those agents that have 
                                                                                 # at least self.THRESHOLD_STEPS of observations
                valid_agents_info.append(agent_info)
                valid_agents_id.append(int(agent_info[0,0]))

        origin = valid_agents_info[0][-1,2:4]
        ego_vehicle_id = valid_agents_info[0][0,0]
        if PLOT_SCENARIO: 
            ## Create figure
                        
            fig = plt.figure(1, figsize=(8, 7))
            ax = fig.add_subplot(111)
            plot_actor_tracks(ax, valid_agents_info, ego_vehicle_id)   
              
        if PREDICT_AGENTS:
            # Predict agents
                            
            predictions, confidences = motion_predictor.predict_agents(valid_agents_info, file_id)

            if PLOT_SCENARIO:
                offset = 150
                
                x_min = origin[0] - offset / 2
                x_max = origin[0] + offset / 2
                y_min = origin[1] - offset / 2
                y_max = origin[1] + offset / 2
                
                plt.xlim([x_min, x_max], auto=False)
                plt.ylim([y_min, y_max], auto=False)
                        
                filename = os.path.join(SAVE_DIR, ADDITIONAL_STRING+str(file_id)+".png")
                plt.savefig(filename, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)
                plt.close('all')
            
            if COMPUTE_METRICS:
                # Compute error w.r.t. future ground-truth (from this timestamp to pred_len positions in the future)

                if np.any(gt) and len(predictions) > 0:
                    ade_frame_k_1 = [] # Metrics in the current traffic scenario
                    ade_frame_k_6 = []
                    fde_frame_k_1 = []
                    fde_frame_k_6 = []

                    for agent_id_index, agent_id in enumerate(valid_agents_id):
                        agent_predictions = predictions[agent_id_index]
                        
                        gt_curr_agent = np.zeros((motion_predictor.PRED_LEN,3))
                        try:
                            gt_curr_agent_ = gt[gt[:,0] == agent_id][file_id+1:file_id+1+motion_predictor.PRED_LEN]
                        except:
                            gt_curr_agent_ = gt[gt[:,0] == agent_id][file_id+1:]
                        
                        obs_gt_frame = gt_curr_agent_[:,1].astype(np.int32)
                        
                        for num_pred in range(gt_curr_agent_.shape[0]):
                            if file_id + num_pred + 1 in obs_gt_frame:
                                gt_curr_agent[num_pred,0] = gt_curr_agent_[num_pred,2] # x
                                gt_curr_agent[num_pred,1] = gt_curr_agent_[num_pred,3] # y
                                gt_curr_agent[num_pred,2] = 1.0 # The ground-truth exists for this timestamp

                        valid_mask = gt_curr_agent[:,2] == 1.0
                        gt_curr_agent = gt_curr_agent[valid_mask,:-1] # Remove auxiliar padding
                
                        agent_ade_k_1 = 50000
                        agent_ade_k_6 = 50000
                        agent_fde_k_1 = 50000
                        agent_fde_k_6 = 50000
                        
                        best_conf = np.argmax(confidences[agent_id_index])
                        
                        if np.any(valid_mask):
                            for num_mode in range(agent_predictions.shape[0]):
                                curr_agent_prediction = agent_predictions[num_mode,valid_mask,:]

                                l2 = curr_agent_prediction - gt_curr_agent # pred_len x 2
                                l2 = l2**2 # pred_len x 2
                                l2 = np.sum(l2,axis=1) # pred_len
                                l2 = np.sqrt(l2) # pred_len

                                ade = np.mean(l2)
                                fde = l2[-1]
                                
                                if ade < agent_ade_k_6:
                                    agent_ade_k_6 = copy.copy(ade)
                                    
                                if fde < agent_fde_k_6:
                                    agent_fde_k_6 = copy.copy(fde)
                                
                                if num_mode == best_conf:
                                    agent_ade_k_1 = copy.copy(ade)
                                    agent_fde_k_1 = copy.copy(fde)
                            
                            ade_frame_k_1.append(agent_ade_k_1)
                            ade_frame_k_6.append(agent_ade_k_6)
                            fde_frame_k_1.append(agent_fde_k_1)
                            fde_frame_k_6.append(agent_fde_k_6)
                    
                    if ade_frame_k_1: 
                        # print("ade k=1 frame: ", np.array(ade_frame_k_1).mean())
                        ade_k_1.append(np.array(ade_frame_k_1).mean()) 
                    if ade_frame_k_6: 
                        # print("ade k=6 frame: ", np.array(ade_frame_k_6).mean())
                        ade_k_6.append(np.array(ade_frame_k_6).mean())
                    if fde_frame_k_1: 
                        # print("fde k=1 frame: ", np.array(fde_frame_k_1).mean())
                        fde_k_1.append(np.array(fde_frame_k_1).mean())
                    if fde_frame_k_6: 
                        # print("fde k=6 frame: ", np.array(fde_frame_k_6).mean())
                        fde_k_6.append(np.array(fde_frame_k_6).mean())
                    
                    if ade_frame_k_6: valid_file_id_pred.append(file_id)
    
    if COMPUTE_METRICS: 
        print("Compute final metrics ...")
        
        ade_k_1 = np.array(ade_k_1)
        ade_k_6 = np.array(ade_k_6)
        fde_k_1 = np.array(fde_k_1)
        fde_k_6 = np.array(fde_k_6)
            
        if INTERPOLATE_METRICS:
            x = np.array(valid_file_id_pred)
            X_ = np.linspace(x.min(), x.max(), len(valid_file_id_pred)*5)
            
            X_Y_Spline = make_interp_spline(x, ade_k_1)
            ade_k_1 = X_Y_Spline(X_)
            
            X_Y_Spline = make_interp_spline(x, ade_k_6)
            ade_k_6 = X_Y_Spline(X_)
            
            X_Y_Spline = make_interp_spline(x, fde_k_1)
            fde_k_1 = X_Y_Spline(X_)
            
            X_Y_Spline = make_interp_spline(x, fde_k_6)
            fde_k_6 = X_Y_Spline(X_)
        
        SCENARIO_ID = 1
        
        plt.title(f"Motion Prediction metrics in Scenario {SCENARIO_ID} \n", fontweight="bold", fontsize=15)
        
        plt.suptitle(r'$\overline{minADE} (K=1): $'f'{round(ade_k_1.mean(),2)},'
                     r' $\overline{minFDE} (K=1): $'f'{round(fde_k_1.mean(),2)},'
                     r' $\overline{minADE} (K=6): $'f'{round(ade_k_6.mean(),2)},'
                     r' $\overline{minFDE} (K=6): $'f'{round(fde_k_6.mean(),2)}', fontsize=12, y=0.93)
        
        plt.plot(X_, ade_k_1, label="ADE k=1")
        plt.plot(X_, ade_k_6, label="ADE k=6")
        plt.plot(X_, fde_k_1, label="FDE k=1")
        plt.plot(X_, fde_k_6, label="FDE k=6")
        
        plt.xlabel("Frame")
        plt.ylabel("L2 error")
        plt.legend()
        
        plt.savefig(METRICS_PATH, bbox_inches='tight', facecolor="white", edgecolor='none', pad_inches=0)
        plt.close('all')

def generate_groundtruth(obs_len=50,pred_len=60):
    """
    We need the generate the ground-truth for each particular traffic scenario
    """  
    
    print("Generating ground-truth ...")
    
    # Get files and sort

    files, num_files = load_list_from_folder(OBSERVATIONS_DIR)
    file_id_list, _ = get_sorted_file_id_list(files,additional_string=ADDITIONAL_STRING) 

    dataframes_list = []
    
    for file_id in file_id_list:
        ## Read .csv file
        
        csv_file = os.path.join(OBSERVATIONS_DIR, ADDITIONAL_STRING+str(file_id)+".csv")      
        df = pd.read_csv(csv_file, delim_whitespace=True,
                         names=["agent_id", "timestamp", "x", "y", "padding"], header=None)
        dataframes_list.append(df)
    
    agents_in_scenario = {}
    
    for i, df in enumerate(dataframes_list):
        agents_id = df["agent_id"].unique()
        
        for agent_id in agents_id:
            last_agent_data = np.array(df[(df["agent_id"] == agent_id)])[-1] # last observation in this frame (i.e. update)
            last_agent_data[1] = file_id_list[i] # Substitute the generatic last observation (obs_len - 1) with the
                                                 # actual current timestamp
            if agent_id not in agents_in_scenario.keys():
                agents_in_scenario[agent_id] = [last_agent_data]
            else:
                agents_in_scenario[agent_id].append(last_agent_data)

    # Convert dict values to numpy arrays
    
    for agent_id in agents_in_scenario.keys():
        agents_in_scenario[agent_id] = np.array(agents_in_scenario[agent_id])
        
    # Write in csv

    with open(GROUNDTRUTH_PATH, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for agent_id in agents_in_scenario.keys():
            for k in range(agents_in_scenario[agent_id].shape[0]):
                writer.writerow([agent_id, 
                                    agents_in_scenario[agent_id][k,1], # timestamp
                                    agents_in_scenario[agent_id][k,2], # x
                                    agents_in_scenario[agent_id][k,3]]) 
                
def main():
    """
    OBS: If the agent is padded, the binary flag = 0. Otherwise (i.e. the agent is observed in a 
    particular timestamp), the binary flag = 1
    
    CSV format:
    
    Agent_ID Timestamp X Y Padding
    """
    
    motion_predictor = Motion_Predictor(get_model=PREDICT_AGENTS)
    
    # generate_groundtruth(obs_len=motion_predictor.OBS_LEN,
    #                      pred_len=motion_predictor.PRED_LEN)

    compute_predictions(motion_predictor)
    # if PLOT_SCENARIO: generate_gif()
                           
if __name__ == "__main__":
    main()