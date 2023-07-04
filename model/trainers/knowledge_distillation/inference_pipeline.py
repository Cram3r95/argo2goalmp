#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 17:55:12 2023
@author: Carlos Gómez-Huélamo
"""

import git
import sys
import pdb
import torch
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from data.argoverse.utils.torch_utils import collate_fn_dict

# The inference pipeline is model agnostic, so, it can be run either with 
# the teacher or the student. It is designed to evaluate the model in the 
# validation dataset

class inference_pipeline:

    def __init__(self, device):
        self.device = device

    def run_inference_pipeline(self, model, data_loader):
        
        # Iterate over dataset and generate predictions
    
        predictions = dict()
        gts = dict()
        cities = dict()
        #probabilities = dict()

        for data in tqdm(data_loader):
            data = dict(data)
            for key, value in data.items():
                
                data[key] = [tensor.cuda(self.device) if torch.is_tensor(tensor) else tensor for tensor in data[key]]
        
            with torch.no_grad():
                output = model(data)
            
                output = [x[0:1].detach().cpu().numpy() for x in output]
            for i, (argo_id, prediction) in enumerate(zip(data["argo_id"], output)):
                # prediction.shape : (1,6,60,2) prediction.squeeze().shape(6,60,2)
                predictions[argo_id] = prediction.squeeze()
                # sum_1 = np.sum(prediction.squeeze(),axis=1)
                # sum_2 = np.sum(sum_1,axis=1)
                # sotmax_out = softmax(sum_2) 
                # sum_soft = np.sum(sotmax_out)
                # if sum_soft > 1 :
                #     index_max = np.argmax(sotmax_out, axis=0)
                #     sotmax_out[index_max] = sotmax_out[index_max] - (sum_soft- 1 )
                    
                # if sum_soft < 1:
                #     index_min = np.argmin(sotmax_out, axis=0)
                #     sotmax_out[index_min] = sotmax_out[index_min] + ( 1 - sum_soft )
            
                # probabilities[argo_id] = sotmax_out
                cities[argo_id] = data["city"][i]
                gts[argo_id] = data["gt"][i][0]

        results_6 = compute_forecasting_metrics(predictions, gts, cities, 6, 60, 2) #, probabilities)

        accuracy = results_6['minADE']
        
        return {"inference_result": accuracy}
    
    
