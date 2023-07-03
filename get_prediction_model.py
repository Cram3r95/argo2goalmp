#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Fri June 30 12:41:19 2023
@author: Carlos Gómez Huélamo
"""

# General purpose imports

import argparse
import os
import git
import sys
import pdb

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

# DL & Math imports

import torch

# Custom imports

repo = git.Repo(__file__, search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

import utils

#######################################


def get_prediction_model():
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

    model = import_module("src.%s" % args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model(exp_name=args.exp_name,
                                                                                distill=args.distill,
                                                                                use_map=args.use_map,
                                                                                use_goal_areas=args.use_goal_areas,
                                                                                map_in_decoder=args.map_in_decoder)

    # Load pretrained model

    ckpt_path = os.path.join(BASE_DIR, args.weight_dir,
                             args.exp_name, args.ckpt)

    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    utils.load_pretrain(net, ckpt["state_dict"])
    net.eval()

    return net
