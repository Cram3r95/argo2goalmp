#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 17:55:12 2023
@author: Carlos Gómez-Huélamo
"""

import sys
import argparse
import yaml
import pdb
import git

from collections import ChainMap

import torch
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from inference_pipeline import inference_pipeline
from kd_training import KnowledgeDistillationTraining

# Custom imports

repo = git.Repo('.', search_parent_directories=True)
BASE_DIR = repo.working_tree_dir
sys.path.append(BASE_DIR)

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.TFMF_TGR_teacher import TMFModel as TeacherModel
from model.TFMF_TGR_student import TMFModel as StudentModel

parser = argparse.ArgumentParser()
parser = TeacherModel.init_args(parser)

parser.add_argument("--teacher_ckpt_path", type=str, default="/path/to/checkpoint.ckpt")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

def get_data_for_kd_training(batch):
    data = torch.cat([sample[0] for sample in batch], dim=0)
    data = data.unsqueeze(1)
    return data,

def main():
    args = parser.parse_args()
    
    config = yaml.load(open('./config.yaml','r'), Loader=yaml.FullLoader)
    gpu_id = config["knowledge_distillation"]["general"]["devices"][0] # We assume here we are using single GPU
    device = torch.device(f"cuda:{gpu_id}")

    # Create data loaders for training and validation
    
    dataset = ArgoCSVDataset(args.train_split, args.train_split_pre_social, args, input_preprocessed_map=args.train_split_pre_map)
    train_data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=False, # n multi-process loading, the drop_last argument drops the last non-full batch of each worker’s iterable-style dataset replica.
        shuffle=True
    )
    
    dataset = ArgoCSVDataset(args.val_split, args.val_split_pre_social, args, input_preprocessed_map=args.val_split_pre_map)
    val_data_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers, # PyTorch provides an easy switch to perform multi-process data loading
        collate_fn=collate_fn_dict, # A custom collate_fn can be used to customize collation, convert the list of dictionaries to the dictionary of lists 
        pin_memory=True # For data loading, passing pin_memory=True to a DataLoader will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
    )
    val_data_loaders = {"argo2_val": val_data_loader}

    # Create inference pipeline for validating the student model
    
    inference_pipeline_example = inference_pipeline(device)

    # Create student and teacher model
    
    student_model = StudentModel(args)
    teacher_model = TeacherModel(args)
    checkpoint = torch.load(args.teacher_ckpt_path)
    teacher_model.load_state_dict(checkpoint["state_dict"])
    teacher_model.eval() # We do not want to train during inference

    # Train a student model with knowledge distillation and get its performance on dev set

    KD_motion_prediction = KnowledgeDistillationTraining(train_data_loader = train_data_loader,
                                                val_data_loaders = val_data_loaders,
                                                inference_pipeline = inference_pipeline_example,
                                                student_model = student_model,
                                                teacher_model = teacher_model,
                                                devices = config["knowledge_distillation"]["general"]["devices"],
                                                temperature = config["knowledge_distillation"]["general"]["temperature"],
                                                final_loss_coeff_dict = config["knowledge_distillation"]["final_loss_coeff"],
                                                logging_param = ChainMap(config["knowledge_distillation"]["general"],
                                                                         config["knowledge_distillation"]["optimization"],
                                                                         config["knowledge_distillation"]["final_loss_coeff"],
                                                                         config["knowledge_distillation"]["pytorch_lightning_trainer"]),
                                                **ChainMap(config["knowledge_distillation"]["optimization"],
                                                           config["knowledge_distillation"]["pytorch_lightning_trainer"],
                                                           config["knowledge_distillation"]["comet_info"])
                                               )
    KD_motion_prediction.start_kd_training()
    student_model = KD_motion_prediction.get_student_model()
    
    # Save student model
    
    pdb.set_trace()
if __name__ == "__main__":
    main()
    
"""
python motion_prediction_distillation.py --use_preprocessed=True \
                                         --teacher_ckpt_path="/home/robesafe/Argoverse2_Motion_Forecasting/lightning_logs/version_9/checkpoints/epoch=9-loss_train=127.62-loss_val=173.21-ade1_val=2.93-fde1_val=7.12-ade_val=2.94-fde_val=6.79.ckpt"
"""