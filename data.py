

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
from typing import Final, List, Optional, Sequence, Set, Tuple
import copy
from pathlib import Path
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet,serialize_argoverse_scenario_parquet
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, ObjectType, Track, TrackCategory
# from map_api import ArgoverseStaticMap
from av2.map.map_api import ArgoverseStaticMap 
from av2.map.lane_segment import *
from lane_segment import *
import time

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}

class ArgoDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train
        if 'preprocess' in config and config['preprocess']:
            if train:
                start = time.time()
                self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
                # self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
                end = time.time()
                # print(f"Load VAL, not train data: {end-start}")
                print(f"Load train data: {end-start}")
            else:
                start = time.time()
                self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
                end = time.time()
                print(f"Load val data: {end-start}")
        else:
            self.dataroot = split
            self.filenames = sorted(os.listdir(self.dataroot))
            self.num_files = len(self.filenames)
            
    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]
            new_data = dict()
            for key in ['orig', 'gt_preds', 'object_types', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                if key in data:
                    new_data[key] = ref_copy(data[key])
            data = new_data
        else:
            data = self.get_track_feats(idx)
            data['idx'] = idx
            data['graph'] = self.get_lane_graph(data)
        return data
      
    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return self.num_files
           
    def get_track_feats(self, idx):
        file_id = self.filenames[idx]
        FILENAME = os.path.join(self.dataroot, file_id, "scenario_"+str(file_id)+".parquet")
        scenario = load_argoverse_scenario_parquet(FILENAME)
        
        trajs, steps, track_ids, object_types = [], [], [], []
        for track in scenario.tracks:
            if track.category == TrackCategory.FOCAL_TRACK:
                #agent_traj_vel: NDArrayFloat = np.array(
                #        [list(object_state.position) + list(object_state.velocity) for object_state in track.object_states])
                agent_traj: NDArrayFloat = np.array([list(object_state.position) for object_state in track.object_states])
                agent_steps = np.array([object_state.timestep for object_state in track.object_states], np.int64)
                agent_track_id = track.track_id
                agent_type = self.get_object_type(track.object_type)
            else:
                if track.object_type in _STATIC_OBJECT_TYPES:
                    continue
                #if track.category == TrackCategory.TRACK_FRAGMENT:
                #    continue
                #actor_traj_vel: NDArrayFloat = np.array(
                #        [list(object_state.position) + list(object_state.velocity) for object_state in track.object_states)
                actor_traj: NDArrayFloat = np.array([list(object_state.position) for object_state in track.object_states])
                actor_steps = np.array([object_state.timestep for object_state in track.object_states], np.int64)
                trajs.append(actor_traj)
                steps.append(actor_steps)
                track_ids.append(track.track_id)
                object_types.append(self.get_object_type(track.object_type))
                
        trajs = [agent_traj] + trajs
        steps = [agent_steps] + steps
        track_ids = [agent_track_id] + track_ids
        object_types = [agent_type] + object_types
        
        current_step_index = steps[0].tolist().index(49)
        pre_current_step_index = current_step_index-1
        orig = trajs[0][current_step_index][:2].copy().astype(np.float32)
        pre = trajs[0][pre_current_step_index][:2] - orig
        theta = np.pi - np.arctan2(pre[1], pre[0])
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)
        
        feats, ctrs, gt_preds, has_preds, valid_track_ids, valid_object_types = [], [], [], [], [], []
        for traj, step, track_id, object_type in zip(trajs, steps, track_ids, object_types):
            if 49 not in step:
                continue
            valid_track_ids.append(track_id)
            valid_object_types.append(object_type)
            gt_pred = np.zeros((60, 2), np.float32)
            has_pred = np.zeros(60, bool)
            future_mask = np.logical_and(step >= 50, step < 110)
            future_step = step[future_mask] - 50
            future_traj = traj[future_mask]
            gt_pred[future_step] = future_traj[:,:2]
            has_pred[future_step] = 1
            
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
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)
            
        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, bool)
        data = dict()
        data['scenario_id'] = scenario.scenario_id
        data['track_ids'] = valid_track_ids
        data['object_types'] = np.asarray(valid_object_types, np.float32)
        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        return data
    
    def get_lane_graph(self, data):
        idx = data['idx']
        file_id = self.filenames[idx]
        map_path = os.path.join(self.dataroot, file_id, "log_map_archive_"+str(file_id)+".json")
        avm = ArgoverseStaticMap.from_json(Path(map_path))
        lane_ids = avm.get_scenario_lane_segment_ids()
        
        centerlines = dict()
        ctrs, feats, lane_type, left_mark_type, right_mark_type, intersect, widths = [], [], [], [], [], [], []
        for lane_id in lane_ids:
            centerline, width = avm.get_lane_segment_centerline_and_width(lane_id)
            ctrln = np.matmul(data['rot'], (centerline[:,:2] - data['orig'].reshape(-1, 2)).T).T
            centerlines[lane_id] = ctrln
            
            num_segs = len(ctrln) - 1
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))
            
            intersect.append(avm.lane_is_in_intersection(lane_id) * np.ones(num_segs, np.float32))
            widths.append(width * np.ones(num_segs, np.float32))
            x = np.zeros((num_segs, 2), np.float32)
            
            lane = avm.vector_lane_segments[lane_id]
            if lane.lane_type == LaneType.VEHICLE:
                x[:, :] = 1
            elif lane.lane_type == LaneType.BUS:
                x[:, 0] = 1
                x[:, 1] = 0
            elif lane.lane_type == LaneType.BIKE:
                x[:, 0] = 0
                x[:, 1] = 1
            else:
                x[:, :] = 0
            lane_type.append(x)
            
            x = np.zeros((num_segs, 4), np.float32)
            left_x = self.get_mark_type(x, lane.left_mark_type)
            left_mark_type.append(left_x)
            
            x = np.zeros((num_segs, 4), np.float32)
            right_x = self.get_mark_type(x, lane.right_mark_type)
            right_mark_type.append(right_x)
            
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count
        
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []   
        
        #predecessors are implicit and available by reversing the directed graph dictated by successors.
        predecessors = dict()
        for i, lane_id in enumerate(lane_ids):
            predecessor = []
            for j, nbr_id in enumerate(lane_ids):
                if lane_id in avm.get_lane_segment_successor_ids(nbr_id):
                    predecessor.append(nbr_id)
            predecessors[lane_id] = predecessor  
        
        for i, lane_id in enumerate(lane_ids):
            idcs = node_idcs[i]
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if len(predecessors[lane_id]) > 0:
                for nbr_id in predecessors[lane_id]:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])
            
            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if len(avm.get_lane_segment_successor_ids(lane_id)) > 0:
                for nbr_id in avm.get_lane_segment_successor_ids(lane_id):
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])
        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)
        
        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            nbr_ids = predecessors[lane_id]
            if len(nbr_ids) > 0:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])
                        
            nbr_ids = avm.get_lane_segment_successor_ids(lane_id)
            if len(nbr_ids) > 0:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = avm.get_lane_segment_left_neighbor_id(lane_id)
            if len(nbr_ids) > 0:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = avm.get_lane_segment_right_neighbor_id(lane_id)
            if len(nbr_ids) > 0:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)        
        
        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['lane_type'] = np.concatenate(lane_type, 0)
        graph['left_mark_type'] = np.concatenate(left_mark_type, 0)
        graph['right_mark_type'] = np.concatenate(right_mark_type, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['width'] = np.concatenate(widths, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['node_idcs'] = node_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        
        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)
        
        for key in ['pre', 'suc']:
            if 'scales' in self.config and self.config['scales']:
                #TODO: delete here
                graph[key] += dilated_nbrs2(graph[key][0], graph['num_nodes'], self.config['scales'])
            else:
                graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])
        return graph
    
    def get_object_type(self, object_type):
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
            
    def get_mark_type(self, x, mark_type):
        # none = 0, sigle_line = 1, double_line =2; dash = 0, solid =1; dash = 0, solid =1; white =0, yellow = 1, blue =2; 
        if mark_type == LaneMarkType.DASH_SOLID_YELLOW:
            x[:, 0] = 2
            x[:, 1] = 0
            x[:, 2] = 1
            x[:, 3] = 1
        elif mark_type == LaneMarkType.DASH_SOLID_WHITE:
            x[:, 0] = 2
            x[:, 1] = 0
            x[:, 2] = 1
            x[:, 3] = 0
        elif mark_type == LaneMarkType.DASHED_WHITE:   
            x[:, 0] = 1
            x[:, 1] = 0
            x[:, 2] = 0
            x[:, 3] = 0
        elif mark_type == LaneMarkType.DASHED_YELLOW:   
            x[:, 0] = 1
            x[:, 1] = 0
            x[:, 2] = 0  
            x[:, 3] = 1
        elif mark_type == LaneMarkType.DOUBLE_SOLID_YELLOW:
            x[:, 0] = 2
            x[:, 1] = 1
            x[:, 2] = 1
            x[:, 3] = 1
        elif mark_type == LaneMarkType.DOUBLE_SOLID_WHITE:   
            x[:, 0] = 2
            x[:, 1] = 1
            x[:, 2] = 1
            x[:, 3] = 0
        elif mark_type == LaneMarkType.DOUBLE_DASH_YELLOW:   
            x[:, 0] = 2
            x[:, 1] = 0
            x[:, 2] = 0  
            x[:, 3] = 1            
        elif mark_type == LaneMarkType.DOUBLE_DASH_WHITE:
            x[:, 0] = 2
            x[:, 1] = 0
            x[:, 2] = 0
            x[:, 3] = 0
        elif mark_type == LaneMarkType.SOLID_YELLOW:   
            x[:, 0] = 1
            x[:, 1] = 1
            x[:, 2] = 1
            x[:, 3] = 1
        elif mark_type == LaneMarkType.SOLID_WHITE:   
            x[:, 0] = 1
            x[:, 1] = 1
            x[:, 2] = 1  
            x[:, 3] = 0
        elif mark_type == LaneMarkType.SOLID_DASH_WHITE:
            x[:, 0] = 2
            x[:, 1] = 1
            x[:, 2] = 0
            x[:, 3] = 0
        elif mark_type == LaneMarkType.SOLID_DASH_YELLOW:   
            x[:, 0] = 2
            x[:, 1] = 1
            x[:, 2] = 0
            x[:, 3] = 1
        elif mark_type == LaneMarkType.SOLID_BLUE:   
            x[:, 0] = 1
            x[:, 1] = 1
            x[:, 2] = 1  
            x[:, 3] = 2 
        elif mark_type == LaneMarkType.UNKNOWN:
            x[:, 0] = 1
            x[:, 1] = 0
            x[:, 2] = 0  
            x[:, 3] = 0 
        elif mark_type == LaneMarkType.NONE:   
            x[:, 0] = 0
            x[:, 1] = 0
            x[:, 2] = 0  
            x[:, 3] = 0 
        return x
    
class ArgoTestDataset(ArgoDataset):
    def __init__(self, split, config, train=False):
        self.config = config
        self.train = train
        split2 = config['val_split'] if split=='val' else config['test_split']
        split = self.config['preprocess_val'] if split=='val' else self.config['preprocess_test']
        
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(split, allow_pickle=True)
            else:
                self.split = np.load(split, allow_pickle=True)
        else:
            self.dataroot = split2
            self.filenames = sorted(os.listdir(self.dataroot))
            self.num_files = len(self.filenames)
            
    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]
            new_data = dict()
            for key in ['scenario_id', 'track_ids', 'object_types', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                if key in data:
                    new_data[key] = ref_copy(data[key])
            data = new_data
            return data

        data = self.get_obj_feats(idx)
        data['graph'] = self.get_lane_graph(data)
        data['idx'] = idx
        return data
    
    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return self.num_files 
    
def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs

def dilated_nbrs2(nbr, num_nodes, scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, max(scales)):
        mat = mat * csr

        if i + 1 in scales:
            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int64)
            nbr['v'] = coo.col.astype(np.int64)
            nbrs.append(nbr)
    return nbrs

def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch

if __name__ == "__main__":     
    config = dict()                               
#    config["argo2_train_split"] = "/home/hadoop-wallepnc-hulk/cephfs/data/wangmingkun/data/argoverse2/val"
    config["argo2_train_split"] = "/Users/wangmingkun/Desktop/workspace/datasets/argoverse2/test_data"
    config["num_scales"] = 6
    argo2_data = ArgoDataset(1, config)    
    argo2_data.__getitem__(0)
