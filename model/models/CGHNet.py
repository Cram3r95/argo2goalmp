import copy
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6'
import sys
from fractions import gcd
from numbers import Number
import pdb

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data import ArgoDataset, collate_fn
from utils import gpu, to_long,  Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null, no_pad_Res1d
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from scipy.special import softmax

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

### config ###
config = dict()
"""Train"""
config["display_iters"] = 200000
config["val_iters"] = 200000 
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = False
config["opt"] = "adam"
config["num_epochs"] = 50
config["start_val_epoch"] = 30
config["lr"] = [1e-3, 1e-4, 5e-5, 1e-5]
config["lr_epochs"] = [36,42,46]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

config["batch_size"] = 128
config["val_batch_size"] = 128
config["workers"] = 0
config["val_workers"] = config["workers"]

"""Dataset"""
# data_path = "/usr/src/app/datasets/motion_forecasting/argoverse2"
data_path = "/home/robesafe/shared_home/benchmarks/argoverse2/motion-forecasting"

# Raw Dataset
config["train_split"] = os.path.join(data_path, "train")
config["val_split"] = os.path.join(data_path, "val")
config["test_split"] = os.path.join(data_path, "test")

# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(
    data_path, "preprocess_c", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    data_path, "preprocess_c", "val_crs_dist6_angle90.p"
)
config['preprocess_test'] = os.path.join(data_path, 'preprocess_c', 'test_test.p')

"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["actor2map_dist"] = 7.0
config["map2actor_dist"] = 6.0
config["actor2actor_dist"] = 100.0
config["pred_size"] = 60
config["pred_step"] = 1
config["num_test_mod"] = 3
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["cls_coef"] = 3.0
config["reg_coef"] = 1.0
config["end_coef"] = 1.0
config["test_cls_coef"] = 1.0
config["test_coef"] = 0.2
config["test_half_coef"] = 0.1
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2
config["use_map"] = False
### end of config ###

class Net(nn.Module):
    """
    Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations 
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes 
           and lane nodes:
            a. A2M: introduces real-time traffic information to 
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the 
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic 
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction header for motion forecasting using 
           feature from A2A
    """
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.use_map = config["use_map"]
        
        self.actor_net1 = ActorNet(config)
        self.actor_net2 = ActorNet(config)
        if self.use_map:
            self.map_net = MapNet(config)

            self.a2a = A2A(config)
            self.a2m = A2M(config)
            self.m2m = M2M(config)
            self.m2a = M2A(config)
            
            self.test_net = TestNet(config)
            self.m2a_test = M2A(config)
            
            self.test_half_net = TestNet_Half(config)
            self.m2a_half_test = M2A(config)
            
            self.a2a_test = A2A(config)     
            
            self.pred_net = PredNet(config)      
        else:
            self.a2a = A2A(config)
            self.test_half_net = TestNet_Half(config)
            self.pred_net = PredNetNoMap(config)

    def forward(self, data: Dict, return_embeddings: bool = False) -> Dict[str, List[Tensor]]:
        # construct actor feature

        actors, actor_idcs = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        actors_1 = self.actor_net1(actors, actor_ctrs)
        actors_2 = self.actor_net2(actors, actor_ctrs)
        actors = actors_1 + actors_2

        if self.use_map:
            # construct map features
            
            graph = graph_gather(to_long(gpu(data["graph"])))
            nodes, node_idcs, node_ctrs = self.map_net(graph)

            # actor-map fusion cycle 

            nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs) # 8604, 128
            nodes = self.m2m(nodes, graph)
            actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs) # 344, 128
            actors = self.a2a(actors, actor_idcs, actor_ctrs)
            # list 16: x, 2
            test_half_ctrs = self.test_half_net(actors, actor_idcs, actor_ctrs)
            
            actors = self.m2a_half_test(actors, actor_idcs, test_half_ctrs, nodes, node_idcs, node_ctrs)
            #actors = self.a2a_test(actors, actor_idcs, test_half_ctrs)
            test_out = self.test_net(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
            test_ctrs = test_out["test_ctrs"]
            ctrs = [x[:,0] for x in test_ctrs]
            actors = self.m2a_test(actors, actor_idcs, ctrs, nodes, node_idcs, node_ctrs)   
            actors = self.a2a_test(actors, actor_idcs, ctrs)

            # prediction
            
            out, feats = self.pred_net(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        else:
            actors = self.a2a(actors, actor_idcs, actor_ctrs)
            test_half_ctrs = self.test_half_net(actors, actor_idcs, actor_ctrs)
            
            # prediction

            out, feats = self.pred_net(actors, actor_idcs, actor_ctrs)

        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        
        out["test_half_ctrs"] = test_half_ctrs
        if self.use_map:
            out["test_ctrs"] = test_ctrs
            out["test_cls"] = test_out["test_cls"]

        # transform prediction to world coordinates
        for i in range(len(test_half_ctrs)):
            out["test_half_ctrs"][i] = torch.matmul(test_half_ctrs[i], rot[i]) + orig[i].view(
                1, -1
            )
        
        if self.use_map:
            for i in range(len(test_ctrs)):
                out["test_ctrs"][i] = torch.matmul(test_ctrs[i], rot[i]) + orig[i].view(
                    1, 1, -1
                )

        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
           
        if return_embeddings:
            return out, actors, feats # out = multimodal prediction, 
                                      # actors = latent space before regression
                                      # feats = latent space before confidences
        else:
            return out

def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs


def graph_gather(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats", "lane_type", "left_mark_type", "right_mark_type", "intersect"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph

class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, config):
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 32, 64, 128]
        blocks = [no_pad_Res1d, Res1d, Res1d, Res1d]
        num_blocks = [1, 2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)
        
        ctrs_in = 2
        self.lstm_h0_init_function = nn.Linear(ctrs_in, n, bias=False)
        self.lstm_encoder = nn.LSTM(n, n, batch_first=True)
        
        self.output = Res1d(n, n, norm=norm, ng=ng)
        
    def forward(self, actors: Tensor, actor_ctrs) -> Tensor:
        actor_ctrs = torch.cat(actor_ctrs, 0)
        out = actors
        M,d,L = actors.shape

        outputs = []
        for i in range(len(self.groups)):   
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])        
        out = self.output(out)
        out_init = out[:, :, -1]
        
        #1. TODO fuse map data as init hidden and cell state
        h0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, config["n_actor"])
        c0 = self.lstm_h0_init_function(actor_ctrs).view(1, M, config["n_actor"])
        #h0 = torch.zeros(1, M, config["n_actor"]).cuda()
        #c0 = torch.zeros(1, M, config["n_actor"]).cuda()
        out = out.transpose(1, 2).contiguous()
        output, (hn, cn) = self.lstm_encoder(out, (h0, c0))
        out_lstm = hn.contiguous().view(M, config["n_actor"])
        out = out_lstm + out_init
        return out

class MapNet(nn.Module):
    """
    Map Graph feature extractor with LaneGraphCNN
    """
    def __init__(self, config):
        super(MapNet, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        if (
            len(graph["feats"]) == 0
            or len(graph["pre"][-1]["u"]) == 0
            or len(graph["suc"][-1]["u"]) == 0
        ):
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                temp.new().resize_(0),
            )

        ctrs = torch.cat(graph["ctrs"], 0)
        feat = self.input(ctrs)
        feat += self.seg(graph["feats"])
        feat = self.relu(feat)

        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, graph["idcs"], graph["ctrs"]

class A2M(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """
    def __init__(self, config):
        super(A2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map + 11, n_map, norm=norm, ng=ng)
        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_actor"]))
        self.att = nn.ModuleList(att)

    def forward(self, feat: Tensor, graph: Dict[str, Union[List[Tensor], Tensor, List[Dict[str, Tensor]], Dict[str, Tensor]]], actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        """meta, static and dyn fuse using attention"""
        meta = torch.cat(
            (
                graph['lane_type'],
                graph['left_mark_type'],
                graph['right_mark_type'],
                graph["intersect"].unsqueeze(1),
            ),
            1,
        )
        feat = self.meta(torch.cat((feat, meta), 1))

        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
            )
        return feat

class M2M(nn.Module):
    """
    The lane to lane block: propagates information over lane
            graphs and updates the features of lane nodes
    """
    def __init__(self, config):
        super(M2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor, graph: Dict) -> Tensor:
        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat

class M2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """
    def __init__(self, config):
        super(M2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_map))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes: Tensor, node_idcs: List[Tensor], node_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
            )
        return actors

class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors

class EncodeDist(nn.Module):
    def __init__(self, n, linear=True):
        super(EncodeDist, self).__init__()
        norm = "GN"
        ng = 1

        block = [nn.Linear(2, n), nn.ReLU(inplace=True)]

        if linear:
            block.append(nn.Linear(n, n))

        self.block = nn.Sequential(*block)

    def forward(self, dist):
        x, y = dist[:, :1], dist[:, 1:]
        dist = torch.cat(
            (
                torch.sign(x) * torch.log(torch.abs(x) + 1.0),
                torch.sign(y) * torch.log(torch.abs(y) + 1.0),
            ),
            1,
        )

        dist = self.block(dist)
        return dist
    
class TestNet_Half(nn.Module):
    """
    Final position prediction with Linear Residual block
    """
    def __init__(self, config):
        super(TestNet_Half, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1
        n_actor = config["n_actor"]
        self.test = nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2))

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        reg = self.test(actors)
        reg = reg.view(-1, 2)
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 2)
            reg[idcs] = reg[idcs] + ctrs
        test_ctrs = []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            test_ctrs.append(reg[idcs])
        return test_ctrs
    
class TestNet(nn.Module):
    """
    Final position prediction with Linear Residual block
    """
    def __init__(self, config):
        super(TestNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1
        n_actor = config["n_actor"]
        pred = []
        for i in range(config["num_test_mod"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2),
                )
            )
        self.pred = nn.ModuleList(pred)
        self.att_dest = myAttDest(n_actor, config["num_test_mod"])
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes, node_idcs, node_ctrs) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg.detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs, actor_idcs, nodes, node_idcs, node_ctrs) 
        cls = self.cls(feats).view(-1, self.config["num_test_mod"])
        cls = self.softmax(cls)
        
        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), 2)
        
        out = dict()
        out["test_cls"] = []
        out["test_ctrs"] = []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            out["test_cls"].append(cls[idcs])
            out["test_ctrs"].append(reg[idcs])
        return out
    
class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """
    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = myAttDest(n_actor, config["num_mods"])
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes, node_idcs, node_ctrs) -> Dict[str, List[Tensor]]:
        # pdb.set_trace()
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        # pdb.set_trace()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs, actor_idcs, nodes, node_idcs, node_ctrs) # 2064, 128
        cls = self.cls(feats).view(-1, self.config["num_mods"])
        cls = self.softmax(cls)
        
        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"], out["test_ctrs"], out["test_cls"] = [], [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out, feats

class PredNetNoMap(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    For the student (without map)
    """
    def __init__(self, config):
        super(PredNetNoMap, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = myAttDestNoMap(n_actor, config["num_mods"])
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs, actor_idcs) ## embedding para distill
        cls = self.cls(feats).view(-1, self.config["num_mods"])
        cls = self.softmax(cls)
        
        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"], out["test_ctrs"], out["test_cls"] = [], [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out, feats

class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """
    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float) -> Tensor:
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts

class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts

class myAttDest(nn.Module):
    def __init__(self, n_agt: int, K):
        super(myAttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)
        self.K = K
        #self.K_agt = Linear(self.K * n_agt, n_agt, norm=norm, ng=ng)
        self.m2a_dest = M2A(config)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor, actor_idcs, nodes, node_idcs, node_ctrs) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)
        ctrs = []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs.append(dest_ctrs[idcs])
        for k in range(num_mods):
            dest_ctr = dest_ctrs[:,k,:]
            dist = (agt_ctrs - dest_ctr).view(-1, 2)
            dist = self.dist(dist)
            ctr = [x[:,k,:] for x in ctrs]
            # pdb.set_trace()
            actors = self.m2a_dest(agts, actor_idcs, ctr, nodes, node_idcs, node_ctrs)   # aplicar linear (?)
            actors = torch.cat((dist, actors), 1)
            actors = self.agt(actors)
            if k == 0:
                k_actors = actors
            else:
                k_actors = torch.cat((k_actors, actors), 1)
        k_actors = k_actors.view(-1, n_agt)
        #agts = self.K_agt(k_actors)
        return k_actors
    
class myAttDestNoMap(nn.Module):
    def __init__(self, n_agt: int, K):
        super(myAttDestNoMap, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)
        self.K = K
        #self.K_agt = Linear(self.K * n_agt, n_agt, norm=norm, ng=ng)
        self.m2a_dest = M2A(config)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor, actor_idcs) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)
        ctrs = []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs.append(dest_ctrs[idcs])
        for k in range(num_mods):
            dest_ctr = dest_ctrs[:,k,:]
            dist = (agt_ctrs - dest_ctr).view(-1, 2)
            dist = self.dist(dist)
            actors = torch.cat((dist, agts), 1)
            actors = self.agt(actors)
            if k == 0:
                k_actors = actors
            else:
                k_actors = torch.cat((k_actors, actors), 1)
        k_actors = k_actors.view(-1, n_agt)
        #agts = self.K_agt(k_actors)
        return k_actors

# Losses

## Original GANet

class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls, reg, test_ctrs, test_half_ctrs, test_cls = out["cls"], out["reg"], out["test_ctrs"], out["test_half_ctrs"], out["test_cls"]
        cls = torch.cat([x for x in cls], 0)
        test_cls = torch.cat([x for x in test_cls], 0)
        reg = torch.cat([x for x in reg], 0)
        test_ctrs = torch.cat([x for x in test_ctrs], 0)
        test_half_ctrs = torch.cat([x for x in test_half_ctrs], 0)
        
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + test_cls.sum()+ reg.sum() + test_ctrs.sum()+ test_half_ctrs.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["test_cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["num_test_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0
        loss_out["test_loss"] = zero.clone()
        loss_out["test_half_loss"] = zero.clone()
        loss_out["num_test"] = 0   
        loss_out["end_loss"] = zero.clone()
        loss_out["num_end"] = 0
        
        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0
        cls = cls[mask]    
        test_cls = test_cls[mask]
        reg = reg[mask]
        test_ctrs = test_ctrs[mask]
        test_half_ctrs = test_half_ctrs[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]
        
        half_idcs = torch.floor(torch.div(last_idcs, 2)).long()
        
        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)
        
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * (self.reg_loss(reg[has_preds], gt_preds[has_preds]))
        loss_out["num_reg"] += has_preds.sum().item()
        
        coef = self.config["end_coef"]
        loss_out["end_loss"] += coef * (self.reg_loss(reg[row_idcs, last_idcs], gt_preds[row_idcs, last_idcs]))
        loss_out["num_end"] += len(reg)
        
        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(config["num_test_mod"]):
            dist.append(
                torch.sqrt(
                    (
                        (test_ctrs[row_idcs, j] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)
        
        mgn = test_cls[row_idcs, min_idcs].unsqueeze(1) - test_cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        
        coef = self.config["test_cls_coef"]
        loss_out["test_cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_test_cls"] += mask.sum().item()
        
        test_ctrs = test_ctrs[row_idcs, min_idcs]
        coef = self.config["test_coef"]
        loss_out["test_loss"] += coef * self.reg_loss(test_ctrs[row_idcs], gt_preds[row_idcs, last_idcs])  
        
        coef = self.config["test_half_coef"]
        loss_out["test_half_loss"] += coef * self.reg_loss(test_half_ctrs[row_idcs], gt_preds[row_idcs, half_idcs])  
        loss_out["num_test"] += len(min_idcs)        
        return loss_out

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        pdb.set_trace()
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (
            loss_out["num_reg"] + 1e-10
        ) + loss_out["end_loss"] / (
            loss_out["num_end"] + 1e-10
        ) + loss_out["test_loss"] / (
            loss_out["num_test"] + 1e-10
        )+ loss_out["test_cls_loss"]/ (
            loss_out["num_test_cls"] + 1e-10
        ) + loss_out["test_half_loss"] / (
            loss_out["num_test"] + 1e-10
        )
        return loss_out

## Loss original LaneGCN (only regression and confidences)

class PredLossLaneGCN(nn.Module):
    def __init__(self, config):
        super(PredLossLaneGCN, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor]) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out

class LossLaneGCN(nn.Module):
    def __init__(self, config):
        super(LossLaneGCN, self).__init__()
        self.config = config
        self.pred_loss = PredLossLaneGCN(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out
    
# Aux classes and functions

class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out,data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["cls"] = [x[0:1].detach().cpu().numpy() for x in out["cls"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1] for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        test = metrics["test_loss"] / (metrics["num_test"] + 1e-10)
        end = metrics["end_loss"] / (metrics["num_end"] + 1e-10)
        test_half = metrics["test_half_loss"] / (metrics["num_test"] + 1e-10)
        test_cls = metrics["test_cls_loss"] / (metrics["num_test_cls"] + 1e-10)
        loss = cls + reg + test + test_half + test_cls + end

        preds = np.concatenate(metrics["preds"], 0)
        preds_cls = np.concatenate(metrics["cls"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = metrics["has_preds"]
        #has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, brier_fde, min_idcs = pred_metrics(preds, gt_preds, has_preds, preds_cls)

        print(
            "loss %2.4f %2.4f %2.4f %2.4f, %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f, brier_fde %2.4f"
            % (loss, cls, reg, end, test_cls, test, test_half, ade1, fde1, ade, fde, brier_fde)
        )
        print()

class PostProcessLaneGCN(nn.Module):
    def __init__(self, config):
        super(PostProcessLaneGCN, self).__init__()
        self.config = config

    def forward(self, out,data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["cls"] = [x[0:1].detach().cpu().numpy() for x in out["cls"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1] for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        # preds = np.concatenate(metrics["preds"], 0)
        # gt_preds = np.concatenate(metrics["gt_preds"], 0)
        # has_preds = np.concatenate(metrics["has_preds"], 0)
        # ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        # print(
        #     "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
        #     % (loss, cls, reg, ade1, fde1, ade, fde)
        # )
        # print()
        
        preds = np.concatenate(metrics["preds"], 0)
        preds_cls = np.concatenate(metrics["cls"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = metrics["has_preds"]
        #has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, brier_fde, min_idcs = pred_metrics(preds, gt_preds, has_preds, preds_cls)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f, brier_fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde, brier_fde)
        )
        print()
        
def pred_metrics(preds, gt_preds, has_preds, preds_cls):
    #assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    cls = np.asarray(preds_cls, np.float32)
    m, num_mods, num_preds, _ = preds.shape
    has_preds = torch.cat([x for x in has_preds], 0)
    last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
    max_last, last_idcs = last.max(1)
    
    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))
    
    row_idcs_last = np.arange(len(last_idcs)).astype(np.int64) 
    ade1 =  np.asarray([err[i, 0, :last_idcs[i]].mean() for i in range(m)]).mean()
    fde1 = err[row_idcs_last, 0, last_idcs].mean()
    #cls = softmax(cls, axis=1)
    min_idcs = err[row_idcs_last, :, last_idcs].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    cls = cls[row_idcs, min_idcs]
    ade = np.asarray([err[i, :last_idcs[i]].mean() for i in range(m)]).mean()
    fde = err[row_idcs_last, last_idcs].mean()
    one_arr = np.ones(m)
    brier_fde = (err[row_idcs_last, last_idcs] + (one_arr-cls)**2).mean()
    
    return ade1, fde1, ade, fde, brier_fde, min_idcs

def get_model(exp_name, distill=False, use_map=False, use_goal_areas=False, map_in_decoder=False):
    """_summary_

    Args:
        exp_name (_type_): _description_
        distill (bool, optional): _description_. Defaults to False.
        use_map (bool, optional): _description_. Defaults to False.
        use_goal_areas (bool, optional): _description_. Defaults to False.
        map_in_decoder (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    # Specify experiment name
    
    if "save_dir" not in config:
        config["save_dir"] = os.path.join(
            root_path, exp_name, model_name
        )

    if not os.path.isabs(config["save_dir"]):
        config["save_dir"] = os.path.join(root_path, exp_name, config["save_dir"])
        
    if not distill:
        config["use_map"] = use_map
        config["use_goal_areas"] = use_goal_areas
        config["map_in_decoder"] = map_in_decoder
    
        if use_map and use_goal_areas and map_in_decoder:
            loss = Loss(config).cuda()
            post_process = PostProcess(config).cuda()
        else:
            loss = LossLaneGCN(config).cuda()
            post_process = PostProcessLaneGCN(config).cuda()
            
        net = Net(config)
        net = net.cuda()

        params = net.parameters()
        opt = Optimizer(params, config)

        return config, ArgoDataset, collate_fn, net, loss, post_process, opt

    else:
        print("\nA. Create teacher: \n")
        
        config_teacher = copy.deepcopy(config)
        config_teacher["name"] = "teacher"
        config_teacher["use_map"] = True
        config_teacher["use_goal_areas"] = False
        config_teacher["map_in_decoder"] = False
        
        teacher = Net(config_teacher)
        
        print("\nB. Create student: ")
        config_student = copy.deepcopy(config)
        config_student["name"] = "student"
        config_student["use_map"] = False
        config_student["use_goal_areas"] = False
        config_student["map_in_decoder"] = False
        config_student["n_actor"] = 192
        
        student = Net(config_student)

        teacher.cuda()
        student.cuda()

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        if config_student["use_map"] == False or config_student["use_goal_areas"] or config_student["map_in_decoder"] == False:
            loss = LossLaneGCN(config).cuda() # Loss that does not consider the map
            # loss_distill = nn.MSELoss()
            loss_distill = nn.SmoothL1Loss(reduction="sum")
            # loss_distill = nn.CosineSimilarity(dim=1, eps=1e-6)
            post_process = PostProcessLaneGCN(config).cuda()
        else:
            loss = Loss(config).cuda()
            loss_distill = nn.SmoothL1Loss(reduction="sum")
            post_process = PostProcess(config).cuda()
            
        params = student.parameters()
        opt = Optimizer(params, config)

        return config, ArgoDataset, collate_fn, teacher, student, \
            loss, loss_distill, post_process, opt
