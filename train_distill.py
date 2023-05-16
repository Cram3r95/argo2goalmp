
# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import pdb
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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
from importlib import import_module
from numbers import Number

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd

from torch.utils.data.distributed import DistributedSampler

from utils import Logger, load_pretrain

from mpi4py import MPI

comm = MPI.COMM_WORLD
hvd.init()
torch.cuda.set_device(hvd.local_rank())

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="GANet", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
# # Teacher GANet without map in decoder
# parser.add_argument(
#     "--weight", default="/home/robesafe/GANet-unofficial/results_ganet_without_mapindecoder/GANet/43.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
# )
# Teacher LaneGCN (without map in decoder and goal areas)
parser.add_argument(
    "--weight", default="/home/robesafe/GANet-unofficial/results_student_lanegcn/GANet/50.000.ckpt", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
    "--use_map", default=False, type=bool)
parser.add_argument(
    "--use_goal_areas", default=False, type=bool)
parser.add_argument(
    "--distill", default=True, type=bool)

def main():
    """_summary_
    """
    seed = hvd.rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, teacher, student, loss, loss_distill, post_process, opt = \
        model.get_model(args.distill)
        
    if config["horovod"]:
        opt.opt = hvd.DistributedOptimizer(
            opt.opt, named_parameters=teacher.named_parameters()
        )

    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(teacher, ckpt["state_dict"])

        if args.resume: # Not required if you are not going to continue training the model
            config["epoch"] = ckpt["epoch"]
            opt.load_state_dict(ckpt["opt_state"])

    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    if hvd.rank() == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sys.stdout = Logger(log)

        src_dirs = [root_path]
        dst_dirs = [os.path.join(save_dir, "files")]
        for src_dir, dst_dir in zip(src_dirs, dst_dirs):
            files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    train_sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    hvd.broadcast_parameters(student.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt.opt, root_rank=0)

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(args, epoch + i, config, train_loader,
              teacher, student, loss, loss_distill, 
              post_process, opt, val_loader)

    if args.eval:
        # Data loader for validation (only for the student)
        dataset = Dataset(config["val_split"], config, train=False)
        val_sampler = DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        val_loader = DataLoader(
            dataset,
            batch_size=config["val_batch_size"],
            num_workers=config["val_workers"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        hvd.broadcast_parameters(student.state_dict(), root_rank=0)
        val(config, val_loader, student, loss, post_process, 999)
        return
    
def worker_init_fn(pid):
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)

def train(
        args, epoch, config, train_loader, teacher, student, 
        loss, loss_distill, post_process, opt, val_loader=None
    ):
    train_loader.sampler.set_epoch(int(epoch))
    teacher.eval()
    student.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))

    # display_iters = int(
    #     config["display_iters"] / (hvd.size() * config["batch_size"])
    # )
    display_iters = 50
    val_iters = int(config["val_iters"] / (hvd.size() * config["batch_size"]))

    print("Display train iters: ", display_iters)
    print("Display val iters: ", val_iters)
    print("Save iters: ", save_iters)
    
    start_time = time.time()
    metrics = dict()
    for i, data in tqdm(enumerate(train_loader),disable=hvd.rank()):
        epoch += epoch_per_batch
        data = dict(data)
        
        output_t, embeddings_pre_a2a_t = teacher(data, return_embeddings=True)
        output_s, embeddings_pre_a2a_s = student(data, return_embeddings=True)

        ## regression loss
        
        loss_out = loss(output_s, data)
        
        ## distillation loss

        # loss_distill_pre_a2a = torch.sum(loss_distill(embeddings_pre_a2a_t, embeddings_pre_a2a_s)) # only cosine
        if teacher.net_config["n_actor"] == student.net_config["n_actor"]:
            loss_distill_pre_a2a = loss_distill(embeddings_pre_a2a_t, embeddings_pre_a2a_s)
        else: # If student latent space is higher than teacher latent 
            loss_distill_pre_a2a = loss_distill(embeddings_pre_a2a_t, embeddings_pre_a2a_s[:,:teacher.net_config["n_actor"]])
        loss_out["distill_loss"] = config["distill_coef"]*loss_distill_pre_a2a

        ## sum losses
        loss_out["loss"] += loss_out["distill_loss"]

        post_out = post_process(output_s, data)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if hvd.rank() == 0 and (
            num_iters % save_iters == 0 or epoch >= config["num_epochs"]
        ):
            save_ckpt(student, opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            metrics = sync(metrics)
            if hvd.rank() == 0:
                post_process.display(metrics, dt, epoch, lr=lr, distill=args.distill)
            start_time = time.time()
            metrics = dict()

        if epoch >= config["start_val_epoch"] and num_iters % val_iters == 0:
            val(args, config, val_loader, student, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(args, config, val_loader, student, loss, post_process, epoch)
            return

def val(args, config, data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)
        break
    dt = time.time() - start_time
    metrics = sync(metrics)
    if hvd.rank() == 0:
        # Distillation is not evaluated in validation
        post_process.display(metrics, dt, epoch)
    net.train()

def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )

def sync(data):
    data_list = comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data

if __name__ == "__main__":
    main()
