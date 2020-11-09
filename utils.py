# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os
import signal
import time

import numpy as np
import torch

from src.logger import create_logger, PD_Stats

import torch.distributed as dist

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = getLogger()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.stack(out_list)


def dist_collect_other(x, return_before_cat=False):
    """ collect all tensor from all GPUs except current one
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    # get only non local ones.
    out_list = [out_list[rank]
                for rank in range(dist.get_world_size()) if rank != dist.get_rank()]
    if return_before_cat: 
        return out_list
    return torch.cat(out_list, dim=0)


def SIGTERMHandler(a, b):
    print('received sigterm')
    pass


def signalHandler(a, b):
    print('Signal received', a, time.time(), flush=True)
    os.environ['SIGNAL_RECEIVED'] = 'True'
    return


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    os.environ['SIGNAL_RECEIVED'] = 'False'
    os.environ['MAIN_PID'] = str(os.getpid())

    signal.signal(signal.SIGUSR1, signalHandler)
    signal.signal(signal.SIGTERM, SIGTERMHandler)
    print("Signal handler installed.", flush=True)


def trigger_job_requeue(checkpoint_filename):
    ''' Submit a new job to resume from checkpoint.
        Be careful to use only for main process.
    '''
    if int(os.environ['SLURM_PROCID']) == 0 and \
            str(os.getpid()) == os.environ['MAIN_PID'] and os.path.isfile(checkpoint_filename):
        print('time is up, back to slurm queue', flush=True)
        command = 'scontrol requeue ' + os.environ['SLURM_JOB_ID']
        print(command)
        if os.system(command):
            raise RuntimeError('requeue failed')
        print('New job submitted to the queue', flush=True)
    exit(0)


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    # prepare distributed
    if not args.bash:	
        dist.init_process_group(	
            backend="nccl",	
            init_method=args.dist_url,	
            world_size=args.world_size,	
            rank=args.rank,	
        )	
        # set cuda device	
        args.gpu_to_work_on = args.rank % torch.cuda.device_count()	
        torch.cuda.set_device(args.gpu_to_work_on)	
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        # set cuda device
        print(torch.cuda.device_count())
        args.gpu_to_work_on = args.local_rank % torch.cuda.device_count()
        print(args.gpu_to_work_on, torch.cuda.device_count())
        torch.cuda.set_device(args.gpu_to_work_on)
    return


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, "checkpoints")
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"), rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")
    return logger, training_stats


def save_checkpoint(args, epoch, model, optimizer, lr_scheduler, ckpt_freq=10):
    checkpoint = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch + 1,
        'args': args
    }
    os.makedirs(os.path.join(args.output_dir, 'model_weights'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    if epoch % 10 == 0:
        torch.save(
            checkpoint,
            os.path.join(args.output_dir, 'model_weights', f'model_{epoch}.pth'.format(epoch))
        )
    torch.save(
        checkpoint,
        os.path.join(args.output_dir, 'checkpoints', 'checkpoint.pth')
    )
    if epoch % ckpt_freq == 0:
        torch.save(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoints', f'ckpt_{epoch}.pth')
        )
    print(f'Saving checkpoint to: {args.output_dir}', flush=True)
    print(f'Checkpoint saved', flush=True)


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path, map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def load_model_parameters(model, model_weights):
    loaded_state = model_weights
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys():
            self_state[name].copy_(param)
        else:
            print("didnt load ", name)


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class QueueAverage(object):
    def __init__(self):
        self.queue = []
        self.val = 0

    def update(self, val):
        if len(self.queue) == 0:
            self.queue.append(val)
            self.val = val
        elif len(self.queue) == 1:
            self.queue.append(val)
            self.val = 2 * self.queue[1] - self.queue[0] 
        elif len(self.queue) == 2:
            self.queue.pop(0)
            self.queue.append(val)
            self.val = 2 * self.queue[1] - self.queue[0]
        

class MovingAverage(object):
    def __init__(self, intertia=0.9):
        self.intertia = intertia
        self.reset()

    def reset(self):
        self.avg = 0.

    def update(self, val):
        self.avg = self.intertia * self.avg + (1 - self.intertia) * val
        self.val = val


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


## Aggerate video level softmaxes into an accuracy score
def aggregrate_video_accuracy(softmaxes, labels, topk=(1,), aggregate="mean"):
    maxk = max(topk)
    output_batch = torch.stack(
        [torch.mean(torch.stack(
            softmaxes[sms]),
            0,
            keepdim=False
        ) for sms in softmaxes.keys()])
    num_videos = output_batch.size(0)
    output_labels = torch.stack(
        [labels[video_id] for video_id in softmaxes.keys()])

    _, pred = output_batch.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(output_labels.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / num_videos))
    return res


def get_loss(activations, targets, headcount=1):
    if headcount == 1:
        loss = torch.nn.functional.cross_entropy(activations, targets)
    else:
        loss = torch.mean(
            torch.stack(
                [torch.nn.functional.cross_entropy(activations[h], targets[:, h])
                 for h in range(headcount)]
            )
        )
    return loss