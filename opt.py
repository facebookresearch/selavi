# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from utils import bool_flag

def parse_arguments():
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
            'Instead, it is %s.' % v)
    
    parser = argparse.ArgumentParser(description="Implementation of SwAV")
    parser.register('type', 'bool', str2bool)

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--ds_name", type=str, default="kinetics",
                        choices=['kinetics', 'vggsound', 'kinetics_sound', 'ave', 'ucf101', 'hmdb51'],
                        help="name of dataset")
    parser.add_argument("--root_dir", type=str, default="/path/to/dataset",
                        help="root dir of dataset")
    parser.add_argument("--data_path", type=str, default="datasets/data",
                        help="path to store dataset pkl files")
    parser.add_argument("--num_data_samples", type=int, default=None,
                        help="number of dataset samples")
    parser.add_argument("--num_frames", type=int, default=30,
                        help="number of frames to sample per clip")
    parser.add_argument("--target_fps", type=int, default=30,
                        help="video fps")
    parser.add_argument("--sample_rate", type=int, default=1,
                        help="rate to sample frames")
    parser.add_argument("--num_train_clips", type=int, default=1,
                        help="number of clips to sample per videos")
    parser.add_argument("--train_crop_size", type=int, default=112,
                        help="train crop size")
    parser.add_argument("--test_crop_size", type=int, default=112,
                        help="test crop size")
    parser.add_argument('--colorjitter', type='bool', default='True',
                        help='use color jitter')
    parser.add_argument('--use_grayscale', type='bool', default='False',
                        help='use grayscale augmentation')
    parser.add_argument('--use_gaussian', type='bool', default='False',
                        help='use gaussian augmentation')
    parser.add_argument("--num_sec_aud", type=int, default=1,
                        help="number of seconds of audio")
    parser.add_argument("--aud_sample_rate", type=int, default=48000,
                        help="audio sample rate")
    parser.add_argument("--aud_spec_type", type=int, default=2,
                        help="audio spec type")
    parser.add_argument('--use_volume_jittering', type='bool', default='False',
                        help='use volume jittering')
    parser.add_argument('--use_audio_temp_jittering', type='bool', default='False',
                        help='use audio temporal jittering')
    parser.add_argument('--z_normalize', type='bool', default='False',
                        help='z-normalize the audio')
    parser.add_argument('--dual_data', type='bool', default='False',
                        help='sample two clips per video')

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--use_warmup_scheduler" , default='True', type='bool',
                        help="use warmup scheduler")
    parser.add_argument("--use_lr_scheduler" , default='False', type='bool',
                        help="use cosine LR scheduler")

    #########################
    #### SK parameters ###
    #########################
    parser.add_argument('--schedulepower', default=1.5, type=float,
                        help='SK schedule power compared to linear (default: 1.5)')
    parser.add_argument('--nopts', default=160, type=int, 
                        help='number of pseudo-opts (default: 100)')
    parser.add_argument('--lamb', default=20, type=int, 
                        help='for pseudoopt: lambda (default:25) ')

    #########################
    #### Selavi parameters ###
    #########################
    parser.add_argument('--ind_groups', default=1, type=int, 
                        help='number of independent groups (default: 100)')
    parser.add_argument('--match', default='True', type='bool', 
                        help='match distributions at beginning of training')
    parser.add_argument('--distribution', default='default', type=str, 
                        help='distribution of SK-clustering', choices=['gauss', 'default', 'zipf'])

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    parser.add_argument("--bash", action='store_true', help="slrum bash mode")
    parser.add_argument("--resume", default='False', type='bool', help="slrum bash mode")

    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--vid_base_arch", default="r2plus1d_18", type=str, 
                        help="video architecture", choices=['r2plus1d_18'])
    parser.add_argument("--aud_base_arch", default="resnet9", type=str, 
                        help="audio architecture", choices=['resnet9', 'resnet18'])
    parser.add_argument('--use_mlp', type='bool', default='True',
                        help='use MLP head')
    parser.add_argument("--mlp_dim", default=256, type=int,
                        help="final layer dimension in projection head")
    parser.add_argument("--headcount", default=1, type=int,
                        help="number of heads")

    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=25,
                        help="Save the model periodically")
    parser.add_argument("--use_fp16", type='bool', default='False',
                        help="whether to train with mixed precision or not")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    return parser
