# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
import datetime
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision

# Custom imports
from src.warmup_scheduler import GradualWarmupScheduler
from utils import (
    AverageMeter,
    accuracy,
    aggregrate_video_accuracy,
    initialize_exp,
    getLogger,
    accuracy,
    save_checkpoint,
    load_model_parameters
)
from datasets.AVideoDataset import AVideoDataset
from model import load_model, Identity, get_video_dim

logger = getLogger()


# DICT with number of classes for each  dataset
NUM_CLASSES = {
    'hmdb51': 51,
    'ucf101': 101,
}


# Create Finetune Model
class Finetune_Model(torch.nn.Module):
    def __init__(
        self, 
        base_arch, 
        num_ftrs=512, 
        num_classes=101, 
        use_dropout=False, 
        use_bn=False, 
        use_l2_norm=False, 
        dropout=0.9
    ):
        super(Finetune_Model, self).__init__()
        self.base = base_arch
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm

        message = 'Classifier to %d classes;' % (num_classes)
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_bn: message += ' + final BN'
        print(message)

        if self.use_bn:
            self.final_bn = nn.BatchNorm1d(num_ftrs)
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(num_ftrs, num_classes)
        self._initialize_weights(self.classifier)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
    
    def forward(self, x):
        x = self.base(x).squeeze()
        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=1)
        if self.use_bn:
            x = self.final_bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


def main(args, writer):

    # Create Logger
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", 
        "loss_val", "prec1_val", "prec5_val"
    )

    # Set CudNN benchmark
    torch.backends.cudnn.benchmark = True

    # Load model
    logger.info("Loading model")
    model = load_model(
        vid_base_arch=args.vid_base_arch, 
        aud_base_arch=args.aud_base_arch, 
        pretrained=args.pretrained,
        num_classes=args.num_clusters,
        norm_feat=False,
        use_mlp=args.use_mlp,
        headcount=args.headcount,
    )

    # Load model weights
    weight_path_type = type(args.weights_path)
    if weight_path_type == str:
        weight_path_not_none = args.weights_path != 'None' 
    else:
        weight_path_not_none = args.weights_path is not None
    if not args.pretrained and weight_path_not_none:
        logger.info("Loading model weights")
        if os.path.exists(args.weights_path):
            ckpt_dict = torch.load(args.weights_path)
            model_weights = ckpt_dict["model"]
            logger.info(f"Epoch checkpoint: {args.ckpt_epoch}")
            load_model_parameters(model, model_weights)
    logger.info(f"Loading model done")

    # Add FC layer to model for fine-tuning or feature extracting
    model = Finetune_Model(
        model.video_network.base, 
        get_video_dim(vid_base_arch=args.vid_base_arch), 
        NUM_CLASSES[args.dataset], 
        use_dropout=args.use_dropout, 
        use_bn=args.use_bn, 
        use_l2_norm=args.use_l2_norm, 
        dropout=0.7
    )

    # Create DataParallel model
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model_without_ddp = model.module

    # Get params for optimization
    params = []
    if args.feature_extract: # feature_extract only classifer
        for name, param in model_without_ddp.classifier.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {'params': param, 
                'lr': args.head_lr, 
                'weight_decay': args.weight_decay
            })
    else: # finetune
        for name, param in model_without_ddp.classifier.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {'params': param, 
                'lr': args.head_lr, 
                'weight_decay': args.weight_decay
            })
        for name, param in model_without_ddp.base.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {'params': param, 
                'lr': args.base_lr, 
                'weight_decay': args.wd_base
            })

    logger.info("Creating AV Datasets")
    dataset = AVideoDataset(
        ds_name=args.dataset,
        root_dir=args.root_dir,
        mode='train',
        num_frames=args.clip_len,
        sample_rate=args.steps_bet_clips,
        num_train_clips=args.train_clips_per_video,
        train_crop_size=128 if args.augtype == 1 else 224,
        seed=None,
        fold=args.fold,
        colorjitter=args.colorjitter,
        temp_jitter=True,
        center_crop=False,
        target_fps=30,
        decode_audio=False,
    )
    dataset_test = AVideoDataset(
        ds_name=args.dataset,
        root_dir=args.root_dir,
        mode='test',
        num_frames=args.clip_len,
        sample_rate=args.steps_bet_clips,
        test_crop_size=128 if args.augtype == 1 else 224,
        num_spatial_crops=args.num_spatial_crops,
        num_ensemble_views=args.val_clips_per_video,
        seed=None,
        fold=args.fold,
        colorjitter=args.test_time_cj,
        temp_jitter=True,
        target_fps=30,
        decode_audio=False,
    )

    # Creating dataloaders
    logger.info("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=None,
        num_workers=args.workers,
        pin_memory=True, 
        drop_last=True,
        shuffle=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        sampler=None,
        num_workers=args.workers,
        pin_memory=True, 
        drop_last=False
    )

    # linearly scale LR and set up optimizer
    if args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.head_lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    elif args.optim_name == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=args.head_lr, 
            weight_decay=args.weight_decay
        )

    # Multi-step LR scheduler
    if args.use_scheduler:
        lr_milestones = args.lr_milestones.split(',')
        milestones = [int(lr) - args.lr_warmup_epochs for lr in lr_milestones]
        if args.lr_warmup_epochs > 0:
            scheduler_step = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=milestones, 
                gamma=args.lr_gamma
            )
            multiplier = 8
            lr_scheduler = GradualWarmupScheduler(
                optimizer, 
                multiplier=multiplier,
                total_epoch=args.lr_warmup_epochs, 
                after_scheduler=scheduler_step
            )
        else: # no warmp, just multi-step
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=milestones, 
                gamma=args.lr_gamma
            )
    else:
        lr_scheduler = None

    # Checkpointing
    if args.resume:
        ckpt_path = os.path.join(
            args.output_dir, 'checkpoints', 'checkpoint.pth')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch']
        logger.info(f"Resuming from epoch: {args.start_epoch}")

    # Only perform evalaution
    if args.test_only:
        scores_val = evaluate(
            model, 
            data_loader_test,
            epoch=args.start_epoch, 
            writer=writer,
            ds=args.dataset,
        )
        _, vid_acc1, vid_acc5 = scores_val
        return vid_acc1, vid_acc5, args.start_epoch

    start_time = time.time()
    best_vid_acc_1 = -1
    best_vid_acc_5 = -1
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f'Start training epoch: {epoch}')
        scores = train(
            model, 
            optimizer, 
            data_loader,
            epoch, 
            writer=writer,
            ds=args.dataset,
        )
        logger.info(f'Start evaluating epoch: {epoch}')
        lr_scheduler.step()
        scores_val = evaluate(
            model, 
            data_loader_test,
            epoch=epoch,
            writer=writer,
            ds=args.dataset,
        )
        _, vid_acc1, vid_acc5 = scores_val
        training_stats.update(scores + scores_val)
        if vid_acc1 > best_vid_acc_1:
            best_vid_acc_1 = vid_acc1
            best_vid_acc_5 = vid_acc5
            best_epoch = epoch
        if args.output_dir:
            logger.info(f'Saving checkpoint to: {args.output_dir}')
            save_checkpoint(args, epoch, model, optimizer, lr_scheduler,
                ckpt_freq=1)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')
    return best_vid_acc_1, best_vid_acc_5, best_epoch


def train(
    model, 
    optimizer, 
    loader, 
    epoch, 
    writer=None,
    ds='hmdb51',
):
    # Put model in train mode
    model.train()

    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    criterion = nn.CrossEntropyLoss().cuda()

    for it, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update iteration
        iteration = epoch * len(loader) + it

        # forward
        video, target, _, _ = batch
        video, target = video.cuda(), target.cuda()
        output = model(video)

        # compute cross entropy loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), video.size(0))
        top1.update(acc1[0], video.size(0))
        top5.update(acc5[0], video.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR {lr}".format(
                    epoch,
                    it,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

            writer.add_scalar(
                f'{ds}/train/loss/iter', 
                losses.val, 
                iteration
            )
            writer.add_scalar(
                f'{ds}/train/clip_acc1/iter', 
                top1.val, 
                iteration
            )
    return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def evaluate(model, val_loader, epoch=0, writer=None, ds='hmdb51'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    # dicts to store labels and softmaxes
    softmaxes = {}
    labels = {}

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for batch_idx, batch in enumerate(val_loader):
            
            (video, target, _, video_idx) = batch

            # move to gpu
            video = video.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output and loss
            output = model(video)
            loss = criterion(output.view(video.size(0), -1), target)

            # Clip level accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), video.size(0))
            top1.update(acc1[0], video.size(0))
            top5.update(acc5[0], video.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # Video Level accuracy
            for j in range(len(video_idx)):
                video_id = video_idx[j].item()
                sm = output[j]
                label = target[j]

                # append it to video dict
                softmaxes.setdefault(video_id, []).append(sm)
                labels[video_id] = label
    
    # Get video acc@1 and acc@5 and output to tb writer
    video_acc1, video_acc5 = aggregrate_video_accuracy(
        softmaxes, labels, topk=(1, 5)
    )

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "ClipAcc@1 {top1.avg:.3f}\t"
            "VidAcc@1 {video_acc1:.3f}".format(
                batch_time=batch_time, 
                loss=losses, 
                top1=top1, 
                video_acc1=video_acc1.item())
        )

        writer.add_scalar(
            f'{ds}/val/vid_acc1/epoch', 
            video_acc1.item(), 
            epoch
        )
        writer.add_scalar(
            f'{ds}/val/vid_acc5/epoch', 
            video_acc5.item(), 
            epoch
        )

    # Log final results to terminal
    return losses.avg, video_acc1.item(), video_acc5.item()


def parse_args():
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
            'Instead, it is %s.' % v)

    import argparse
    parser = argparse.ArgumentParser(description='Finetuning')
    parser.register('type', 'bool', str2bool)

    ### DATA
    parser.add_argument('--dataset', default='ucf101', type=str,
                        choices=['kinetics', 'vggsound', 'kinetics_sound', 'ave', 'ucf101', 'hmdb51'],
                        help='name of dataset')
    parser.add_argument("--root_dir", type=str, default="/path/to/dataset",
                        help="root dir of dataset")
    parser.add_argument('--fold', default='1,2,3', type=str, 
                        help='fold number')
    parser.add_argument('--clip_len', default=32, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--augtype', default=1, type=int, 
                        help='augmentation type (default: 1)')
    parser.add_argument('--colorjitter', default='True', type='bool', 
                        help='color jittering as augmentations')
    parser.add_argument('--steps_bet_clips', default=1, type=int, 
                        help='number of steps between clips in video')
    parser.add_argument('--num_data_samples', default=None, type=int, 
                        help='number of samples in dataset')
    parser.add_argument('--train_clips_per_video', default=10, type=int, 
                        help='maximum number of clips per video for training')
    parser.add_argument('--val_clips_per_video', default=10, type=int, 
                        help='maximum number of clips per video for testing')
    parser.add_argument('--num_spatial_crops', default=3, type=int, 
                        help='number of spatial clips for testing')
    parser.add_argument('--test_time_cj', default='False', type='bool', 
                        help='test time CJ augmentation')
    parser.add_argument('--workers', default=0, type=int, 
                        help='number of data loading workers (default: 16)')

    ### MODEL
    parser.add_argument('--weights_path', default='', type=str,
                        help='Path to weights file',)
    parser.add_argument('--ckpt_epoch', default='0', type=str,
                        help='Epoch of model checkpoint')
    parser.add_argument('--vid_base_arch', default='r2plus1d_18', 
                        help='Video Base Arch for A-V model')
    parser.add_argument('--aud_base_arch', default='resnet9', 
                        help='Audio Base Arch for A-V model')
    parser.add_argument('--pretrained', default='False', type='bool', 
                        help='Use pre-trained models from the modelzoo')
    parser.add_argument('--use_mlp', default='True', type='bool', 
                        help='Use MLP projection head')
    parser.add_argument('--mlptype', default=0, type=int, 
                        help='MLP type (default: 0)')
    parser.add_argument('--headcount', default=10, type=int,
                        help='how many heads each modality has')
    parser.add_argument('--num_clusters', default=309, type=int,
                        help='number of clusters in last dimension')

    ### FINETUNE
    parser.add_argument('--feature_extract', default='False', type='bool', 
                        help="Use model as feature extractor")
    parser.add_argument("--use_dropout", default='False', type='bool', 
                        help='Use dropout in classifier')
    parser.add_argument('--use_bn', default='False', type='bool',
                        help='Use BN in classifier')
    parser.add_argument('--use_l2_norm', default='False', type='bool', 
                        help='Use L2-Norm in classifier')

    ### TRAINING
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=12, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--optim_name', default='sgd', type=str, 
                        help='Name of optimizer', choices=['sgd', 'adam'])
    parser.add_argument('--head_lr', default=0.0025, type=float, 
                        help='initial learning rate')
    parser.add_argument('--base_lr', default=0.00025, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum')
    parser.add_argument('--weight_decay', default=0.005, type=float,
                        help='weight decay')
    parser.add_argument('--wd_base', default=5e-3, type=float)
    parser.add_argument("--use_scheduler", default='True', type='bool', 
                        help='Use LR scheduler')
    parser.add_argument('--lr_warmup_epochs', default=2, type=int, 
                        help='number of warmup epochs')
    parser.add_argument('--lr_milestones', default='6,10', type=str, 
                        help='decrease lr on milestones (epochs)')
    parser.add_argument('--lr_gamma', default=0.05, type=float, 
                        help='decrease lr by a factor of lr-gamma')
    
    ### LOGGING
    parser.add_argument('--output_dir', default='.', type=str,
                        help='path where to save')

    ### CHECKPOINTING
    parser.add_argument('--resume', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch')
    parser.add_argument('--test_only', type='bool', default='False', 
                        help='Only test the model')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.dump_path = args.output_dir
    args.rank = 0
    logger.info(args)

    # Make output dir
    tbx_path = os.path.join(args.output_dir, 'tensorboard')
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set up tensorboard
    writer = writer = SummaryWriter(tbx_path)
    writer.add_text("namespace", repr(args))
    
    # Number of seconds
    if args.clip_len > 32:
        args.num_sec = int(args.clip_len / 30)

    # Run over different folds
    best_accs_1 = []
    best_accs_5 = []
    best_epochs = []
    folds = [int(fold) for fold in args.fold.split(',')]
    print(f"Evaluating on folds: {folds}")
    for fold in folds:
        args.fold = fold
        best_acc1, best_acc5, best_epoch = main(args, writer)
        best_accs_1.append(best_acc1)
        best_accs_5.append(best_acc5)
        best_epochs.append(best_epoch)
    avg_acc1 = np.mean(best_accs_1)
    avg_acc5 = np.mean(best_accs_5)
    logger.info(f"{len(folds)}-Fold ({args.dataset}): ")
    logger.info(f"Vid Acc@1 {avg_acc1:.3f}, Video Acc@5 {avg_acc5:.3f}")
