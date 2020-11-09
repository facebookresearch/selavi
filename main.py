# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import shutil
import time
from logging import getLogger

# Import torch and other dependecies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.tensorboard import SummaryWriter


from datasets.AVideoDataset import AVideoDataset
from model import load_model
from opt import parse_arguments
from src.sk_utils import cluster
from src.warmup_scheduler import GradualWarmupScheduler
from utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    init_signal_handler,
    trigger_job_requeue,
    get_loss
)

logger = getLogger()

# global variables
sk_schedule = None
group = None
sk_counter = 0

def main():

    # parse arguments
    global args
    parser = parse_arguments()
    args = parser.parse_args()

    # exp setup: logger, distributed mode and seeds
    init_distributed_mode(args)
    init_signal_handler()
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    if args.rank == 0:
        writer = SummaryWriter(args.dump_path)
    else:
        writer = None

    # build data
    train_dataset = AVideoDataset(
        ds_name=args.ds_name,
        root_dir=args.root_dir,
        mode='train',
        path_to_data_dir=args.data_path,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        sample_rate=args.sample_rate,
        num_train_clips=args.num_train_clips,
        train_crop_size=args.train_crop_size,
        test_crop_size=args.test_crop_size,        
        num_data_samples=args.num_data_samples,
        colorjitter=args.colorjitter,
        use_grayscale=args.use_grayscale,
        use_gaussian=args.use_gaussian,
        temp_jitter=True,
        decode_audio=True,
        aug_audio=None,
        num_sec=args.num_sec_aud,
        aud_sample_rate=args.aud_sample_rate,
        aud_spec_type=args.aud_spec_type,
        use_volume_jittering=args.use_volume_jittering,
        use_temporal_jittering=args.use_audio_temp_jittering,
        z_normalize=args.z_normalize,
        dual_data=args.dual_data
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Loaded data with {} videos.".format(len(train_dataset)))

    # Load model
    model = load_model(
        vid_base_arch=args.vid_base_arch, 
        aud_base_arch=args.aud_base_arch,
        use_mlp=args.use_mlp,
        num_classes=args.mlp_dim, 
        pretrained=False,
        norm_feat=True,
        use_max_pool=False,
        headcount=args.headcount,
    )

    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        process_group = None
        if args.world_size // 8 > 0:
            process_group = apex.parallel.create_syncbn_process_group(args.world_size // 8)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)

    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    if args.use_warmup_scheduler:
        lr_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.world_size,
            total_epoch=args.warmup_epochs,
            after_scheduler=None
        )
    else:
        lr_scheduler = None
        
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )

    # SK-Init
    N_dl = len(train_loader)
    N = len(train_loader.dataset)
    N_distr = N_dl * train_loader.batch_size
    selflabels = torch.zeros((N, args.headcount), dtype=torch.long, device='cuda')
    global sk_schedule
    sk_schedule = (args.epochs * N_dl * (np.linspace(0, 1, args.nopts) ** args.schedulepower)[::-1]).tolist()
    # to make sure we don't make it empty
    sk_schedule = [(args.epochs + 2) * N_dl] + sk_schedule
    logger.info(f'remaining SK opts @ epochs {[np.round(1.0 * t / N_dl, 2) for t in sk_schedule]}')

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0, 'selflabels': selflabels}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        amp=apex.amp if args.use_fp16 else None,
    )
    start_epoch = to_restore["epoch"]
    selflabels = to_restore['selflabels']

    # Set CuDNN benhcmark
    cudnn.benchmark = True

    # Restart schedule correctly
    if start_epoch != 0:
        include = [(qq / N_dl > args.start_epoch) for qq in sk_schedule]
        global sk_counter
        # (total number of sk-opts) - (number of sk-opts outstanding)
        sk_counter = len(sk_schedule) - sum(include)
        sk_schedule = (np.array(sk_schedule)[include]).tolist()
        if lr_scheduler:
            [lr_scheduler.step() for _ in range(to_restore['epoch'])]

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores, selflabels = train(
            train_loader, model, optimizer, epoch, writer, selflabels)
        training_stats.update(scores)

        # Update LR scheduler
        if lr_scheduler:
            lr_scheduler.step()

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "selflabels": selflabels
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth")
                )


def train(train_loader, model, optimizer, epoch, writer, selflabels):
    
    global sk_schedule

    # Put model in train mode
    model.train()

    # Init Logger meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    world_size = args.world_size
    dataset_bs = train_loader.batch_size

    end = time.time()
    batches_thusfar = epoch * len(train_loader)
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # ============ Get inputs ... ============
        video, audio, _, selected, _ = inputs
        video, audio = video.cuda(), audio.cuda()

        # ============ Occasional clustering via Sinkhorn-Knopp ... ===========
        if batches_thusfar + it >= sk_schedule[-1]:
            # optimize labels
            with torch.no_grad():
                _ = sk_schedule.pop()
                selflabels = cluster(
                    args, selflabels, train_loader.dataset, model, 
                    logger, writer, group,
                    (batches_thusfar + it) * dataset_bs * world_size,
                    sk_counter
                )

        # ============ forward passes ... ============
        feat_v, feat_a = model(video, audio)
        
        # ============ SeLaVi loss ... ============
        if args.headcount == 1:
            labels = selflabels[selected, 0]  
        else:
            labels = selflabels[selected, :]
        loss_vid = get_loss(feat_v, labels, headcount=args.headcount)
        loss_aud = get_loss(feat_a, labels, headcount=args.headcount)
        loss = 0.5 * loss_vid + 0.5 * loss_aud

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        
        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        iteration = epoch * len(train_loader) + it
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

            # Log onto tensorboard
            if writer:
                writer.add_scalar(
                    f'loss/iter', loss.item(), iteration)
                writer.add_scalar(
                    f'lr/iter', optimizer.param_groups[0]["lr"], iteration)
                writer.add_scalar(
                    f'batch_time/iter', batch_time.avg, iteration)
                writer.add_scalar(
                    f'data_time/iter', data_time.avg, iteration)
            
        # ============ signal handling ... ============
        if os.environ['SIGNAL_RECEIVED'] == 'True':
            if args.rank == 0:
                logger.info("Beginning reqeue")
                trigger_job_requeue(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"))
    
    dist.barrier()
    torch.cuda.empty_cache()
    return (epoch, losses.avg), selflabels


if __name__ == "__main__":
    main()
