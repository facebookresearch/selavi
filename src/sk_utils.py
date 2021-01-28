# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
import time
import torch
import torch.distributed as dist
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.stats import entropy
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score
)

from utils import trigger_job_requeue


def cluster(
        args,
        selflabels,
        dataset,
        model,
        sk_counter,
        logger,
        writer,
        group,
        iter_num):
    selflabels_old = selflabels.clone()

    # get cluster assignments
    with torch.no_grad():
        selflabels = get_cluster_assignments_gpu(
            args, dataset, model, logger, writer, group, iter_num)
    self_labels_np  = selflabels[:, 0].cpu().numpy()

    # increment counter
    sk_counter += 1

    if selflabels is not None:
        nmi_v = normalized_mutual_info_score(
            self_labels_np,
            selflabels_old[:,0].cpu().numpy(),
            average_method='arithmetic'
        )
        if args.rank == 0:
            logger.info(f'NMI_v: {nmi_v}')
        if writer:
            writer.add_scalar(
                f'train/nmi_v/iter',
                nmi_v,
                iter_num
            )
            writer.add_scalar(
                f'train/optim_count/iter',
                sk_counter,
                iter_num
            )

    true_labels = np.array(dataset._labels)[dataset.valid_indices]
    nmi_to_labels_v = normalized_mutual_info_score(
        self_labels_np,
        true_labels,
        average_method='arithmetic'
    )
    anmi_to_labels_v = adjusted_mutual_info_score(
        self_labels_np,
        true_labels,
        average_method='arithmetic'
    )
    if args.rank == 0:
        logger.info(f"NMI-tolabels: {nmi_to_labels_v}")
        logger.info(f"aNMI-tolabels: {anmi_to_labels_v}")
    if writer:
        writer.add_scalar(
            f'train/nmi-tolabels_v/iter',
            nmi_to_labels_v,
            iter_num
        )
        writer.add_scalar(
            f'train/a-nmi-tolabels_v/iter',
            anmi_to_labels_v,
            iter_num
        )
    if sk_counter % 10 == 0:
        entropies = []
        purities = []
        for sk_label in np.unique(self_labels_np):
            of_this_cluster = self_labels_np == sk_label
            size = of_this_cluster.sum()
            if size != 0:
                uniq, counts = np.unique(
                    true_labels[of_this_cluster], return_counts=True)
                purities.append(max(counts)/sum(1.0*counts))
                entropies.append(entropy(counts/sum(1.0*counts)))
        logger.info(f"Avg entropy: {np.mean(entropies)}")
        logger.info(f"Avg purity: {np.mean(purities)}")
        if writer:
            writer.add_histogram(
                'train/entropies',
                np.array(entropies),
                iter_num
            )
            writer.add_histogram(
                'train/purities',
                np.array(purities),
                iter_num
            )
            writer.add_scalar(
                'train/avg-entropy',
                np.mean(entropies),
                iter_num
            )
            writer.add_scalar(
                'train/avg-purity',
                np.mean(purities),
                iter_num
            )
    # signal received, relaunch experiment
    if os.environ['SIGNAL_RECEIVED'] == 'True':
        if args.rank == 0:
            logger.info("Beginning requeue", logger=logger)
            trigger_job_requeue(os.path.join(
                args.dump_path, "checkpoint.pth.tar"))
    # Ensure processes reach to end of optim clusters
    if group is not None:
        dist.barrier(group=group)
    else:
        dist.barrier()
    return selflabels


def get_cluster_assignments_gpu(
        args,
        dataset,
        model,
        logger=None,
        writer=None,
        group=None,
        iter_num=0
):
    # clear cache at beginning
    torch.cuda.empty_cache()

    # Put model in eval mode
    model.eval()

    # Get length of dataset
    N = len(dataset)

    # this process deals only with a subset of the dataset
    sampler = None
    local_nmb_data = N // args.world_size
    train_indices = torch.arange(
        args.rank * local_nmb_data,
        (args.rank + 1) * local_nmb_data
    ).int()
    # create subset sampler
    sampler = SubsetRandomSampler(train_indices)

    # create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=None
    )

    # Ensure processes reach to end of optim clusters
    if group is not None:
        dist.barrier(group=group)
    else:
        dist.barrier()

    # can't have more independent head-groups than heads
    assert args.ind_groups <= args.headcount

    if args.headcount > 1:
        # aggregate GAP features when using multi heads
        model.module.return_features = True
    aggregtensor = torch.cuda.DoubleTensor if args.headcount == 1 else torch.cuda.FloatTensor
    dtype = torch.float64 if args.headcount == 1 else torch.float32
    L = torch.zeros((N, args.headcount), dtype=torch.long, device='cuda')
    order_heads = list(range(args.headcount))
    np.random.shuffle(order_heads) # is inplace

    for hd_grp_idx in range(args.ind_groups):
        # 1. aggregate inputs:
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            video, audio, _, idx, _ = batch

            # Move to GPU
            video = video.cuda(non_blocking=True)
            audio = audio.cuda(non_blocking=True)
            idx = idx.cuda(non_blocking=True)

            # Forward pass
            feat_v, feat_a = model(video, audio)
            if args.headcount == 1:
                feat_v = torch.nn.functional.softmax(
                    feat_v, dim=1, dtype=torch.float64)
                feat_a = torch.nn.functional.softmax(
                    feat_a, dim=1, dtype=torch.float64)

            # gather the features computed by all processes
            all_feat_v_list = [aggregtensor(feat_v.size()) for src in range(args.world_size)]
            all_feat_a_list = [aggregtensor(feat_a.size()) for src in range(args.world_size)]
            all_indices_list = [torch.IntTensor(feat_v.size(0)).random_(0, N).cuda() for src in
                                range(args.world_size)]

            dist.all_gather(all_feat_v_list, feat_v)
            dist.all_gather(all_feat_a_list, feat_a)
            dist.all_gather(all_indices_list, idx)

            # only main process stores all features
            if args.rank == 0:
                all_feat_v = torch.cat(all_feat_v_list)
                all_feat_a = torch.cat(all_feat_a_list)
                all_indices = torch.cat(all_indices_list).cpu()

            if batch_idx == 0 and (args.rank == 0):
                fr = 0
                K = feat_v.size(1)
                PS_v = torch.zeros((N, K), dtype=dtype, device='cuda')
                PS_a = torch.zeros((N, K), dtype=dtype, device='cuda')
                indices = torch.zeros(N, dtype=torch.long)

            # fill in arrays on main node
            if args.rank == 0:
                to = fr + all_feat_v.shape[0]
                PS_v[fr: to] = all_feat_v
                PS_a[fr: to] = all_feat_a
                indices[fr: to] = all_indices
                fr = to

            # signal received, relaunch experiment
            if os.environ['SIGNAL_RECEIVED'] == 'True':
                if args.rank == 0:
                    logger.info("Beginning requeue", logger=logger)
                    trigger_job_requeue(os.path.join(
                        args.dump_path, "checkpoint.pth.tar"))

            if group is not None:
                dist.barrier(group=group)
            else:
                dist.barrier()

        # 2. solve label assignment via sinkhorn-knopp:
        if args.match and (iter_num == 0):
            for head in order_heads[hd_grp_idx::args.ind_groups]:
                # optimize to get labels
                if args.headcount == 1:
                    if args.rank == 0:
                        PS_a_sk = PS_a
                        PS_v_sk = PS_v
                    else:
                        PS_v_sk, PS_a_sk = None, None
                    head_a = model.module.mlp_a

                else:
                    head_a = getattr(model.module, f'mlp_a{head}')
                    head_v = getattr(model.module, f'mlp_v{head}')
                    if args.rank == 0:
                        PS_a_sk = torch.nn.functional.softmax(head_a.forward(PS_a),
                                                              dim=1, dtype=torch.float64)
                        PS_v_sk = torch.nn.functional.softmax(head_v.forward(PS_v),
                                                              dim=1, dtype=torch.float64)
                    else:
                        PS_v_sk, PS_a_sk = None, None
                # align heads of audio and video:
                match_order(args,
                            PS_v_sk,
                            PS_a_sk,
                            list(head_a.modules())[-1] if model.module.use_mlp else head_a,
                            steps=50000,
                            restarts=2,
                            logger=logger
                            )
        if args.rank == 0:
            logger.info("Optimizing via sinkhorn-knopp on master GPU")
            if os.environ['SIGNAL_RECEIVED'] == 'True':
                if args.rank == 0:
                    logger.info("Beginning requeue")
                    trigger_job_requeue(os.path.join(
                        args.dump_path, "checkpoint.pth.tar"))

            _costs = [0 for i in range(args.headcount)]
            _times = [0 for i in range(args.headcount)]


            # optimize heads
            for head in order_heads[hd_grp_idx::args.ind_groups]:
                # optimize to get labels
                if args.headcount == 1:
                    PS_a_sk = PS_a
                    PS_v_sk = PS_v
                    head_a = model.module.mlp_a
                else:
                    head_a = getattr(model.module, f'mlp_a{head}')
                    head_v = getattr(model.module, f'mlp_v{head}')
                    PS_a_sk = torch.nn.functional.softmax(head_a.forward(PS_a),
                                                          dim=1, dtype=torch.float64)
                    PS_v_sk = torch.nn.functional.softmax(head_v.forward(PS_v),
                                                          dim=1, dtype=torch.float64)

                # move activations to PS_v_sk
                torch.mul(PS_v_sk, PS_a_sk, out=PS_v_sk)
                sk_start = time.time()

                # optimize
                cost, L_head = optimize_L_sk_gpu(args, PS_v_sk, hc=head, logger=logger)
                # cost, L_head = optimize_L_sk_gpu_log(args, PS_v_sk, hc=head, logger=logger)

                # put it in correct order
                L[indices, head] = L_head.to('cuda')

                _costs[head] = cost
                _times[head] = time.time() - sk_start
                logger.info(f"Head {head}, Cost: (video): {_costs[head]:.3f}; time: {_times[head]:.3f}")

            logger.info(f"Final Cost: (video): {np.mean(_costs):.3f}; time: {np.mean(_times):.3f}")
            del PS_v
            del PS_a

            # processes wait on main process compute PS features
            # Write costs to log
            if writer:
                writer.add_scalar('train/LP-cost', np.mean(_costs), iter_num)

    if group is not None:
        dist.barrier(group=group)
    else:
        dist.barrier()

    torch.cuda.synchronize()

    if group is not None:
        torch.distributed.broadcast(L, 0, group)
    else:
        torch.distributed.broadcast(L, 0)

    if group is not None:
        dist.barrier(group=group)
    else:
        dist.barrier()
    model.module.return_features = False
    model.train()
    return L


def optimize_L_sk_gpu(args, PS, hc, logger=None):
    print('doing optimization now',flush=True)

    # create L
    N = PS.size(0)
    K = PS.size(1)
    tt = time.time()
    _K_dist = torch.ones((K, 1), dtype=torch.float64, device='cuda') # / K
    if args.distribution != 'default':
        marginals_argsort = torch.argsort(PS.sum(0))
        if (args.dist is None) or args.diff_dist_every:
            if args.distribution == 'gauss':
                if args.diff_dist_per_head:
                    _K_dists = [(torch.randn(size=(K, 1), dtype=torch.float64, device='cuda')*args.gauss_sd + 1) * N / K
                                for _ in range(args.headcount)]
                    args.dist = _K_dists
                    _K_dist = _K_dists[hc]
                else:
                    _K_dist = (torch.randn(size=(K, 1), dtype=torch.float64, device='cuda')*args.gauss_sd + 1) * N / K
                    _K_dist = torch.clamp(_K_dist, min=1)
                    args.dist = _K_dist
            if args.rank == 0:
                logger.info(f"distribution used: {_K_dist}")

        else:
            if args.diff_dist_per_head:
                _K_dist = args.dist[hc]
            else:
                _K_dist = args.dist
        _K_dist[marginals_argsort] = torch.sort(_K_dist)[0]

    beta = torch.ones((N, 1), dtype=torch.float64, device='cuda') / N
    PS.pow_(0.5*args.lamb)
    r = 1./_K_dist
    r /= r.sum()

    c = 1./N
    err = 1e6
    _counter = 0

    ones = torch.ones(N, device='cuda:0', dtype=torch.float64)
    while (err > 1e-1) and (_counter < 2000):
        alpha = r / torch.matmul(beta.t(), PS).t()
        beta_new = c / torch.matmul(PS, alpha)
        if _counter % 10 == 0:
            err = torch.sum(torch.abs((beta.squeeze() / beta_new.squeeze()) - ones)).cpu().item()
        beta = beta_new
        _counter += 1
    if args.rank == 0:
        logger.info(f"error: {err}, step : {_counter}")

    # inplace calculations
    torch.mul(PS, beta, out=PS)
    torch.mul(alpha.t(), PS, out=PS)
    newL = torch.argmax(PS, 1).cuda()

    # return back to obtain cost (optional)
    torch.mul((1./alpha).t(), PS, out=PS)
    torch.mul(PS, 1./beta, out=PS)
    sol = np.nansum(torch.log(PS[torch.arange(0, len(newL)).long(), newL]).cpu().numpy())
    cost = -(1. / args.lamb) * sol / N
    if args.rank == 0:
        logger.info(f"opt took {(time.time() - tt) / 60.} min, {_counter} iters")
    return cost, newL

@torch.no_grad()
def match_order(args, emb1, emb2_in, W2, steps=50000, restarts=2, logger=None):
    fin_perm = torch.arange(0, len(W2.bias.data)).cuda()
    if args.rank == 0:
        assert type(W2) == torch.nn.modules.linear.Linear
        K = emb1.shape[1]
        def c(a, b):
            return (torch.abs(a - b)).sum(0).sum(0)
        last_iter = 0
        cost = c(emb1, emb2_in)
        best_cost = cost
        logger.info(f'initial cost: {cost:.1f}')
        for retries in range(restarts):
            cost_try = cost.item()
            perm = torch.arange(0, K)
            emb2 = emb2_in.clone().detach()
            for _iter in range(steps):
                # what would happen if we switch cluster i with j in emb2
                [i, j] = np.random.choice(K, 2, replace=False)
                current = c(emb1[:,i], emb2[:,i])  + c(emb1[:,j], emb2[:,j])
                future =  c(emb1[:,i], emb2[:,j])  + c(emb1[:,j], emb2[:,i])
                delta = current - future
                if delta > 0:
                    # switch i and j
                    emb2[:,j], emb2[:,i] = emb2[:,i].clone().detach(), emb2[:,j].clone().detach()
                    cost_try -= delta
                    _i = int(perm[i])
                    perm[i] = int(perm[j])
                    perm[j] = _i
                    last_iter = _iter
                if _iter - last_iter > 1000:
                    break

            cost_try = c(emb1, emb2_in[:, perm])
            logger.info(f"cost of this try: {cost_try:.2f}")
            if cost_try < best_cost:
                best_cost = cost_try
                fin_perm = perm.cuda()
    dist.broadcast(fin_perm, 0)
    fin_perm = fin_perm.cpu()
    if args.rank == 0:
        logger.info(f"final cost: {best_cost:.2f}")
    W2.bias.data = W2.bias.data[fin_perm]
    W2.weight.data = W2.weight.data[fin_perm]
