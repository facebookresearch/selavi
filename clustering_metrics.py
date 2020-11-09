# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pickle
import torch
from scipy.stats import entropy
from sklearn.metrics.cluster import (
    normalized_mutual_info_score, 
    adjusted_mutual_info_score, 
    adjusted_rand_score
)


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
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


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    from scipy.optimize import linear_sum_assignment

    assert (isinstance(flat_preds, torch.Tensor) and isinstance(flat_targets, torch.Tensor))
        
    num_samples = flat_targets.shape[0]
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)

    # return as list of tuples, out_c to gt_c
    res = []
    for i in range(len(match[0])):
        out_c, gt_c = match[0][i], match[1][i]
        res.append((out_c, gt_c))

    return res


def _acc(preds, targets, num_k, verbose=0):
    assert (isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor))

    if verbose >= 2:
        print("calling acc...")

    assert (preds.shape == targets.shape)
    assert (preds.max() < num_k and targets.max() < num_k)

    acc = int((preds == targets).sum()) / float(preds.shape[0])

    return acc


def cluster_acc(match, preds, targets, num_k=309, verbose=1):
    # reorder predictions to be same cluster assignments as gt_k
    reordered_preds = np.zeros(len(targets), dtype=np.int32)
    for pred_i, target_i in match:
      reordered_preds[preds == pred_i] = target_i
      if verbose > 1:
        print((pred_i, target_i))

    acc = _acc(torch.tensor(reordered_preds).to(torch.long), targets, num_k, True)
    return acc


def k_means(
    path="cluster_fit_PS_matrices_scratch_vgg_sound_train.pkl", 
    ncentroids=512, 
    use_all_heads=False
):
    
    # Load matrics
    PS = pickle.load(open(path, 'rb'))
    
    # SELAVI
    if use_all_heads:
        PS_v_all_heads = PS[0]
        PS_a_all_heads = PS[2]
        true_labels = PS[1].cpu().numpy()
        num_heads = len(PS_v_all_heads)
        best_nmi = 0
        best_self_labels = None
        for h in range(num_heads):
            PS_v_sk = torch.nn.functional.softmax(
                PS_v_all_heads[h], dim=1, dtype=torch.float64)
            PS_a_sk = torch.nn.functional.softmax(
                PS_a_all_heads[h], dim=1, dtype=torch.float64)
            PS_av = torch.mul(PS_v_sk, PS_a_sk)
            self_labels_np  = PS_av.argmax(1).cpu().numpy()
            nmi = normalized_mutual_info_score(
                self_labels_np, true_labels, average_method='arithmetic')
            print(f"Head {h}: {nmi}")
            if nmi > best_nmi:
                best_nmi = nmi
                best_self_labels = self_labels_np
        self_labels_np = best_self_labels
    else:
        PS_v = PS[0]
        PS_a = PS[2]
        PS_v_sk = torch.nn.functional.softmax(PS_v, dim=1, dtype=torch.float64)
        PS_a_sk = torch.nn.functional.softmax(PS_a, dim=1, dtype=torch.float64)
        PS_av = torch.mul(PS_v_sk, PS_a_sk)
        self_labels_np  = PS_av.argmax(1).cpu().numpy()
        true_labels = PS[1].cpu().numpy()

    # Get NMI and a-NMI values
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
    ari_to_labels_v = adjusted_rand_score(
        self_labels_np,
        true_labels,
    )
    print(f"NMI-tolabels: {nmi_to_labels_v}")
    print(f"aNMI-tolabels: {anmi_to_labels_v}")
    print(f"aRI-tolabels: {ari_to_labels_v}")

    # Get entropy and purtiy values
    purities = []
    entropies = []
    for sk_label in np.unique(self_labels_np):
        of_this_cluster = self_labels_np == sk_label
        size = of_this_cluster.sum()
        if size != 0:
            uniq, counts = np.unique(true_labels[of_this_cluster], return_counts=True)
            purities.append(max(counts)/sum(1.0*counts))
            entropies.append(entropy(counts/sum(1.0*counts)))
    print(f'Avg entropy: {np.mean(entropies)}   avg purity: {np.mean(purities)}')

    translate_to_low_classes = {n:a for a,n in enumerate(np.unique(true_labels))}
    true_labels = [translate_to_low_classes[n] for _,n in enumerate(true_labels)]
    print(f"Number of unique classes: {len(np.unique(true_labels))}")
    print(f"Number of centroids: {ncentroids}")

    self_labels = torch.tensor(self_labels_np)
    true_labels = torch.tensor(true_labels)
    match = _hungarian_match(self_labels, true_labels, ncentroids, ncentroids)
    clust_acc = cluster_acc(match, self_labels, true_labels, ncentroids)
    print(f'Clustering Acc: {clust_acc * 100}%')


if  __name__ == '__main__':
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
                         'Instead, it is %s.' % v)

    import argparse
    parser = argparse.ArgumentParser(description='Video Representation Learning')
    parser.register('type', 'bool', str2bool)

    parser.add_argument(
        '--path',
        default='cluster_fit_PS_matrices_scratch_vgg_sound_train.pkl',
        help='path where file is located'
    )
    parser.add_argument(
        '--ncentroids',
        default=309,
        type=int,
        help='nnum of clusters in k-means'
    )
    parser.add_argument(
        '--use_all_heads',
        default='True',
        type='bool',
        help='Use all heads'
    )
    args = parser.parse_args()
    k_means(
        path=args.path, 
        ncentroids=args.ncentroids, 
        use_all_heads=args.use_all_heads
    )
