# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import glob
import os
import pickle
import torch


def get_clusters(ckpt_path='', num_clusters=400):
    result_dict = {}

    kinetic_train_path = 'datasets/data/kinetics_train.txt'
    with open(kinetic_train_path, 'r') as f:
        kinetics_paths = f.readlines()

    vid_valid_file = f'datasets/data/kinetics_valid.pkl'

    if os.path.exists(vid_valid_file):
        with open(vid_valid_file, 'rb') as handle:
            valid_indices = pickle.load(handle)

    final_kinetics_paths = []
    for ix in valid_indices:
        final_kinetics_paths.append(kinetics_paths[ix])

    ckpt = torch.load(ckpt_path)
    self_labels = ckpt['selflabels'][:, 0]
    epoch = ckpt['epoch']
    full_list = []
    for cluster_i in range(num_clusters):
        print(f"Epoch: {epoch}, cluster: {cluster_i}")
        cluster_indices = (self_labels == cluster_i).nonzero().cpu().numpy()
        cluster_list = []
        for index in cluster_indices[:, 0]:
            path = final_kinetics_paths[index]
            vid_name = path.split('.')[0].split('/')[-1]
            gt_class = path.split('.')[0].split('/')[-2]
            youtube_id = '_'.join(vid_name.split('_')[0:-2])
            start_time = vid_name.split('_')[-2]
            end_time = vid_name.split('_')[-1]
            res_tuple = (youtube_id, int(start_time), int(end_time), gt_class)
            cluster_list.append(res_tuple)
        full_list.append(cluster_list)
    result_dict[str(epoch)] = full_list
    
    with open(f'cluster_vis/selavi_kinetics.pkl', 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Get clusters')

    ### Retrieval params
    parser.add_argument(
        '--ckpt_path', 
        default='',
        type=str, 
        help='path to checkpoint'
    )
    parser.add_argument(
        '--num_clusters', 
        default=400,
        type=int, 
        help='number of clusters'
    )
    args = parser.parse_args()

    get_clusters(
        ckpt_path=args.ckpt_path,
        num_clusters=args.num_clusters
    )
