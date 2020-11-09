# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
import time
import torch
from torch.utils.data.dataloader import default_collate

from datasets.AVideoDataset import AVideoDataset
from model import Flatten, load_model
import utils


def collate_fn(batch):
    batch = [(d[0], d[1], d[2], d[3], d[4]) for d in batch if d is not None]
    if len(batch) == 0:
        return None
    else:
        return default_collate(batch)


def save_pickle(obj, name):
    with open(name, 'wb') as handle:
        print("Dumping data as pkl file", flush=True)
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(pkl_path):
    if os.path.exists(pkl_path):
        print(f"Loading pickle file: {pkl_path}", flush=True)
        with open(pkl_path, 'rb') as handle:
            result = pickle.load(handle)
            return result
    else:
        raise


def get_model(args, get_video_encoder_only=True, logger=None):
    
    # Load model
    model = load_model(
        vid_base_arch=args.vid_base_arch, 
        aud_base_arch=args.aud_base_arch, 
        pretrained=args.pretrained,
        num_classes=args.num_clusters,
        norm_feat=False,
        use_mlp=args.use_mlp,
        headcount=args.headcount
    )

    # Load model weights
    start = time.time()
    weight_path_type = type(args.weights_path)
    if weight_path_type == str:
        weight_path_not_none = args.weights_path != 'None' 
    else:
        weight_path_not_none = args.weights_path is not None
    if weight_path_not_none:
        print("Loading model weights")
        if os.path.exists(args.weights_path):
            ckpt_dict = torch.load(args.weights_path)
            model_weights = ckpt_dict["model"]
            args.ckpt_epoch = ckpt_dict['epoch']
            print(f"Epoch checkpoint: {args.ckpt_epoch}", flush=True)
            utils.load_model_parameters(model, model_weights)
    print(f"Time to load model weights: {time.time() - start}")

    # Put model in eval mode
    model.eval()

    # Get video encoder for video-only retrieval
    if get_video_encoder_only:
        model = model.video_network.base
        if args.pool_op == 'max': 
            pool = torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        elif args.pool_op == 'avg': 
            pool = torch.nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))
        else:
            assert("Only 'max' and 'avg' pool operations allowed")

        # Set up model
        model = torch.nn.Sequential(*[
            model.stem,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            pool,
            Flatten(),
        ])

    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    return model


def init(args, get_video_encoder_only=True, logger=None):

    # Loading Train data
    print("Loading training data")
    dataset = AVideoDataset(
        ds_name=args.dataset,
        root_dir=args.root_dir,
        mode='train',
        num_frames=args.clip_len,
        sample_rate=args.steps_bet_clips,
        num_train_clips=args.train_clips_per_video,
        train_crop_size=112,
        seed=None,
        fold=args.fold,
        colorjitter=False,
        temp_jitter=True,
        center_crop=False,
        target_fps=30,
        decode_audio=False,
    )

    print("Loading validation data")
    dataset_test = AVideoDataset(
        ds_name=args.dataset,
        root_dir=args.root_dir,
        mode='test',
        num_frames=args.clip_len,
        sample_rate=args.steps_bet_clips,
        num_spatial_crops=1,
        num_ensemble_views=args.train_clips_per_video,
        test_crop_size=112,
        seed=None,
        fold=args.fold,
        colorjitter=False,
        temp_jitter=True,
        center_crop=False,
        target_fps=30,
        decode_audio=False,
    )

    model = get_model(args, 
        get_video_encoder_only=get_video_encoder_only, logger=logger)
    return model, dataset, dataset_test


def get_features(
    args, 
    dataset, 
    model, 
    get_audio=False, 
    logger=None, 
    mode='train', 
    print_freq=250, 
    pretext=None
):

    # clear cache at beginning
    torch.cuda.empty_cache()

    # dtype
    dtype = np.float64
    N = len(dataset)
    print(f"Size of DS: {N}")

    # we need a data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn if get_audio else None,
        drop_last=False
    )
    print(f"Size of Dataloader: {len(dataloader)}")

    # 1. aggregate inputs:
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            if get_audio:
                video, audio, label, _, video_idx = batch
            else:
                video, label, _, video_idx = batch

            # Move to GPU
            if torch.cuda.is_available():
                video = video.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                video_idx = video_idx.cuda(non_blocking=True)
                if get_audio:
                    audio = audio.cuda(non_blocking=True)

            # Forward pass
            if get_audio:
                feat_v, feat_a = model(video, audio)
            else:
                feat_v = model(video)

            feat_v = feat_v.cpu()
            video_idx = video_idx.cpu()
            label = label.cpu()
            all_feat_v = feat_v.numpy()
            all_indices = video_idx.numpy().astype(np.int32)
            all_labels = label.numpy().astype(np.int32)
            if get_audio:
                feat_a = feat_a.cpu()
                all_feat_a = feat_a.numpy()

            if batch_idx == 0:
                K = feat_v.size(1)
                PS_v_np = [] 
                indices_np = [] 
                labels_np = []
                if get_audio:
                    PS_a_np = []

            # fill in arrays on main node
            PS_v_np.append(all_feat_v) 
            indices_np.append(all_indices) 
            labels_np.append(all_labels) 
            if get_audio:
                PS_a_np.append(all_feat_a)

            print(f'{batch_idx} / {len(dataloader)}', end='\r')
        print("Done collecting features")

        # Concat numpy errors
        PS_v = np.concatenate(PS_v_np, axis=0) 
        indices = np.concatenate(indices_np, axis=0) 
        labels = np.concatenate(labels_np, axis=0) 
        if get_audio:
            PS_a = np.concatenate(PS_a_np, axis=0)

        if args.save_pkl:
            if pretext is None:
                pretext = f"{args.vid_base_arch}_{args.dataset}_{args.train_clips_per_video}_{mode}"
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            save_pickle(PS_v, 
                os.path.join(args.output_dir, f"{pretext}_feats.pkl"))
            save_pickle(indices, 
                os.path.join(args.output_dir, f"{pretext}_indices.pkl"))
            save_pickle(labels, 
                os.path.join(args.output_dir, f"{pretext}_labels.pkl"))
            if get_audio:
                save_pickle(PS_a, 
                    os.path.join(args.output_dir, f"{pretext}_feats_aud.pkl"))

        if get_audio:
            return PS_v, PS_a, indices, labels
        else:
            return PS_v, indices, labels


def load_or_get_features(
    args, 
    dataset,
    model, 
    get_audio=False, 
    logger=None, 
    mode='train', 
    pretext=None
):
    # Get train features
    if pretext is None:
        pretext = f"{args.vid_base_arch}_{args.dataset}_{args.train_clips_per_video}_{mode}"
    if args.use_cache_feats:
        try: 
            features = load_pickle(
                os.path.join(args.output_dir, f"{pretext}_feats.pkl")
            )
            vid_indices = load_pickle(
                os.path.join(args.output_dir, f"{pretext}_indices.pkl")
            )
            labels = load_pickle(
                os.path.join(args.output_dir, f"{pretext}_labels.pkl")
            )
            if get_audio:
                aud_features = load_pickle(
                    os.path.join(args.output_dir, f"{pretext}_feats_aud.pkl")
                )
                return features, aud_features, vid_indices, labels
            else:
                return features, vid_indices, labels
        except: 
            if get_audio:
                features, aud_features, vid_indices, labels = get_features(
                    args, dataset, model, 
                    get_audio=get_audio, logger=logger, mode=mode
                )   
                return features, aud_features, vid_indices, labels
            else:
                features, vid_indices, labels = get_features(
                    args, dataset, model, 
                    get_audio=get_audio, logger=logger, mode=mode
                )
                return features, vid_indices, labels
    else:
        if get_audio:
            features, aud_features, vid_indices, labels = get_features(
                args, dataset, model, 
                get_audio=get_audio, logger=logger, mode=mode
            )   
            return features, aud_features, vid_indices, labels
        else:
            features, vid_indices, labels = get_features(
                args, dataset, model, 
                get_audio=get_audio, logger=logger, mode=mode
            )
            return features, vid_indices, labels


def average_features(
    args, 
    features, 
    vid_indices, 
    labels, 
    get_audio=False, 
    aud_features=None, 
    logger=None
):
    feat_dict = defaultdict(list)
    label_dict = defaultdict(list)
    if get_audio and aud_features is not None:
        aud_feat_dict = defaultdict(list)
    print(f"Total Number of features: {len(features)}")
    for i in range(len(features)):
        if args.norm_feats:
            v = features[i]
            feat = v / np.sqrt(np.sum(v**2))
            if get_audio and aud_features is not None:
                a = aud_features[i]
                feat_a = a / np.sqrt(np.sum(a**2))
        else:
            feat = features[i]
            if get_audio and aud_features is not None:
                feat_a = aud_features[i]
        label = labels[i]
        vid_idx = vid_indices[i]
        feat_dict[vid_idx].append(feat)
        label_dict[vid_idx].append(label)
        if get_audio and aud_features is not None:
            aud_feat_dict[vid_idx].append(feat_a)
        print(f'{i} / {len(features)}', end='\r')

    avg_features, avg_vid_indices, avg_labels = [], [], []
    if get_audio and aud_features is not None:
        avg_features_aud = []
    num_features = 0
    for vid_idx in feat_dict:
        stcked_feats = np.stack(feat_dict[vid_idx])
        feat = np.mean(stcked_feats, axis=0)
        vid_ix_feat_len = stcked_feats.shape[0]
        num_features += vid_ix_feat_len
        if get_audio and aud_features is not None:
            feat_a = np.mean(np.stack(aud_feat_dict[vid_idx]), axis=0)
        label = label_dict[vid_idx][0]
        avg_features.append(feat)
        avg_vid_indices.append(vid_idx)
        avg_labels.append(label)
        if get_audio and aud_features is not None:
            avg_features_aud.append(feat_a)
    avg_features = np.stack(avg_features, axis=0)
    avg_indices = np.stack(avg_vid_indices, axis=0)
    avg_labels = np.stack(avg_labels, axis=0)
    if get_audio and aud_features is not None:
        avg_features_aud = np.stack(avg_features_aud, axis=0)
    if get_audio and aud_features is not None:
        return avg_features, avg_features_aud, avg_vid_indices, avg_labels
    else:
        return avg_features, avg_vid_indices, avg_labels


def retrieval(
    train_features, 
    train_labels,
    train_vid_indices,
    val_features, 
    val_labels, 
    val_vid_indices, 
    train_aud_features=None, 
    val_aud_features=None, 
    task='v-v'
):

    assert task in ['v-a', 'a-v', 'v-v', 'a-a']
    if task in ['v-a', 'a-v', 'a-a']:
        assert(train_aud_features is not None)
        assert(val_aud_features is not None)

    if task == 'v-v':
        feat_val = val_features
        feat_train = train_features
    elif task == 'v-a':
        feat_val = val_features
        feat_train = train_aud_features
    elif task == 'a-v':
        feat_val = val_aud_features
        feat_train = train_features
    elif task == 'a-a':
        feat_val = val_aud_features
        feat_train = train_aud_features

    # Create 
    neigh = NearestNeighbors(50)
    neigh.fit(feat_train)
    recall_dict = defaultdict(list)
    retrieval_dict = {}
    for i in range(len(feat_val)):
        feat = np.expand_dims(feat_val[i], 0)
        vid_idx = val_vid_indices[i]
        vid_label = val_labels[i]
        retrieval_dict[vid_idx] = {
            'label': vid_label,
            'recal_acc': {
                '1': 0, '5': 0, '10': 0, '20': 0, '50': 0
            },
            'neighbors': {
                '1': [], '5':[], '10': [], '20': [], '50': []
            }
        }
        for recall_treshold in [1, 5, 10, 20, 50]:
            neighbors = neigh.kneighbors(feat, recall_treshold)
            neighbor_indices = neighbors[1]
            neighbor_indices = neighbor_indices.flatten()
            neighbor_labels = set([train_labels[vid_index] for vid_index in neighbor_indices])
            recall_value = 100 if vid_label in neighbor_labels else 0
            acc_value = len([1 for neigh_label in neighbor_labels if neigh_label == vid_label]) / float(len(neighbor_labels))
            retrieval_dict[vid_idx]['recal_acc'][str(recall_treshold)] = acc_value
            retrieval_dict[vid_idx]['neighbors'][str(recall_treshold)] = neighbor_indices
            recall_dict[recall_treshold].append(recall_value)
        print(f'{i} / {len(feat_val)}', end='\r')

    # Calculate mean recall values
    for recall_treshold in [1, 5, 10, 20, 50]:
        mean_recall = np.mean(recall_dict[recall_treshold]) 
        print(f"{task}: Recall @ {recall_treshold}: {mean_recall}")
    return retrieval_dict


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
    parser = argparse.ArgumentParser(description='Video Retrieval')
    parser.register('type', 'bool', str2bool)

    ### Retrieval params
    parser.add_argument('--use_cache_feats', default='False', type='bool', 
                        help='use cache features')
    parser.add_argument('--save_pkl', default='False', type='bool', 
                        help='save pickled feats')
    parser.add_argument('--avg_feats', default='True', type='bool', 
                        help='Average features of video')
    parser.add_argument('--norm_feats', default='True', type='bool', 
                        help='L2 normalize features of video')
    parser.add_argument('--pool_op', default='max', type=str, 
                        choices=['max', 'avg'],
                        help='Type of pooling operation: [max, avg]')
    parser.add_argument('--get_audio', default='False', type='bool', 
                        help='Get audio features')

    ### Dataset params
    parser.add_argument('--dataset', default='hmdb51', type=str,
                        choices=['kinetics', 'vggsound', 'kinetics_sound', 'ave', 'ucf101', 'hmdb51'],
                        help='name of dataset')
    parser.add_argument("--root_dir", type=str, default="/path/to/dataset",
                        help="root dir of dataset")
    parser.add_argument('--batch_size', default=96, type=int,
                        help='Size of batch')
    parser.add_argument('--fold', default='1', type=str,
                        help='name of dataset')
    parser.add_argument('--clip_len', default=32, type=int, 
                        help='number of frames per clip')
    parser.add_argument('--augtype', default=1, type=int, 
                        help='augmentation type (default: 1)')
    parser.add_argument('--steps_bet_clips', default=1, type=int, 
                        help='number of steps between clips in video')
    parser.add_argument('--train_clips_per_video', default=10, type=int, 
                        help='maximum number of clips per video for training')
    parser.add_argument('--val_clips_per_video', default=10, type=int, 
                        help='maximum number of clips per video for testing')
    parser.add_argument('--workers', default=0, type=int, 
                        help='number of data loading workers (default: 16)')

    ### MODEL
    parser.add_argument('--weights_path', default='', type=str,
                        help='Path to weights file')
    parser.add_argument('--vid_base_arch', default='r2plus1d_18', type=str,
                        help='Video Base Arch for A-V model')
    parser.add_argument('--aud_base_arch', default='resnet9', 
                        help='Audio Base Arch for A-V model')
    parser.add_argument('--pretrained', type='bool', default='False',
                        help="Use pre-trained models from the modelzoo")
    parser.add_argument('--use_mlp', default='True', type='bool', 
                        help='Use MLP projection head')
    parser.add_argument('--headcount', default=10, type=int, 
                        help='how many heads each modality has')
    parser.add_argument('--num_clusters', default=309, type=int,
                        help="number of clusters")

    # distributed training parameters
    parser.add_argument('--output_dir', default='./retrieval_results', 
                        help='path where to save')

    args = parser.parse_args()
    return args
