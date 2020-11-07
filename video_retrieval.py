# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from src.retrieval_utils import (
    average_features, 
    init, 
    load_or_get_features, 
    parse_args, 
    retrieval
)


def main(args, logger=None):

    # Get model and datasets
    model, dataset, dataset_test = init(args, 
        get_video_encoder_only=True, logger=logger)

    # Get train features
    train_features, train_vid_indices, train_labels = load_or_get_features(
        args, dataset, model, 
        logger=logger, mode='train', get_audio=args.get_audio
    )

    # Get val features
    val_features, val_vid_indices, val_labels = load_or_get_features(
        args, dataset_test, model, 
        logger=logger, mode='test', get_audio=args.get_audio
    )

    # Average features to get mean feat per video
    print("Averaging features")
    train_features, train_vid_indices, train_labels = average_features(
        args, train_features, train_vid_indices, train_labels, 
        get_audio=args.get_audio, aud_features=None, logger=logger
    )
    val_features, val_vid_indices, val_labels = average_features(
        args, val_features, val_vid_indices, val_labels, 
        get_audio=args.get_audio, aud_features=None, logger=logger
    )

    # Get retrieval benchmarks
    retrieval(
        train_features, 
        train_labels,
        train_vid_indices,
        val_features, 
        val_labels, 
        val_vid_indices,
        train_aud_features=None, 
        val_aud_features=None, 
        task='v-v'
    )


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    args.get_audio = False
    
    # Run main script
    main(args, logger=None)
