# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
import json
import random
import torch


def get_data(filename, meta_filename=None):
    with open(filename, 'rb') as handle:
        clusters = pickle.load(handle)

    epochs = [int(epoch) for epoch in clusters.keys()]
    epochs.sort()

    last_epoch = clusters[f'{epochs[-1]}']

    for i in range(len(last_epoch)):
        random.shuffle(last_epoch[i])

    if meta_filename is not None:
        with open("meta-classes.json", 'rb') as handle:
            meta = json.load(handle)
        keys = list(meta.keys())
        for k in keys:
            new_key = ('' + k).replace(' ', '_')
            meta[new_key] = meta.pop(k)
    else:
        meta = {}
        for c in last_epoch:
            for v in c:
                meta[v[3]] = "people"
    print(set(meta.values()))
    data = {"clusters": last_epoch, "metaclasses": meta}
    return data


def main(vgg_sound_path=None, kinetics_path=None):
    vgg_sound = None
    kinetics = None
    vgg_sound = get_data(vgg_sound_path, 'meta-classes.json')
    kinetics = get_data(kinetics_path, None)
    assert(vgg_sound is not None and kinetics is not None)

    with open('./data/clusters.js', 'w') as handle:
        handle.write("function getVGGSoundClusterData() { return " + json.dumps(vgg_sound) + "; }\n")
        handle.write("function getKineticsClusterData() { return " + json.dumps(kinetics) + "; }")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess clusters')

    ### Retrieval params
    parser.add_argument(
        '--kinetics_path', 
        default='./selavi_kin_27264459.pkl',
        type=str, 
        help='path to kineitcs cluster file'
    )
    parser.add_argument(
        '--vgg_sound_path', 
        default='./selavi_vgg_sounds_27065177.pkl',
        type=str, 
        help='path to kineitcs cluster file'
    )
    
    args = parser.parse_args()
    main(vgg_sound_path=args.vgg_sound_path, kinetics_path=args.kinetics_path)
