# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 170752
    K = 309
    gauss_path = "path_gauss_ckp-100.pth"
    uniform_path = "path_uniform_ckp-100.pth"
    ckpt_dir_dict = {'uniform': uniform_path, 'gaussian': gauss_path}
    for distribtion in ['uniform', 'gaussian']:
        path = ckpt_dir_dict[distribtion]
        ckpt = torch.load(path)
        selflabels = ckpt['selflabels']
        if distribtion == 'uniform':
            target_counts = np.array([N/K for i in range(K)])
        else: # gaussian
            gauss_sd = 0.05
            target_counts = (np.random.randn(K, 1) * gauss_sd + 1) * N / K
        for i in range(10):
            u, counts = np.unique(selflabels[:, i].cpu().numpy(), return_counts=True)
            plt.plot(sorted(counts)[::-1], label="SK")
            plt.plot(sorted(target_counts)[::-1], label="Target")
            plt.xlabel('cluster-ID')
            plt.ylabel('#Assigned images')
            plt.legend()
            plt.savefig(f"cluster_vis/{distribtion}_hist_{i}.png")
            plt.clf()


if __name__ == '__main__':
    main()
