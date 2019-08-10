# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from time import time

from utils import load_fea_target


def set_args():
    parser = argparse.ArgumentParser(description='Thyroid ROI Classification')
    parser.add_argument('--fea_dir',         type=str,   default="../data/ThyroidS1/FeasROI/L2Feas")
    # parser.add_argument('--model_name',      type=str,   default="resnet50")
    parser.add_argument('--seed',            type=int,   default=1234)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.seed)
    fea_methods = ["vgg16bn", "inceptionv3", "resnet50", "gist"]
    (fig, subplots) = plt.subplots(1, len(fea_methods), figsize=(20, 5))
    for ind, method in enumerate(fea_methods):
        # load deep features
        fea_dir = os.path.join(args.fea_dir, method)
        train_fea, train_target = load_fea_target(os.path.join(fea_dir, 'train'))
        val_fea, val_target = load_fea_target(os.path.join(fea_dir, 'val'))
        all_feas = np.concatenate((train_fea, val_fea), axis=0)
        all_targets = np.concatenate((train_target, val_target), axis=0)

        # given color to each catgory
        red = all_targets == 1
        green = all_targets == 2
        blue = all_targets == 3

        t0 = time()
        tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=30.0)
        embeds = tsne.fit_transform(all_feas)
        t1 = time()
        print("TSNE takes {:.3f} sec".format(t1 - t0))
        ax = subplots[ind]
        ax.scatter(embeds[red, 0], embeds[red, 1], c="r", label="Benign")
        ax.scatter(embeds[green, 0], embeds[green, 1], c="g", label="Uncertain")
        ax.scatter(embeds[blue, 0], embeds[blue, 1], c="b", label="Malignant")

        ax.legend()
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.set_title(method)
        ax.axis('tight')
    plt.show()
