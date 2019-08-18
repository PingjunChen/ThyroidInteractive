# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    save_pdf_str = "cls_power_exponent_cmp.pdf"
    feature_methods = ["vgg16bn", "inceptionv3", "resnet50", "GIST"]
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))

    cls_methods = ["DT", "SVM", "RF", "MLP"]
    tick_locs = np.arange(1, len(cls_methods)+1)
    p1_cls_list = [[0.925, 0.945, 0.952, 0.956], [0.937, 0.947, 0.945, 0.957],
                   [0.944, 0.943, 0.957, 0.961], [0.752, 0.555, 0.777, 0.817]]
    p2_cls_list = [[0.929, 0.956, 0.954, 0.957], [0.937, 0.950, 0.945, 0.954],
                   [0.929, 0.954, 0.957, 0.959], [0.751, 0.555, 0.776, 0.779]]
    p3_cls_list = [[0.928, 0.960, 0.954, 0.959], [0.937, 0.950, 0.945, 0.951],
                   [0.929, 0.951, 0.957, 0.957], [0.752, 0.555, 0.776, 0.641]]

    # drawing on different methods
    for ind, method in enumerate(feature_methods):
        ax = axes[ind]
        ax.plot(tick_locs, p1_cls_list[ind], label="p=1")
        ax.plot(tick_locs, p2_cls_list[ind], label="p=2")
        ax.plot(tick_locs, p3_cls_list[ind], label="p=3")
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(cls_methods, rotation=45)
        ax.legend()
        ax.set_title(method)

    # plt.show()
    plt.tight_layout()
    fig.savefig(save_pdf_str)
