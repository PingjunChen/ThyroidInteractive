# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage


def prepare_data():
    TMdata = hdf5storage.loadmat('./results/TM10.mat')
    map_arr = TMdata['TM10'][0]
    sample_arr = np.arange(50, 501, 50)

    all_ret = {}
    ret_dict = {}
    ret_dict['sample_num'] = sample_arr
    ret_dict['map'] = map_arr

    all_ret['test'] = ret_dict
    return all_ret



def draw_retrieval(all_ret):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    markers = ('o', 'v', '^', '<', '>', 's', '8', 'p')
    colors = ('r', 'b', 'g', 'k', 'm')

    ax.set_xlabel('The number of retrieved samples')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 500)
    ax.set_ylim(0.8, 1.05)
    for index, key in enumerate(all_ret):
        ax.plot(all_ret[key]['sample_num'], all_ret[key]['map'], label=key, color=colors[index], marker=markers[index])
    ax.legend(loc='lower right')

    plt.tight_layout()
    # plt.show()
    fig.savefig('retrieval_map.pdf')

if __name__ == "__main__":
    all_ret = prepare_data()
    draw_retrieval(all_ret)
