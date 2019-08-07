# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage


def prepare_data():
    TPdata = hdf5storage.loadmat('./results/TP10.mat')
    precision_arr = TPdata['TP10'][0]
    TRdata = hdf5storage.loadmat('./results/TR10.mat')
    recall_arr = TRdata['TR10'][0]
    sample_arr = np.arange(50, 501, 50)

    all_ret = {}
    ret_dict = {}
    ret_dict['sample_num'] = sample_arr
    ret_dict['precision'] = precision_arr
    ret_dict['recall'] = recall_arr

    all_ret['test'] = ret_dict
    return all_ret



def draw_retrieval(all_ret):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
    markers = ('o', 'v', '^', '<', '>', 's', '8', 'p')
    colors = ('r', 'b', 'g', 'k', 'm')

    # ax1.set_title('Precision-Recall')
    ax1.set_xlabel('Recall @ 8 bits')
    ax1.set_ylabel('Precision')
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    # ax1.set_yticklabels([str(ele) for ele in np.arange(0.0, 1.05, 0.1)])
    for index, key in enumerate(all_ret):
        ax1.plot(all_ret[key]['recall'], all_ret[key]['precision'], label=key, color=colors[index], marker=markers[index])
    ax1.legend(loc='lower left')

    ax2.set_xlabel('The number of retrieved samples')
    ax2.set_ylabel('Recall @ 8 bits')
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0.0, 1.05)
    for index, key in enumerate(all_ret):
        ax2.plot(all_ret[key]['sample_num'], all_ret[key]['recall'], label=key, color=colors[index], marker=markers[index])
    ax2.legend(loc='lower right')

    ax3.set_xlabel('The number of retrieved samples')
    ax3.set_ylabel('Precision @ 8 bits')
    ax3.set_xlim(0, 500)
    ax3.set_ylim(0.0, 1.05)
    for index, key in enumerate(all_ret):
        ax3.plot(all_ret[key]['sample_num'], all_ret[key]['precision'], label=key, color=colors[index], marker=markers[index])
    ax3.legend(loc='lower left')

    plt.tight_layout()
    # plt.show()
    fig.savefig('retrieval_pr.pdf')

if __name__ == "__main__":
    all_ret = prepare_data()
    draw_retrieval(all_ret)
