# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage


if __name__ == "__main__":
    all_ret = {}
    roiFeaRetRoot = "./roiFeas"
    sample_arr = np.arange(50, 501, 50)
    fea_list = ['gist', 'vgg16bn', 'inceptionv3', 'resnet50']
    ksh_str, cosdish_str, sdh_str = "02", "06", "10"
    cur_hash_str = sdh_str
    bit_index, bit_str = 0, "8 bits"
    save_pdf_str = "sdh_retrieval_pr_8bits.pdf"

    splits = [ele for ele in os.listdir(roiFeaRetRoot) if os.path.isdir(os.path.join(roiFeaRetRoot, ele))]
    for ele in fea_list:
        ret_dict = {}
        ttl_split_precision = np.zeros((3, 10), dtype=np.float32)
        ttl_split_recall = np.zeros((3, 10), dtype=np.float32)
        for split in splits:
            cur_pmat_path = os.path.join(roiFeaRetRoot, split, ele, 'TP'+cur_hash_str+'.mat')
            TPdata = hdf5storage.loadmat(cur_pmat_path)
            ttl_split_precision += TPdata['TP'+cur_hash_str]
            cur_rmat_path = os.path.join(roiFeaRetRoot, split, ele, 'TR'+cur_hash_str+'.mat')
            TRdata = hdf5storage.loadmat(cur_rmat_path)
            ttl_split_recall += TRdata['TR'+cur_hash_str]
        ret_dict['sample_num'] = sample_arr
        ret_dict['precision'] = ttl_split_precision / len(splits)
        ret_dict['recall'] = ttl_split_recall / len(splits)
        all_ret[ele] = ret_dict

    # drawing
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
    markers = ('o', 'v', '8', 's', 'p', '^', '<', '>')
    colors = ('r', 'b', 'g', 'k', 'm')
    # ax1.set_title('Precision-Recall')
    ax1.set_xlabel('Recall @ ' + bit_str)
    ax1.set_ylabel('Precision')
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)

    for index, key in enumerate(all_ret):
        ax1.plot(all_ret[key]['recall'][bit_index], all_ret[key]['precision'][bit_index], label=key, color=colors[index], marker=markers[index])
    ax1.legend(loc='lower left')

    ax2.set_xlabel('The number of retrieved samples')
    ax2.set_ylabel('Recall @ ' + bit_str)
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0.0, 1.05)
    for index, key in enumerate(all_ret):
        ax2.plot(all_ret[key]['sample_num'], all_ret[key]['recall'][bit_index], label=key, color=colors[index], marker=markers[index])
    ax2.legend(loc='lower right')

    ax3.set_xlabel('The number of retrieved samples')
    ax3.set_ylabel('Precision @ ' + bit_str)
    ax3.set_xlim(0, 500)
    ax3.set_ylim(0.0, 1.05)
    for index, key in enumerate(all_ret):
        ax3.plot(all_ret[key]['sample_num'], all_ret[key]['precision'][bit_index], label=key, color=colors[index], marker=markers[index])
    ax3.legend(loc='lower left')

    plt.tight_layout()
    # plt.show()
    fig.savefig(save_pdf_str)
