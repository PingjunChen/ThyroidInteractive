# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from statistics import mode
import hdf5storage
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_ret = {}
    RetMatPath = "./roiFeas/RetInd/TIdx02.mat"
    RetIndData = hdf5storage.loadmat(RetMatPath)
    RetResults = RetIndData['TIdx02']


    # # Consturct train and test label dictionary
    # train_label_dict = {}
    # for ind in np.arange(0, 294):
    #     train_label_dict[ind] = 1
    # for ind in np.arange(294, 339):
    #     train_label_dict[ind] = 2
    # for ind in np.arange(339, 545):
    #     train_label_dict[ind] = 3
    # test_label_dict = {}
    # for ind in np.arange(0, 77):
    #     test_label_dict[ind] = 1
    # for ind in np.arange(77, 88):
    #     test_label_dict[ind] = 2
    # for ind in np.arange(88, 126):
    #     test_label_dict[ind] = 3
    #
    # ret_cls_dict = {}
    # bit_inds = np.arange(0, 3)
    # ret_k = np.arange(1, 12, 2)
    # for b_ind in bit_inds:
    #     ret_cls_dict[b_ind] = {}
    #     for k in ret_k:
    #         cat_list = [[], [], []]
    #         for ref_i in np.arange(0, RetResults.shape[1]):
    #             ret_inds = RetResults[b_ind, ref_i, :k]
    #             ret_label = [train_label_dict[ele] for ele in ret_inds]
    #             ret_label = mode(ret_label)
    #             cat_list[test_label_dict[ref_i]-1].append(ret_label == test_label_dict[ref_i])
    #         cat_acc_list = [np.sum(ele)/len(ele) for ele in cat_list]
    #         ret_cls_dict[b_ind][k] = cat_acc_list
    #
    #
    # (fig, subplots) = plt.subplots(1, len(bit_inds), figsize=(16, 5))
    # for ind, method in enumerate(bit_inds):
    #     ax = subplots[ind]
    #     cat_acc_list = [[], [], []]
    #     for k in ret_k:
    #         cat_acc_list[0].append(ret_cls_dict[ind][k][0])
    #         cat_acc_list[1].append(ret_cls_dict[ind][k][1])
    #         cat_acc_list[2].append(ret_cls_dict[ind][k][2])
    #     ax.plot(ret_k, cat_acc_list[0],  "ro-", label="Benign")
    #     ax.plot(ret_k, cat_acc_list[1],  "go-", label="Uncertain")
    #     ax.plot(ret_k, cat_acc_list[2],  "bo-", label="Malignant")
    #     ax.legend()
    #     # ax.axis('tight')
    # plt.show()
