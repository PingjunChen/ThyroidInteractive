# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def aggregate_feas(fea_arr):
    # maxpooling
    fuse_fea = np.mean(fea_arr, axis=0)

    return fuse_fea


def load_fea_target(fea_dir):
    fea_list, target_list = [], []
    cat_list = os.listdir(fea_dir)
    for cat in cat_list:
        sub_fea_list, sub_target_list = [], []
        cur_cat_dir = os.path.join(fea_dir, cat)
        slide_list = [ele for ele in os.listdir(cur_cat_dir) if "h5" in ele]
        for cur_slide in slide_list:
            cur_slide_path = os.path.join(cur_cat_dir, cur_slide)
            fea_dict = dd.io.load(cur_slide_path)
            fea_arr = fea_dict['feat']
            fuse_fea_arr = aggregate_feas(fea_arr)
            sub_fea_list.append(fuse_fea_arr)
            sub_target_list.append(int(cat))
        fea_list.extend(sub_fea_list)
        target_list.extend(sub_target_list)

    fea_arr = np.asarray(fea_list)
    target_arr = np.asarray(target_list)

    return fea_arr, target_arr


def label2Onehot(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded


def draw_multiclass_roc(targets, probs):
    cats = sorted(np.unique(targets))
    n_classes = len(cats)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for ind, cur_cat in enumerate(cats):
        cat_targets = np.zeros_like(targets)
        cat_targets[targets==cur_cat] = 1
        fpr[cur_cat], tpr[cur_cat], _ = roc_curve(cat_targets, probs[:, ind])
        roc_auc[cur_cat] = auc(fpr[cur_cat], tpr[cur_cat])


    # Plot of a ROC curve for a specific class
    for cur_cat in cats:
        plt.figure()
        plt.plot(fpr[cur_cat], tpr[cur_cat], label='ROC curve (area = %0.2f)' % roc_auc[cur_cat])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
