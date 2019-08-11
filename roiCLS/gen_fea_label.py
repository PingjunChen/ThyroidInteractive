# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.io import savemat
from pydaily import format

from utils import load_fea_target, label2Onehot


def set_args():
    parser = argparse.ArgumentParser(description='Generate ROI Classification')

    parser.add_argument('--fea_dir',         type=str,   default="../data/ThyroidS1/FeasROI/L2Feas")
    parser.add_argument('--model_name',      type=str,   default="inceptionv3-bk")
    parser.add_argument('--seed',            type=int,   default=1234)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.seed)
    fea_dir = os.path.join(args.fea_dir, args.model_name)
    train_fea, train_target, train_list = load_fea_target(os.path.join(fea_dir, 'train'))
    train_label = label2Onehot(train_target)

    val_fea, val_target, val_list = load_fea_target(os.path.join(fea_dir, 'val'))
    val_label = label2Onehot(val_target)

    fea_label_dir = os.path.join(args.fea_dir, args.model_name)
    train_dat_dict = {
        'trainData': train_fea,
        'trainLabel': train_label,
        }
    savemat(os.path.join(fea_label_dir, 'trainDataLabel.mat'), train_dat_dict)
    format.list_to_txt(train_list, os.path.join(fea_label_dir, 'train_roi_list.txt'))

    test_dat_dict = {
        'testData': val_fea,
        'testLabel': val_label,
    }
    savemat(os.path.join(fea_label_dir, 'testDataLabel.mat'), test_dat_dict)
    format.list_to_txt(val_list, os.path.join(fea_label_dir, 'test_roi_list.txt'))
