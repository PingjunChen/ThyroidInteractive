# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.io import savemat

from utils import load_fea_target, label2Onehot


def set_args():
    parser = argparse.ArgumentParser(description='Thyroid Classification')

    parser.add_argument('--fea_dir',         type=str,   default="../data/ThyroidS5/FeasROI/L2Feas")
    parser.add_argument('--model_name',      type=str,   default="resnet50")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    fea_dir = os.path.join(args.fea_dir, args.model_name)
    train_fea, train_target = load_fea_target(os.path.join(fea_dir, 'train'))
    train_label = label2Onehot(train_target)
    val_fea, val_target = load_fea_target(os.path.join(fea_dir, 'val'))
    val_label = label2Onehot(val_target)

    fea_label_dir = os.path.join(args.fea_dir, args.model_name)
    train_dat_dict = {
        'trainData': train_fea,
        'trainLabel': train_label,
        }
    savemat(os.path.join(fea_label_dir, 'trainDataLabel.mat'), train_dat_dict)
    test_dat_dict = {
        'testData': val_fea,
        'testLabel': val_label,
    }
    savemat(os.path.join(fea_label_dir, 'testDataLabel.mat'), test_dat_dict)
