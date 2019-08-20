# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np

from fea_gen import gen_gist_features

def set_args():
    parser = argparse.ArgumentParser(description='Gist feature extraction')
    parser.add_argument('--seed',            type=int,   default=1234)
    parser.add_argument('--patch_size',      type=int,   default=256)
    parser.add_argument('--model_name',      type=str,   default="gist")

    parser.add_argument('--roi_dir',         type=str,   default="../data/ThyroidS1/ImgsROI")
    parser.add_argument('--data_mode',       type=str,   default="val")
    parser.add_argument('--fea_dir',         type=str,   default="../data/ThyroidS1/FeasROI/L2Feas")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    print("Starting gist feature generation...")
    gen_gist_features(args.roi_dir, args.fea_dir, args.data_mode, args)
