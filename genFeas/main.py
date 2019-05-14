# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from fea_gen import gen_features


def set_args():
    parser = argparse.ArgumentParser(description='Thyroid Classification')
    parser.add_argument('--seed',            type=int,   default=1234)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--patch_size',      type=int,   default=256)
    parser.add_argument('--model_dir',       type=str,   default="../data/Models/PatchL2Models")
    parser.add_argument('--data_name',       type=str,   default="Thyroid")
    parser.add_argument('--model_name',      type=str,   default="resnet34")
    parser.add_argument('--model_path',      type=str,   default="Thyroid-ft-0.891.pth")

    parser.add_argument('--roi_dir',         type=str,   default="../data/ImgsROI")
    parser.add_argument('--data_mode',       type=str,   default="val")
    parser.add_argument('--fea_dir',         type=str,   default="../data/Feas/L2Feas")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    model_full_path = os.path.join(args.model_dir, args.data_name, args.model_name, args.model_path)
    ft_model = torch.load(model_full_path)
    ft_model.cuda()

    print("Starting feature generation...")
    gen_features(args.roi_dir, args.fea_dir, args.data_mode, ft_model, args)
