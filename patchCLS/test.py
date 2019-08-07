# -*- coding: utf-8 -*-

import os, sys
import argparse
import torch
import torch.backends.cudnn as cudnn

from train_eng import validate_model
from patchloader import val_loader


def set_args():
    parser = argparse.ArgumentParser(description='Thyroid Classification')
    parser.add_argument('--seed',            type=int,   default=1234)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--model_dir',       type=str,   default="../data/ThyroidS6/Models/PatchL2Models")
    parser.add_argument('--model_name',      type=str,   default="inceptionv3")
    parser.add_argument('--model_path',      type=str,   default="Thyroid03-0.9235.pth")
    parser.add_argument('--device_id',       type=str,   default="1")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    model_full_path = os.path.join(args.model_dir, args.model_name, args.model_path)
    ft_model = torch.load(model_full_path)
    ft_model.cuda()
    test_data_loader = val_loader(args.batch_size)

    print("Start testing...")
    test_acc = validate_model(test_data_loader, ft_model)
    print("Testing accuracy is: {:.3f}".format(test_acc))
