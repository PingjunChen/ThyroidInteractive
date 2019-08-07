# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import deepdish as dd
from skimage import io, transform
from PIL import Image
from shapely.geometry import Polygon, Point
Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.simplefilter("ignore")
import time
import gist

from pydaily import format
from pycontour import poly_transform
from pyslide import patch


label_map = {
    'Benign': 1,
    'Uncertain': 2,
    'Malignant': 3
}


class PatchDataset(data.Dataset):
    """
    Dataset for thyroid slide testing. Thyroid slides would be splitted into multiple patches.
    Prediction is made on these splitted patches.
    """

    def __init__(self, slide_patches):
        self.patches = slide_patches
        self.rgb_mean = (0.800, 0.600, 0.800)
        self.rgb_std = (0.125, 0.172, 0.100)
        self.transform = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((299, 299)), transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        sample = self.patches[idx,...]
        if self.transform:
            sample = self.transform(sample)

        return sample


def extract_deep_feas(model, x, model_name):
    model.eval()
    if "resnet" in model_name:
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        fea = x.reshape(x.size(0), -1)
        logit = model.fc(fea)
        prob = F.softmax(logit, dim=-1)
    elif "vgg" in model_name:
        x = model.features(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        fea = model.classifier[:4](x)
        logit = model.classifier[4:](fea)
        prob = F.softmax(logit, dim=-1)
    elif "inception" in model_name:
        if model.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=model.training)
        # N x 2048 x 1 x 1
        fea = torch.flatten(x, 1)
        # N x 2048
        logit = model.fc(fea)
        prob = F.softmax(logit, dim=-1)
    else:
        raise AssertionError("Unknown model name {}".format(model_name))

    return prob, logit, fea


def pred_feas(model, patches, args):
    slide_dset = PatchDataset(np.asarray(patches))
    dset_loader = data.DataLoader(slide_dset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, drop_last=False)

    probs, logits, vecs = [], [], []
    with torch.no_grad():
        for ind, inputs in enumerate(dset_loader):
            x = Variable(inputs.cuda())
            start_time = time.time()
            prob, logit, fea = extract_deep_feas(model, x, args.model_name)
            elapsed_time = time.time() - start_time
            # print("Takes {} for {} images.".format(elapsed_time, len(prob)))

            probs.extend(prob.cpu().numpy())
            logits.extend(logit.cpu().numpy())
            vecs.extend(fea.cpu().numpy())

    return probs, logits, vecs


def sort_by_prob(BBoxes, ClsProbs, ClsLogits, FeaVecs):
    fea_dict = {}
    norm_prob_list = [ele[0] for ele in ClsProbs]
    sorting_indx = np.argsort(norm_prob_list)

    fea_dict["prob"] = np.array([ClsProbs[ind] for ind in sorting_indx])
    fea_dict["logit"] = np.array([ClsLogits[ind] for ind in sorting_indx])
    fea_dict["feat"] = np.array([FeaVecs[ind] for ind in sorting_indx])
    fea_dict["bbox"] = np.array([BBoxes[ind] for ind in sorting_indx])

    return fea_dict


def gen_deep_features(roi_dir, fea_dir, mode, ft_model, args):
    fea_dir = os.path.join(fea_dir, args.model_name, mode)
    data_dir = os.path.join(roi_dir, mode)
    img_list = [ele for ele in os.listdir(data_dir) if "png" in ele]

    for ind, ele in enumerate(img_list):
        if ind > 0 and ind % 10 == 0:
            print("processing {}/{}".format(ind, len(img_list)))

        cur_img_path = os.path.join(data_dir, ele)
        img_name = os.path.splitext(ele)[0]
        cur_anno_path = os.path.join(data_dir, img_name+".json")

        if not (os.path.exists(cur_img_path) and os.path.exists(cur_anno_path)):
            print("File not available")

        img = io.imread(cur_img_path)
        anno_dict = format.json_to_dict(cur_anno_path)
        for cur_r in anno_dict:
            cur_anno = anno_dict[cur_r]
            region_label = str(label_map[cur_anno['label']])
            region_name = "_".join([img_name, 'r'+cur_r])
            x_coors, y_coors = cur_anno['w'], cur_anno['h']
            cnt_arr = np.zeros((2, len(x_coors)), np.int32)
            cnt_arr[0], cnt_arr[1] = y_coors, x_coors
            poly_cnt = poly_transform.np_arr_to_poly(cnt_arr)

            start_x, start_y = min(x_coors), min(y_coors)
            cnt_w = max(x_coors) - start_x + 1
            cnt_h = max(y_coors) - start_y + 1
            coors_arr = patch.wsi_coor_splitting(cnt_h, cnt_w, args.patch_size, overlap_flag=True)

            BBoxes, patch_list = [], []
            for cur_h, cur_w in coors_arr:
                patch_start_w, patch_start_h = cur_w+start_x, cur_h + start_y
                patch_center = Point(patch_start_w+args.patch_size/2, patch_start_h+args.patch_size/2)
                if patch_center.within(poly_cnt) == True:
                    patch_img = img[patch_start_h:patch_start_h+args.patch_size, patch_start_w:patch_start_w+args.patch_size, :]
                    # patch_img = transform.resize(patch_img, (256, 256))
                    BBoxes.append([patch_start_h, patch_start_w, args.patch_size, args.patch_size])
                    patch_list.append(patch_img)

            ClsProbs, ClsLogits, FeaVecs = pred_feas(ft_model, patch_list, args)
            fea_dict = sort_by_prob(BBoxes, ClsProbs, ClsLogits, FeaVecs)

            # save features
            cat_fea_dir = os.path.join(fea_dir, region_label)
            if not os.path.exists(cat_fea_dir):
                os.makedirs(cat_fea_dir)
            dd.io.save(os.path.join(cat_fea_dir, region_name+".h5"), fea_dict)


def gen_gist_features(roi_dir, fea_dir, mode, args):
    fea_dir = os.path.join(fea_dir, args.model_name, mode)
    data_dir = os.path.join(roi_dir, mode)
    img_list = [ele for ele in os.listdir(data_dir) if "png" in ele]

    for ind, ele in enumerate(img_list):
        if ind > 0 and ind % 10 == 0:
            print("processing {}/{}".format(ind, len(img_list)))

        cur_img_path = os.path.join(data_dir, ele)
        img_name = os.path.splitext(ele)[0]
        cur_anno_path = os.path.join(data_dir, img_name+".json")

        if not (os.path.exists(cur_img_path) and os.path.exists(cur_anno_path)):
            print("File not available")

        img = io.imread(cur_img_path)
        anno_dict = format.json_to_dict(cur_anno_path)
        for cur_r in anno_dict:
            cur_anno = anno_dict[cur_r]
            region_label = str(label_map[cur_anno['label']])
            region_name = "_".join([img_name, 'r'+cur_r])
            x_coors, y_coors = cur_anno['w'], cur_anno['h']
            cnt_arr = np.zeros((2, len(x_coors)), np.int32)
            cnt_arr[0], cnt_arr[1] = y_coors, x_coors
            poly_cnt = poly_transform.np_arr_to_poly(cnt_arr)

            start_x, start_y = min(x_coors), min(y_coors)
            cnt_w = max(x_coors) - start_x + 1
            cnt_h = max(y_coors) - start_y + 1
            coors_arr = patch.wsi_coor_splitting(cnt_h, cnt_w, args.patch_size, overlap_flag=True)

            Feas, BBoxes = [], []
            for cur_h, cur_w in coors_arr:
                patch_start_w, patch_start_h = cur_w+start_x, cur_h + start_y
                patch_center = Point(patch_start_w+args.patch_size/2, patch_start_h+args.patch_size/2)
                if patch_center.within(poly_cnt) == True:
                    patch_img = img[patch_start_h:patch_start_h+args.patch_size, patch_start_w:patch_start_w+args.patch_size, :]
                    patch_desp = gist.extract(patch_img)
                    Feas.append(patch_desp)
                    BBoxes.append([patch_start_h, patch_start_w, args.patch_size, args.patch_size])
            fea_dict = {
                'feat': np.asarray(Feas),
                'bbox': np.asarray(BBoxes)
            }

            # save features
            cat_fea_dir = os.path.join(fea_dir, region_label)
            if not os.path.exists(cat_fea_dir):
                os.makedirs(cat_fea_dir)
            dd.io.save(os.path.join(cat_fea_dir, region_name+".h5"), fea_dict)
