# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import uuid, shutil
from skimage import io, transform
from scipy import ndimage
from PIL import Image
from shapely.geometry import Point
from pydaily import format
from pycontour import poly_transform
from pyslide import patch
Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.simplefilter("ignore")


label_map = {
    'Benign': 1,
    'Uncertain': 2,
    'Malignant': 3
}



def organize_imgs(imgs_dir, img_list, mode):
    save_dir = os.path.join(imgs_dir, mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for ele in img_list:
        img_path = os.path.join(imgs_dir, ele)
        img_name = os.path.splitext(ele)[0]
        json_path = os.path.join(imgs_dir, img_name+".json")
        shutil.move(img_path, save_dir)
        shutil.move(json_path, save_dir)



def split_trainval(imgs_dir, ratio=0.2, organize=False):
    img_list = [ele for ele in os.listdir(imgs_dir) if "png" in ele]
    np.random.shuffle(img_list)
    split_point = int(round((1-ratio)*len(img_list)))
    train_list, val_list = img_list[:split_point], img_list[split_point:]

    if organize == True:
        organize_imgs(imgs_dir, train_list, 'train')
        organize_imgs(imgs_dir, val_list, 'val')

    return train_list, val_list



def gen_patches(imgs_dir, patch_dir, img_list, dset, patch_size=256):
    for ind, ele in enumerate(img_list):
        if ind > 0 and ind % 10 == 0:
            print("processing {}/{}".format(ind, len(img_list)))
        img_path = os.path.join(imgs_dir, ele)
        img_name = os.path.splitext(ele)[0]
        json_path = os.path.join(imgs_dir, img_name+".json")

        if not (os.path.exists(img_path) and os.path.exists(json_path)):
            print("File not available")

        img = io.imread(img_path)
        anno_dict = format.json_to_dict(json_path)
        for cur_r in anno_dict:
            cur_anno = anno_dict[cur_r]
            x_coors, y_coors = cur_anno['w'], cur_anno['h']
            cnt_arr = np.zeros((2, len(x_coors)), np.int32)
            cnt_arr[0], cnt_arr[1] = y_coors, x_coors
            poly_cnt = poly_transform.np_arr_to_poly(cnt_arr)

            start_x, start_y = min(x_coors), min(y_coors)
            cnt_w = max(x_coors) - start_x + 1
            cnt_h = max(y_coors) - start_y + 1
            coors_arr = patch.wsi_coor_splitting(cnt_h, cnt_w, patch_size, overlap_flag=True)
            for cur_h, cur_w in coors_arr:
                patch_start_w, patch_start_h = cur_w+start_x, cur_h + start_y
                patch_center = Point(patch_start_w+patch_size/2, patch_start_h+patch_size/2)
                if patch_center.within(poly_cnt) == True:
                    patch_img = img[patch_start_h:patch_start_h+patch_size, patch_start_w:patch_start_w+patch_size, :]
                    patch_img = transform.resize(patch_img, (256, 256))
                    patch_cat_dir = os.path.join(patch_dir, dset, str(label_map[cur_anno['label']]))
                    if os.path.exists(patch_cat_dir) == False:
                        os.makedirs(patch_cat_dir)
                    patch_path = os.path.join(patch_cat_dir, str(uuid.uuid4())[:8] + '.png')
                    io.imsave(patch_path, patch_img)


if __name__ == "__main__":
    np.random.seed(1239)
    imgs_dir = "/media/pingjun/Pingjun350/ThyroidData/AnalysisROI"
    patch_dir = "/media/pingjun/Pingjun350/ThyroidData/ThyroidS6/PatchesL2"


    # # generate patches
    # train_list, val_list = split_trainval(imgs_dir, ratio=0.2)
    # gen_patches(imgs_dir, patch_dir, val_list, "val", patch_size=256)
    # gen_patches(imgs_dir, patch_dir, train_list, "train", patch_size=256)

    # move slides into train/val
    train_list, val_list = split_trainval(imgs_dir, ratio=0.2, organize=True)
