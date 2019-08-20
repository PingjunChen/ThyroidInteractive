# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import uuid
from skimage import io
from shapely.geometry import Polygon, Point
from pydaily import format
from pycontour import poly_transform
from pyslide import patch

label_map = {
    'Benign': 1,
    'Uncertain': 2,
    'Malignant': 3
}


def gen_slide_patches(slide_dir, slide_name, patch_dir, patch_size=256):
    img_path = os.path.join(slide_dir, slide_name+".png")
    json_path = os.path.join(slide_dir, slide_name+".json")
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
                patch_cat_dir = os.path.join(patch_dir, str(label_map[cur_anno['label']]))
                if os.path.exists(patch_cat_dir) == False:
                    os.makedirs(patch_cat_dir)
                patch_path = os.path.join(patch_cat_dir, str(uuid.uuid4())[:8] + '.png')
                io.imsave(patch_path, patch_img)




if __name__ == "__main__":
    slide_dir = "/media/pingjun/DataArchiveZizhao/Pingjun/ThyroidData/AnalysisROI"
    slide_name = "1054084-2"
    patch_dir = "/media/pingjun/DataArchiveZizhao/Pingjun/ThyroidData/Patches/test"

    gen_slide_patches(slide_dir, slide_name, patch_dir)
