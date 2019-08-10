# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io, transform
from pydaily import format


def extract_patches(data_dir, file_name):
    json_path = os.path.join(data_dir, file_name+".json")
    anno_dict = format.json_to_dict(json_path)
    region_annos = anno_dict["1"]
    h_max, h_min = max(region_annos['h']), min(region_annos['h'])
    w_max, w_min = max(region_annos['w']), min(region_annos['w'])
    h_len, w_len = int((h_max - h_min) / 3), int((w_max - w_min) / 3)
    crop_coors = [(h_min, w_min), (h_min, w_min+w_len),  (h_min, w_min+w_len*2),
                  (h_min+h_len, w_min), (h_min+h_len, w_min+w_len),  (h_min+h_len, w_min+w_len*2),
                  (h_min+h_len*2, w_min), (h_min+h_len*2, w_min+w_len),  (h_min+h_len*2, w_min+w_len*2)]

    wsi_img = io.imread(os.path.join(data_dir, file_name+".png"))
    for ind, cur_coors in enumerate(crop_coors):
        h_start, w_start = cur_coors[0], cur_coors[1]
        cur_patch_img = wsi_img[h_start:h_start+h_len, w_start:w_start+w_len]
        cur_patch_img = transform.resize(cur_patch_img, (448, 448))
        patch_save_path = os.path.join(data_dir, file_name+'_'+str(ind+1)+".png")
        io.imsave(patch_save_path, cur_patch_img)


if __name__ == "__main__":
    data_dir = json_path = "/home/pingjun/Dropbox (UFL)/Poems/WSI-ROI/drawio/wsi_roi_representation_gen"
    file_name = "1210100"
    extract_patches(data_dir, file_name)
