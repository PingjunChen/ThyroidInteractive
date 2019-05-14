# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
import openslide
from shapely.geometry import Polygon, Point
from pyslide import patch
from pydaily import format
from pycontour import poly_transform

def gen_l2_data(slides_dir, annotation_dir, level_dir, level=2, size=256):
    slide_list = [ele for ele in os.listdir(slides_dir) if "tiff" in ele]
    json_list = [ele for ele in os.listdir(annotation_dir) if "json" in ele]
    if len(slide_list) != len(json_list):
        raise AssertionError("Annotation not complete")

    for ind, ele in enumerate(slide_list):
        if ind > 0 and ind % 20 == 0:
            print("Processing {:3d}/{}".format(ind, len(slide_list)))
        slide_name = os.path.splitext(ele)[0]
        json_path = os.path.join(annotation_dir, slide_name + ".json")
        anno_dict = format.json_to_dict(json_path)
        region_annos = anno_dict["regions"]
        if len(region_annos) <= 0:
            continue

        slide_path = os.path.join(slides_dir, ele)
        slide_head = openslide.OpenSlide(slide_path)
        level_dim = slide_head.level_dimensions[level]
        img_w, img_h = level_dim

        new_anno_dict = {}
        for cur_r in region_annos:
            cur_cnt = region_annos[cur_r]['cnts']
            cur_desp = region_annos[cur_r]['desp']
            num_ps = len(cur_cnt['h'])
            cnt_arr = np.zeros((2, num_ps), np.int32)
            cnt_arr[0] = [ele/np.power(2, level) for ele in cur_cnt['h']]
            cnt_arr[1] = [ele/np.power(2, level) for ele in cur_cnt['w']]
            if np.min(cnt_arr[0]) < 0 or np.min(cnt_arr[1]) < 0:
                continue
            if np.max(cnt_arr[0]) > img_h or np.max(cnt_arr[1]) > img_w:
                continue

            poly_cnt = poly_transform.np_arr_to_poly(cnt_arr)
            start_h, start_w = np.min(cnt_arr[0]), np.min(cnt_arr[1])
            cnt_h = np.max(cnt_arr[0]) - start_h + 1
            cnt_w = np.max(cnt_arr[1]) - start_w + 1

            coors_arr = patch.wsi_coor_splitting(cnt_h, cnt_w, size, overlap_flag=True)
            for cur_h, cur_w in coors_arr:
                patch_center = Point(cur_w+start_w+size/2, cur_h+start_h+size/2)
                if patch_center.within(poly_cnt) == True:
                    new_anno_dict[cur_r] = {
                        'label': cur_desp,
                        'h': cnt_arr[0].tolist(),
                        'w': cnt_arr[1].tolist()
                    }
                    break
        if len(new_anno_dict) > 0:
            wsi_img = slide_head.read_region((0, 0), level, level_dim)
            wsi_img = np.array(wsi_img)[:,:,:3]
            io.imsave(os.path.join(level_dir, slide_name+".png"), wsi_img)
            format.dict_to_json(new_anno_dict, os.path.join(level_dir, slide_name+".json"))
        else:
            print("---{} have no proper regions---".format(slide_name))

if __name__ == "__main__":
    slides_dir = "/media/pingjun/Pingjun350/ThyroidData/Training/Slides"
    annotation_dir = "/media/pingjun/Pingjun350/ThyroidData/Training/Annotations"
    level_dir = "/media/pingjun/Pingjun350/ThyroidData/AnalysisROI"

    # gen_contour_overlay(slides_dir, annotation_dir, overlay_dir)
    gen_l2_data(slides_dir, annotation_dir, level_dir)
