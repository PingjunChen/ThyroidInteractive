# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import openslide
from shapely.geometry import Polygon, Point
from pydaily import format, filesystem
from pycontour import poly_transform, cv2_transform
from skimage import io
import cv2



def gen_contour_overlay(slides_dir, annotation_dir, overlap_dir, img_level=4):
    slide_list = [ele for ele in os.listdir(slides_dir) if "tiff" in ele]
    json_list = [ele for ele in os.listdir(annotation_dir) if "json" in ele]
    if len(slide_list) != len(json_list):
        raise AssertionError("Annotation not complete")

    filesystem.overwrite_dir(overlap_dir)

    for ele in slide_list:
        slide_path = os.path.join(slides_dir, ele)
        slide_head = openslide.OpenSlide(slide_path)
        wsi_img = slide_head.read_region((0, 0), img_level, slide_head.level_dimensions[img_level])
        wsi_img = np.ascontiguousarray(np.array(wsi_img)[:,:,:3])
        json_path = os.path.join(annotation_dir, os.path.splitext(ele)[0] + ".json")
        anno_dict = format.json_to_dict(json_path)
        region_annos = anno_dict["regions"]
        if len(region_annos) <= 0:
            print("Not annotated regions in {}".format(ele))

        for cur_r in region_annos:
            r_desp = region_annos[cur_r]['desp']
            cur_cnt = region_annos[cur_r]['cnts']
            num_ps = len(cur_cnt['h'])
            cnt_arr = np.zeros((2, num_ps), np.float32)
            cnt_arr[0] = cur_cnt['h'] / np.power(2, img_level)
            cnt_arr[1] = cur_cnt['w'] / np.power(2, img_level)
            cv_cnt = cv2_transform.np_arr_to_cv_cnt(cnt_arr).astype(np.int32)
            cv2.drawContours(wsi_img, [cv_cnt], 0, (0, 255, 0), 3)
            overlay_roi_path = os.path.join(overlap_dir, os.path.splitext(ele)[0] + "_r" + cur_r + ".png")
            start_h, end_h = int(min(cnt_arr[0])), int(max(cnt_arr[0]))
            start_w, end_w = int(min(cnt_arr[1])), int(max(cnt_arr[1]))
            io.imsave(overlay_roi_path, wsi_img[start_h:end_h, start_w:end_w])



if __name__ == "__main__":
    slides_dir = "/media/pingjun/DataArchiveZizhao/Pingjun/ThyroidData/Training/Slides"
    annotation_dir = "/media/pingjun/DataArchiveZizhao/Pingjun/ThyroidData/Training/Annotations"
    overlay_dir = "/media/pingjun/DataArchiveZizhao/Pingjun/ThyroidData/Training/Overlay"

    gen_contour_overlay(slides_dir, annotation_dir, overlay_dir)
