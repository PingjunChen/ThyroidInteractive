# -*- coding: utf-8 -*-

import os, sys
import shutil
import pydaily


def copy_feas(src_fea_dir, ref_fea_dir, dst_fea_dir):
    src_h5_list = pydaily.filesystem.find_ext_files(src_fea_dir, "h5")
    ref_h5_list = pydaily.filesystem.find_ext_files(ref_fea_dir, "h5")
    for ele in src_h5_list:
        h5_file = os.path.basename(ele)
        for goal in ref_h5_list:
            if h5_file in goal:
                goal_dirpath = os.path.dirname(goal)
                goal_sub_path = goal_dirpath[len(ref_fea_dir)+1:]
                dst_fea_path = os.path.join(dst_fea_dir, goal_sub_path)
                if not os.path.exists(dst_fea_path):
                    os.makedirs(dst_fea_path)
                shutil.copy(ele, dst_fea_path)



if __name__ == "__main__":
    src_fea_dir = "../data/ThyroidS6/FeasROI/L2Feas/gist"
    ref_fea_dir = "../data/ThyroidS1/FeasROI/L2Feas/resnet50"
    dst_fea_dir = "../data/ThyroidS1/FeasROI/L2Feas/gist"

    copy_feas(src_fea_dir, ref_fea_dir, dst_fea_dir)
