# -*- coding: utf-8 -*-
"""找出核型图中染色体着丝粒的位置

Author: Lei Wang
Date: Dec 19, 2024
"""

__author__ = "王磊"
__copyright__ = "Copyright 2024 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import os
import math
import cv2
import numpy as np

from karyotype import Karyotype
import banding_pattern_extraction as bpe
from utils.chromo_cv_utils import (
    cv_imread,
    cv_imwrite,
    calc_intersections_between_skeleton_vline_and_contour,
)


KYT_IMG_DIR_FP = r"E:/染色体测试数据/241219-骨架垂线办法找着丝粒centromere/src_kyt"
OUTPUT_DIR_FP = r"E:/染色体测试数据/241219-骨架垂线办法找着丝粒centromere/output"

DRAW_SKELETON = True
DRAW_CENTROMERE = True
DRAW_CONTOUR = True
DRAW_VLINE = True

if not os.path.isdir(KYT_IMG_DIR_FP):
    raise ValueError(f"Invalid directory: {KYT_IMG_DIR_FP}")

kyt_img_fps = [os.path.join(KYT_IMG_DIR_FP, f) for f in os.listdir(KYT_IMG_DIR_FP) if f.endswith(".JPG")]

for kyt_img_fp in kyt_img_fps:
    # 对核型图进行解析
    karyotype_chart = Karyotype(kyt_img_fp)
    karyotype_chart.read_karyotype()

    # 准备核型图画布
    canvas = cv_imread(kyt_img_fp)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR) if len(canvas.shape) == 2 else canvas

    # 处理每一个染色体轮廓
    for chromo_cntr_dict in karyotype_chart.chromo_cntr_dicts:
        chromo_img = chromo_cntr_dict["wbg_cropped"]
        chromo_img_size = chromo_img.shape[:2]

        # 求骨架
        result = bpe.get_banding_pattern(chromo_img, pixel_sampling=5, chromosome_threshold=253)
        skeleton = np.full(shape=chromo_img_size, fill_value=0, dtype=np.uint8)
        r = result['r_sampled'] # r means row -> X
        c = result['c_sampled'] # c means column -> Y
        skeleton_points = [(int(x), int(y)) for x, y in zip(c, r)]
        # 去重
        skeleton_points = list(set(skeleton_points))
        # 按y坐标排序
        skeleton_points.sort(key=lambda x: x[1])
        # 掐头去尾
        h, w = chromo_img_size
        head_limit = 0.1 * h
        tail_limit = h - (0.25 * h)
        # final skeleton points
        skeleton_points = [coord for coord in skeleton_points if head_limit < coord[1] < tail_limit]

        chromo_cntr = chromo_cntr_dict["cropped_cntr"][0]
        # 核型图上的染色体轮廓外包正矩形左上角坐标
        chromo_kyt_bbox_x, chromo_kyt_bbox_y, _, _ = chromo_cntr_dict["rect"]
        # 割下来的染色体轮廓外包正矩形左上角坐标
        chromo_bbox_x, chromo_bbox_y, _, _ = cv2.boundingRect(chromo_cntr)

        # 求单根染色体到核型图的转换
        delta_x = chromo_kyt_bbox_x - chromo_bbox_x
        delta_y = chromo_kyt_bbox_y - chromo_bbox_y


        # 在核型图中画出骨架
        # 将骨架点转换到核型图中的坐标
        skeleton_points_in_kyt = [
            (int(x + delta_x), int(y + delta_y))
            for x, y in skeleton_points
        ]
        # draw skeleton
        if DRAW_SKELETON:
            for x, y in skeleton_points_in_kyt:
                cv2.circle(canvas, (x, y), 1, (255, 0, 0), -1)

        # 求骨架的垂线同轮廓的交点集合
        skeleton_points = np.array(skeleton_points)
        vline_length = int(math.sqrt(h**2 + w**2))
        try:
            intersections = calc_intersections_between_skeleton_vline_and_contour(chromo_cntr, skeleton_points, 2, vline_length)
        except Exception as e:
            print(f"Error: {e}")
            continue

        # 画出骨架的垂线
        if DRAW_VLINE:
            min_distance_intersection = min(intersections, key=lambda x: x["distance"])
            line_start = (int(min_distance_intersection["intersection_points"][0][0]+delta_x), int(min_distance_intersection["intersection_points"][0][1]+delta_y))
            line_end = (int(min_distance_intersection["intersection_points"][1][0]+delta_x), int(min_distance_intersection["intersection_points"][1][1]+delta_y))
            cv2.line(canvas, line_start, line_end, (255, 255, 0), 1)
            # for intersection in intersections:
            #     line_start = (int(intersection["intersection_points"][0][0]+delta_x), int(intersection["intersection_points"][0][1]+delta_y))
            #     line_end = (int(intersection["intersection_points"][1][0]+delta_x), int(intersection["intersection_points"][1][1]+delta_y))
            #     cv2.line(canvas, line_start, line_end, (255, 255, 0), 1)

        # 求染色体上的着丝粒位置
        chromo_centromere_point = min(intersections, key=lambda x: x["distance"])["midpoint"]
        chromo_centromere_point = (int(chromo_centromere_point[0]), int(chromo_centromere_point[1]))

        # 转换到核型图中染色体着丝粒的位置

        # 将单独的染色体上的着丝粒坐标转换到核型图中的坐标
        centromere_point_in_kty = (
            chromo_centromere_point[0] + delta_x,
            chromo_centromere_point[1] + delta_y,
        )

        # 在核型图中标记染色体着丝粒
        if DRAW_CENTROMERE:
            cv2.circle(canvas, centromere_point_in_kty, 3, (0, 255, 0), -1)
        # 在核型图中把染色体轮廓画出来
        if DRAW_CONTOUR:
            cv2.drawContours(canvas, [chromo_cntr_dict['cntr']], -1, (0, 0, 255), 1)

    # 保存标记了染色体着丝粒的核型图
    kyt_img_dir, kyt_img_fn = os.path.split(kyt_img_fp)
    kyt_img_fbasename = os.path.splitext(kyt_img_fn)[0]
    net_img_fp = f"{os.path.join(OUTPUT_DIR_FP, kyt_img_fbasename)}_centromere.jpg"
    cv_imwrite(net_img_fp, canvas)
    print(f"Saved: {net_img_fp}")