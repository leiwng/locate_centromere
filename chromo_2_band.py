# -*- coding: utf-8 -*-
"""模块注释

Author: Lei Wang
Date: April 24, 2024
"""
__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import os
import sys
import cv2
import numpy as np

from karyotype import Karyotype
from utils.chromo_cv_utils import cv_imread, cv_imwrite, contour_bbox_img
from band import visualization4band


def get_chromo_id(chromo_img_fn):
    # A2308236001.126.K_0X-0_1chromo_wbg
    chromo_id = chromo_img_fn.split('_')[1]
    chromo_id = chromo_id.split('-')[0]
    if chromo_id == '0X':
        chromo_id = 'X'
    elif chromo_id == '0Y':
        chromo_id = 'Y'
    else:
        chromo_id = str(int(chromo_id))
    return chromo_id


def get_idiogram_filename(chromo_img_fn, band_level):
    chromo_id = get_chromo_id(chromo_img_fn)
    return f"{chromo_id}-{band_level}.png"


def get_idiogram_img(chromo_img_fn, band_level, idiogram_dir_fp):
    idiogram_fn = get_idiogram_filename(chromo_img_fn, band_level)
    idiogram_fp = os.path.join(idiogram_dir_fp, idiogram_fn)
    return cv_imread(idiogram_fp, cv2.IMREAD_COLOR)


if __name__ == '__main__':

    # cmd line: python chromo_2_band.py <chromo_img_path> <band_level> <gray_level_4_band> <idiogram_dir>
    if len(sys.argv) != 5:
        print("Usage: python chromo_2_band.py <chromo_img_path> <band_level> <gray_level_4_band> <idiogram_dir>")
        sys.exit(1)

    chromo_img_dir_fp = sys.argv[1]
    if not os.path.exists(chromo_img_dir_fp):
        print(f"{chromo_img_dir_fp} does not exist.")
        sys.exit(1)

    band_level = int(sys.argv[2])
    if band_level not in [300, 350, 400, 450, 500, 550, 700, 850, 900]:
        print(f"Invalid band level: {band_level}.")
        sys.exit(1)

    if not sys.argv[3].isdigit():
        print(f"{sys.argv[3]} is not a digit.")
        sys.exit(1)

    gray_level_4_band = int(sys.argv[3])
    if gray_level_4_band < 1 and gray_level_4_band > 20:
        print(f"{gray_level_4_band} is invalid number.")
        sys.exit(1)

    idiogram_dir_fp = sys.argv[4]
    if not os.path.exists(idiogram_dir_fp):
        print(f"{idiogram_dir_fp} does not exist.")
        sys.exit(1)

    for chromo_img_fn_ext in os.listdir(chromo_img_dir_fp):
        if not chromo_img_fn_ext.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        chromo_img_fp = os.path.join(chromo_img_dir_fp, chromo_img_fn_ext)
        chromo_img_fn = os.path.splitext(chromo_img_fn_ext)[0]

        chromo_img = cv_imread(chromo_img_fp, cv2.IMREAD_COLOR)
        band_img, chromo_band_img = visualization4band(chromo_img, gray_level_4_band)

        # 在chromo_band_img的右边拼接idiogram_img
        print(chromo_img_fn, band_level, gray_level_4_band, idiogram_dir_fp)
        idiogram_img = get_idiogram_img(chromo_img_fn, band_level, idiogram_dir_fp)

        idiogram_img_h, idiogram_img_w = idiogram_img.shape[:2]
        chromo_img_h, chromo_img_w, = chromo_img.shape[:2]

        idiogram_img_new_h = chromo_img_h
        idiogram_img_new_w = idiogram_img_w * idiogram_img_new_h // idiogram_img_h

        new_idiogram_img = cv2.resize(idiogram_img, (idiogram_img_new_w, idiogram_img_new_h),interpolation=cv2.INTER_AREA)

        chromo_band_idiogram_img = np.hstack([chromo_band_img, new_idiogram_img])

        band_img_fp = os.path.join(chromo_img_dir_fp, f"{chromo_img_fn}_{band_level}_band_{gray_level_4_band}c.png")
        chromo_band_img_fp = os.path.join(chromo_img_dir_fp, f"{chromo_img_fn}_{band_level}_chromo_band_{gray_level_4_band}c.png")
        chromo_band_idiogram_img_fp = os.path.join(chromo_img_dir_fp, f"{chromo_img_fn}_{band_level}_chromo_band_idiogram_{gray_level_4_band}c.png")

        cv_imwrite(band_img_fp, band_img)
        cv_imwrite(chromo_band_img_fp, chromo_band_img)
        cv_imwrite(chromo_band_idiogram_img_fp, chromo_band_idiogram_img)


