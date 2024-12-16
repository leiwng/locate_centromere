# -*- coding: utf-8 -*-
"""对核型图中的染色体进行归一化处理（归一化+直方图均衡化），
并对比处理结果

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

from segment_util.karyotype import Karyotype
from lib.chromo_cv_utils import cv_imread, cv_imwrite, normalization_with_contours_mask


karyogram_dir_fp = r'E:\染色体测试数据\240520-数报告图条带数量\01_src_with_band_num'
output_root_dir_fp = r'E:\染色体测试数据\240520-数报告图条带数量\02_equalizehist_normalization'

for karyogram_fn in os.listdir(karyogram_dir_fp):

    print(karyogram_fn)

    karyogram_fp = os.path.join(karyogram_dir_fp, karyogram_fn)
    karyogram_img = cv_imread(karyogram_fp)

    # 取染色体轮廓
    karyotype = Karyotype(karyogram_fp)
    chromo_dicts_orgby_cy = karyotype.read_karyotype()
    chromo_cntrs = [cntr["cntr"] for cntrs in chromo_dicts_orgby_cy.values() for cntr in cntrs]

    # 将染色体提取出来，从新保存到白色背景的图片中
    # 创建一个与原图大小相同的全黑掩膜
    mask = np.zeros_like(karyogram_img)
    # 在掩膜上绘制所有轮廓
    cv2.drawContours(mask, chromo_cntrs, -1, (255, 255, 255), thickness=cv2.FILLED)
    # 将原图中的所有轮廓区域抠取出来,染色体在黑背景上
    objects_roi = cv2.bitwise_and(karyogram_img, mask)
    # 将抠取出来的物体叠加到白色背景图像上
    white_background = cv2.bitwise_not(mask)
    white_background = cv2.bitwise_or(white_background, objects_roi)

    # 争对轮廓掩码继续归一化处理
    karyogram_normalized = normalization_with_contours_mask(white_background, chromo_cntrs, 27, 252)

    # 将原图拷贝到输出目录
    output_fp = os.path.join(output_root_dir_fp, karyogram_fn)
    cv_imwrite(output_fp, karyogram_img)
    # 将归一化处理后的图像拷贝到输出目录
    # 目标文件名：原文件名_normalized源文件名后缀
    output_fp_normalized = os.path.join(output_root_dir_fp, f'{os.path.splitext(karyogram_fn)[0]}_normalized{os.path.splitext(karyogram_fn)[1]}')
    cv_imwrite(output_fp_normalized, karyogram_normalized)
