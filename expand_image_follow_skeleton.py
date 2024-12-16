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


import cv2
import numpy as np
from skimage.morphology import skeletonize
from segment_util.karyotype import Karyotype
from lib.chromo_cv_utils import contour_bbox_img, normalization_with_contours_mask, find_external_contours_en, get_proper_threshold_4_contours, find_contours_en

def skeletonize_image(image):
    # 将图像转换为二值图像
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 将二值图像转换为0和1
    binary_image = binary_image // 255

    # 使用skimage的skeletonize函数提取骨架
    skeleton = skeletonize(binary_image)

    return skeleton

def expand_along_skeleton(image, skeleton):
    # 获取骨架的坐标
    skeleton_coords = np.column_stack(np.where(skeleton > 0))

    # 创建一个空白的扩展图像
    expanded_image = np.zeros_like(image)

    # 沿着垂直方向扩展图像
    for (y, x) in skeleton_coords:
        expanded_image[:, x] = image[:, x]

    return expanded_image

KYT_IMG_FP = r'E:\染色体测试数据\240520-数报告图条带数量\01_src_with_band_num\300-A2308236001.126.K.JPG'

kyt = Karyotype(KYT_IMG_FP)
chromo_dicts_by_cy = kyt.read_karyotype()

# 由于从报告图中解析出的染色体信息不包含染色体图片，需要添加上用于后续处理
# 保存染色体信息的简单结构
chromos = []
for chromo_dicts in chromo_dicts_by_cy.values():
    for chromo_dict in chromo_dicts:
        chromo_bbox_bbg, chromo_bbox_wbg = contour_bbox_img(kyt.img["img"], chromo_dict["cntr"])
        chromo_dict["bbox_bbg"] = chromo_bbox_bbg
        chromo_dict["bbox_wbg"] = chromo_bbox_wbg
        chromos.append(chromo_dict)

# 取一根染色体
chromo = chromos[36]
chromo_bbox_bbg = chromo["bbox_bbg"]
chromo_bbox_wbg = chromo["bbox_wbg"]
chromo_cntr = find_external_contours_en(chromo_bbox_wbg, bin_thresh=-1, bin_thresh_adjustment=0)[0]

# 单根染色体条带增强
chromo_bbox_bbg_norm = normalization_with_contours_mask(chromo_bbox_bbg, chromo_cntr, 46, 253)
chromo_bbox_wbg_norm = normalization_with_contours_mask(chromo_bbox_wbg, chromo_cntr, 46, 253)

image = chromo_bbox_wbg_norm
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

threshold = get_proper_threshold_4_contours(image)
_, bin_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

# 获取图像骨架
skeleton = skeletonize_image(bin_image)

# 沿骨架的垂直方向扩展图像
expanded_image = expand_along_skeleton(image, skeleton)

# 显示原图像、骨架图像和扩展后的图像
cv2.imshow("Original Image", image)
cv2.imshow("Binary Image", bin_image)
cv2.imshow("Skeleton", skeleton.astype(np.uint8) * 255)  # 将骨架图像转换为0和255显示
cv2.imshow("Expanded Image", expanded_image)

# 等待用户按下任意键并退出
cv2.waitKey(0)
cv2.destroyAllWindows()
