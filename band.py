import os
import sys

import banding_pattern_extraction as bpe
import cv2
import matplotlib.pyplot as plt
import numpy as np

from lib.find_pattern import *
from util import get_skeleton

SIZE = (400, 400)

def img2bandseq(chromo_img, gray_level_4_band=4):
    """

    :param chromo_img: chromo_img = cv2.imread('path/to/image')
    :return: band sequence
    """
    # 由于224*224*3的图像中染色体的像素点少，求得的直线误差较大，所以将染色体放大到400*400
    # chromo_img = cv2.resize(chromo_img, SIZE)

    # 对染色体进行腐蚀，消除边缘上模糊的地方。
    kernel = np.ones((3, 3), dtype=np.uint8)
    chromo_img = cv2.erode(chromo_img, kernel=kernel)

    # 得到染色体一端到另一端的条带的灰度值“band_pattern”。具体方法被封装在band_pattern中
    result = bpe.get_banding_pattern(cv2.cvtColor(chromo_img, cv2.COLOR_BGR2GRAY))
    if result['error']:
        raise ValueError('Failed to get banding pattern..')
    band_pattern = result['banding_pattern']  # ndarray of rank1.

    # 非线性滤波
    band_pattern = filter_(band_pattern)

    # 获得条带“abstract_pattern”
    milestones = get_milestone(band_pattern, gray_level_4_band=gray_level_4_band)  # K-means聚类结果，即四种颜色的分界线
    abstract_pattern = []
    for i in range(len(band_pattern)):
        for j in range(len(milestones) - 1):
            if milestones[j] < band_pattern[i] <= milestones[j + 1]:
                abstract_pattern.append(j)
                break

    return abstract_pattern

def visualization4skeleton(chromo_img, stage, bold=True, save_path=''):
    """
    :param chromo_img: cv2.imread
    :param stage:
        0：细化算法的结果
        1：剪枝后的结果
        2：采样结果（点）
        3：采样结果延长（点）
        4：skeleton最终结果 （一端到另一端的skeleton）
        5：作垂线
    :param bold: 是否对skeleton进行加粗，方便展示
    :param save_path: 保存路径。为空则不保存，直接使用matplotlib展示；若指定保存路径，则不展示，直接保存。
    :return: None
    """
    assert stage in [0, 1, 2, 3, 4, 5]
    # 由于224*224*3的图像中染色体的像素点少，求得的直线误差较大，所以将染色体放大到400*400
    chromo_img_size = chromo_img.shape[:2]
    chromo_img = cv2.resize(chromo_img, chromo_img_size)

    # 对染色体进行腐蚀，消除边缘上模糊的地方。
    kernel = np.ones((3, 3), dtype=np.uint8)
    chromo_img = cv2.erode(chromo_img, kernel=kernel)

    # 根据不同的stage获得不同的skeleton
    result = bpe.get_banding_pattern(cv2.cvtColor(chromo_img, cv2.COLOR_BGR2GRAY))
    if stage == 0:
        skeleton = result['skeleton']
    elif stage == 1:
        skeleton = np.full(shape=chromo_img_size,
                           fill_value=0,
                           dtype='uint8')
        r = result['r']
        c = result['c']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    elif stage == 2:
        skeleton = np.full(shape=chromo_img_size,
                           fill_value=0,
                           dtype='uint8')
        r = result['r_sample']
        c = result['c_sample']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    elif stage == 3:
        skeleton = np.full(shape=chromo_img_size,
                           fill_value=0,
                           dtype='uint8')
        r = result['r_interpolated']
        c = result['c_interpolated']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    else:  # stage = 4 or 5
        skeleton = np.full(shape=chromo_img_size,
                           fill_value=0,
                           dtype='uint8')
        r = result['r_sampled']
        c = result['c_sampled']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    skeleton = np.stack((skeleton, skeleton, skeleton), axis=2)

    # 对染色体skeleton进行加粗，调整颜色
    if bold:
        skeleton = cv2.dilate(skeleton, kernel=(np.ones((3, 3), dtype=np.uint8)))
    skeleton[:, :, 0] = 0
    skeleton[:, :, 1][skeleton[:, :, 1] != 0] = 200
    skeleton[:, :, 2][skeleton[:, :, 2] != 0] = 200

    chromo_img = cv2.add(chromo_img, skeleton)

    # 如果是最后一个阶段，把垂线也画上。
    if stage == 5:
        pstarts, pends = result['pstart'], result['pend']  # 垂线的起点和终点
        for i in range(len(pends)):
            if i % 5 == 0:  # 便于展示，每五条直线画一条。
                cv2.line(chromo_img, (int(pstarts[i][1]), int(pstarts[i][0])),
                         (int(pends[i][1]), int(pends[i][0])), color=(255, 255, 255))

    # 展示或保存(由于opencv和matplotlib的差异，展示的skeleton是蓝色，保存的skeleton是黄色)
    if save_path:
        cv2.imwrite(save_path, chromo_img)
    else:
        plt.imshow(chromo_img)
        plt.xticks([])
        plt.yticks([])
        plt.show()


def create_color_mapper(num_colors):
    step = 255 // (num_colors - 1) if num_colors > 1 else 0
    color_mapper = {i: (i * step, i * step, i * step) for i in range(num_colors)}
    return color_mapper


def visualization4band(chromo_img, gray_level_4_band):
    """
    可视化图像的条带。
    :param chromo_img: chromo_img = cv2.imread('path/to/chromo_img')
    :param save_path: 保存路径。为空则不保存，直接使用matplotlib展示；若指定保存路径，则不展示，直接保存。
    :return:
    """
    # 获得图像条带序列
    # resize do outside
    # chromo_img = cv2.resize(chromo_img, dsize=SIZE)

    abstract_pattern = img2bandseq(chromo_img, gray_level_4_band=gray_level_4_band)

    chromo_img_size = chromo_img.shape[:2]

    #创建空画布以画条带
    band_canvas = np.full(shape=chromo_img_size, fill_value=255, dtype=np.uint8)
    band_canvas = np.stack([band_canvas, band_canvas, band_canvas], axis=-1)

    # 定义画条带中的参数，包括条带深浅颜色，线宽。
    # color_mapper = {0: (0, 0, 0), 1: (84, 84, 84), 2: (168, 168, 168), 3: (255, 255, 255)}
    color_mapper = create_color_mapper(gray_level_4_band)

    horizontal_start = 0
    horizontal_end = chromo_img_size[1]
    vertical_start = int((chromo_img_size[0] - len(abstract_pattern)) / 2)

    # 画条带，外面用矩形框框住。
    cv2.rectangle(band_canvas,
                  pt1=(horizontal_start - 1, vertical_start - 1),
                  pt2=(horizontal_end + 1, vertical_start + len(abstract_pattern)),
                  color=(0, 125, 200))
    for i in range(len(abstract_pattern)):
        cv2.line(band_canvas,
                 (horizontal_start, vertical_start + i),
                 (horizontal_end, vertical_start + i),
                 color=color_mapper[abstract_pattern[i]])

    # 条带图和原始染色体拼在一起
    chromo_band_img = np.hstack([chromo_img, band_canvas])

    return band_canvas, chromo_band_img

if __name__ == '__main__':

    # 命令行输入：python band.py 单根染色体图片访问路径 条带水平 第几号染色体
    if len(sys.argv) != 4:
        print("Usage: python band.py <path/to/image> <band_level> <gray_level_4_band>")
        sys.exit(1)

    src_img_fp = sys.argv[1]
    if not os.path.exists(src_img_fp):
        print(f"{src_img_fp} does not exist.")
        sys.exit(1)
    # get source image file name, no extension
    src_img_fn = os.path.splitext(os.path.basename(src_img_fp))[0]
    # get full path of the directory of the band image file
    src_img_dir_fp = os.path.dirname(src_img_fp)

    band_level = int(sys.argv[2])
    # 判定band_level是否在合理范围内[300, 350, 400, 450, 500, 550, 700, 850, 900]
    if band_level not in [300, 350, 400, 450, 500, 550, 700, 850, 900]:
        print(f"Invalid band level: {band_level}.")
        sys.exit(1)

    gray_level_4_band = int(sys.argv[3])
    if gray_level_4_band < 1 and gray_level_4_band > 20:
        print(f"{gray_level_4_band} is invalid number.")
        sys.exit(1)

    # chromo_img = cv2.imread(src_img_fp)
    chromo_img = cv2.imdecode(np.fromfile(src_img_fp, dtype=np.uint8), cv2.IMREAD_COLOR) #cv2.IMREAD_COLOR
    assert chromo_img is not None, f"{src_img_fp}, cannot be read."

    band_img, chromo_band_img = visualization4band(chromo_img, gray_level_4_band)

    # save band image and chromo_band image with information of band_level,
    # and key word "band" or "chromo_band" in the file name.
    band_img_fp = os.path.join(src_img_dir_fp, f"{src_img_fn}_{band_level}_band_{gray_level_4_band}c.png")
    chromo_band_img_fp = os.path.join(src_img_dir_fp, f"{src_img_fn}_{band_level}_chromo_band_{gray_level_4_band}c.png")

    cv2.imencode('.png', band_img)[1].tofile(band_img_fp)
    cv2.imencode('.png', chromo_band_img)[1].tofile(chromo_band_img_fp)



