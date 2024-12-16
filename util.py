import gc
import os
import pickle
import shutil
import time
import json

import banding_pattern_extraction as bpe
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import KMeans as kmeans
from lib.Algorithm.DTW import *
from lib.dshw.chromosome_band_forecast import chromosome_band_forecast
from lib.find_pattern import *

'''
def test(img):
    result = bpe.get_banding_pattern(img)
    if not result['error']:
        band_pattern = result['banding_pattern']

        print(len(band_pattern))
        plt.barh(range(len(band_pattern)),
                 band_pattern)
        plt.show()
    else:
        print('Error in get_banding_pattern!!')
'''

def cv_imread(file_path, mode=cv2.IMREAD_COLOR):
    """读取带中文路径的图片文件
    Args:
        file_path (_type_): _description_
    Returns:
        _type_: _description_
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), mode)  # 前值 cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED


def cv_imwrite(file_path, img):
    """保存带中文路径的图片文件
    Args:
        file_path (_type_): _description_
        img (_type_): _description_
    """
    cv2.imencode(".png", img)[1].tofile(file_path)


def get_result_from_BPE(img):
    return bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def get_length(img):
    return len(bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['banding_pattern'])


def get_skeleton(img, SIZE=(400, 400), stage=5):
    if stage == 0:
        skeleton = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['skeleton']
    elif stage == 1:
        skeleton = np.full(shape=SIZE,
                           fill_value=0,
                           dtype='uint8')
        r = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['r']
        c = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['c']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    elif stage == 2:
        skeleton = np.full(shape=SIZE,
                           fill_value=0,
                           dtype='uint8')
        r = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['r_smoothed']
        c = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['c_smoothed']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    elif stage == 3:
        skeleton = np.full(shape=SIZE,
                           fill_value=0,
                           dtype='uint8')
        r = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['r_sample']
        c = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['c_sample']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    elif stage == 4:
        skeleton = np.full(shape=SIZE,
                           fill_value=0,
                           dtype='uint8')
        r = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['r_interpolated']
        c = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['c_interpolated']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    elif stage == 5:
        skeleton = np.full(shape=SIZE,
                           fill_value=0,
                           dtype='uint8')
        r = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['r_sampled']
        c = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['c_sampled']
        for i in range(len(r)):
            skeleton[int(r[i])][int(c[i])] = 255
    else:
        raise ValueError('Unsupported stage value!')

    return skeleton


def get_width(img):
    """

    :param img: BGR image
    :return: a list of width.
    """
    width_2 = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['width']
    for i in range(len(width_2)):
        width_2[i] = width_2[i] * 2
    return width_2


def get_origin_band(img):
    return bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))['banding_pattern']


def find_centromere(abstract_pattern, widths, n, anoamly=False):
    """
    According to ISCN and width to determine the position of centromere
    :param anoamly: if anomaly exists in current chromosome
    :param abstract_pattern:
    :param widths:
    :param n: range from 1 to 24
    :return:
    """
    if anoamly:
        edge = 5
        min_value = min(widths[edge: -edge])
        min_positions = [index for index, value in enumerate(widths[edge: -edge]) if value == min_value]
        return edge + min_positions[int(np.ceil(len(min_positions) / 2))]

    offset_percentage = 0.03
    length = len(widths)
    mapper = {'1': '0.480', '2': '0.384', '3': '0.455', '4': '0.269',
              '5': '0.260', '6': '0.341', '7': '0.349', '8': '0.303',
              '9': '0.327', '10': '0.320', '11': '0.399', '12': '0.259',
              '13': '0.147', '14': '0.149', '15': '0.162', '16': '0.413',
              '17': '0.332', '18': '0.257', '19': '0.453', '20': '0.434',
              '21': '0.311', '22': '0.279', '23': '0.357', '24': '0.317'}
    try:
        centromere_percentage = eval(mapper[str(n)])
    except Exception as e:
        raise ValueError('The id of the chromosome must be a integer from 1 to 24.(23 for x and 24 for y)')

    start = int(length * (centromere_percentage - offset_percentage))
    end = int(length * (centromere_percentage + offset_percentage))
    min_value = min(widths[start: end])
    min_positions = [index for index, value in enumerate(widths[start: end]) if value == min_value]

    return start + min_positions[int(np.ceil(len(min_positions) / 2)) - 1]


def find_centromere_and_angle_in_img(result, rect, n):
    '''

    :param n: ID of a chromosome
    :param rect: minimum rectangle of the chromosome
    :param result: bpe result of input image of a single chromosome
    :return: tuple, position of the centromere
    '''
    # find position (index) of a chromosome
    r = result['r_sampled']  # y
    c = result['c_sampled']  # x
    width = result['width']
    abpt = np.full(shape=(len(width),), fill_value=2)
    centro = find_centromere(abpt, width, n + 1)

    # find the angle of the perpendicular line of chromosome.
    center_x, center_y = rect[0]
    w, h = rect[1]
    theta = rect[2]
    # in case the rectangle is casually rotated.
    if 80 < theta < 100:
        h, w = w, h
    x_start = c[centro - 3] if centro - 3 >= 0 else c[centro]
    x_end = c[centro + 3] if centro + 3 <= len(r) - 1 else c[centro]
    y_start = r[centro - 3] if centro - 3 >= 0 else r[centro]
    y_end = r[centro + 3] if centro + 3 <= len(r) - 1 else r[centro]
    perpendicular = (x_end - x_start, y_end - y_start)
    std = (1, 0)
    angle = np.degrees(np.arccos(np.dot(perpendicular, std) / (np.linalg.norm(perpendicular) * np.linalg.norm(std))))

    return ((c[centro] - center_x + w / 2) / w, (r[centro] - center_y + h / 2) / h), angle - 90


def gray_band_reconstruct(abstract_pattern, banding_count):
    """

    :param banding_count: count of bands
    :param abstract_pattern: abstract pattern without centermere info.
    :return: modified abstract pattern
    """

    # record pattern imformation (type, length, start of each band)
    pattern_info = np.zeros(shape=(banding_count, 3), dtype='int32')
    first_of_a_band = True
    count = -1
    for i in range(len(abstract_pattern)):
        if first_of_a_band:
            count += 1
            pattern_info[count][0] = abstract_pattern[i]
            pattern_info[count][1] += 1
            pattern_info[count][2] = i
            first_of_a_band = False
        else:
            pattern_info[count][1] += 1
        if i != len(abstract_pattern) - 1 and abstract_pattern[i] != abstract_pattern[i + 1]:
            first_of_a_band = True

    # modify gray bands (1\2 in 0\1\2\3) according to the pattern info. and the rules:
    # 1. If only a '1' shorter than 16 is between two '0's, make it '2'.
    # 2. If only a '2' shorter than 16 is between two '3's, make it '1'.
    # 3. If only a '1' shorter than 16 is between two '3's, make it '0'.
    # 4. Do nothing to the first and end bands.
    for i in range(1, banding_count - 1):
        prev_color = pattern_info[i - 1][0]
        next_color = pattern_info[i + 1][0]
        curr_color = pattern_info[i][0]
        curr_length = pattern_info[i][1]

        if prev_color == 0 and next_color == 0 \
                and curr_color == 1 and curr_length < 16:
            for j in range(pattern_info[i][1]):
                abstract_pattern[pattern_info[i][2] + j] = curr_color + 1

        if prev_color == 3 and next_color == 3 \
                and curr_color != 0 and curr_length < 16:
            for j in range(pattern_info[i][1]):
                abstract_pattern[pattern_info[i][2] + j] = curr_color - 1


def imitate(img, n, reconstruction=False, erode=True, for_demo=True):
    '''
    visualize the band of a chromosome.

    :param img: result of cv2.imread
           n: integer, id of a chromosome
           reconstruction(optional): whether to preserve width info when 'for_demo'
           erode(optional): whether to erode the image before processing.
           for_demo(optional): if true, this function will return a useful canvas.
                               if false, this function can be faster with a useless canvas returned.
    :return: canvas: the result of imitation.
            abstract_pattern: 0\1\2\3 for band classification(3 for white) and 9 for centromere.
            coord: centromere coordinate
            angle: centermere rotate angle
            banding_count: number of bands
    '''
    '''
    需要：只在这个函数中计算一次骨架（即只调用一次get_banding_pattern函数）
    1、条带信息
    2、着丝粒坐标（最小矩形框 比例）
    3、着丝粒处的垂线角度
    4、条带数
    '''
    if erode:
        kernel = np.ones((4, 4), dtype=np.uint8)
        img = cv2.erode(img, kernel=kernel)

    time_s = time.time()
    # get result from Banding-Pattern-Extraction
    result = bpe.get_banding_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if result['error']:
        raise ValueError('Failed to get banding pattern..')

    band_pattern = result['banding_pattern']  # ndarray of rank1.
    widths = result['width']

    band_pattern = filter_(band_pattern)  # filter

    # widths, band_pattern = pop_white(widths, band_pattern)  # dealing with the start and end part
    milestones = get_milestone(band_pattern)

    abstract_pattern = []
    canvas = np.full(shape=(cfg.windoww, int(cfg.windowh / 2), 3), fill_value=255, dtype='uint8')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 253, 255, cv2.THRESH_BINARY_INV)
    c, h = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # only the maximum contour is needed
    countour_i = 0
    for i in range(1, len(c)):
        if cv2.contourArea(c[i]) > cv2.contourArea(c[countour_i]):
            countour_i = i
    c = c[countour_i]

    if for_demo:
        extTop = c[c[:, :, 1].argmin()][0]
        extBot = c[c[:, :, 1].argmax()][0]
        rec_start = 10
        rec_end = 50 + np.max(widths)
        # top of the likelyhood- and the chromosome should be aligned
        vert_start = int(extTop[1])

        # draw the rectangle (containing the chromosome).
        if not reconstruction:
            cv2.rectangle(canvas,
                          (rec_start - 1, vert_start - 1),
                          (rec_end - 39, vert_start + len(band_pattern) + 1),
                          color=cfg.rect_color)
        else:
            cv2.rectangle(canvas,
                          (rec_start - 1, vert_start - 1),
                          (rec_end + 1, vert_start + len(band_pattern) + 1),
                          color=cfg.rect_color)

    # get band pattern
    for i in range(len(band_pattern)):
        for j in range(cfg.color):
            if milestones[j] < band_pattern[i] <= milestones[j + 1]:
                abstract_pattern.append(j)

    # statistic of the number of bands
    banding_count = 1
    for i in range(len(abstract_pattern) - 1):
        if abstract_pattern[i] != abstract_pattern[i + 1]:
            banding_count += 1

    # modify abstract pattern
    gray_band_reconstruct(abstract_pattern, banding_count)
    banding_count = 1
    for i in range(len(abstract_pattern) - 1):
        if abstract_pattern[i] != abstract_pattern[i + 1]:
            banding_count += 1

    # draw the bands
    if for_demo:
        if not reconstruction:
            for i in range(len(abstract_pattern)):
                cv2.line(canvas,
                         (rec_start, vert_start + i),
                         (rec_end - 40, vert_start + i),
                         color=eval('cfg.color' + str(abstract_pattern[i] + 1)))
        else:
            for i in range(len(abstract_pattern)):
                cv2.line(canvas,
                         (int((rec_start + rec_end) / 2 - widths[i]), vert_start + i),
                         (int((rec_start + rec_end) / 2 + widths[i]), vert_start + i),
                         color=(int(band_pattern[i]), int(band_pattern[i]), int(band_pattern[i])))
    else:
        pass

    # find the centromere and draw
    if n != -1:
        centromere = find_centromere(abstract_pattern=abstract_pattern,
                                     widths=widths,
                                     n=n + 1)
        abstract_pattern[centromere] = 9

        if for_demo:
            if not reconstruction:
                cv2.arrowedLine(canvas,
                                (rec_end + 10, vert_start + centromere),
                                (rec_end - 45, vert_start + centromere),
                                color=cfg.color_spec,
                                thickness=5)
            else:
                cv2.arrowedLine(canvas,
                                (rec_end + 60, vert_start + centromere),
                                (rec_end, vert_start + centromere),
                                color=cfg.color_spec,
                                thickness=5)

        # get centromere position in the image and angle of perpendicular line of chromosome
        rect = cv2.minAreaRect(c)
        coord, angle = find_centromere_and_angle_in_img(result, rect, n)
    else:
        coord, angle = -1000, -1000

    time_e = time.time()

    del result
    del gray
    del binary
    gc.collect()
    '''
    plt.plot(abstract_pattern, color='#aaaaaa')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('gray scale value', fontsize=15)
    plt.ylabel('chromosome sequence', fontsize=15)
    plt.show()
    return'''
    return canvas, abstract_pattern, coord, angle, banding_count


def form_new_dataset(n, dt_path, new_dt_path):
    '''
    according to the result of clustering:
    Given a dataset of one type of chromosome, create the clustered new dataset.
    '''
    os.mkdir(new_dt_path)
    for i in range(cfg.cluster):
        os.mkdir(new_dt_path + '/' + str(i))

    with open(cfg.model_dir + str(n) + '.pickle', 'rb') as f:
        cluster = pickle.load(f)
    for item in os.listdir(dt_path):
        feature = kmeans.image_to_data(cv2.imread(dt_path + '/' + item, cv2.IMREAD_GRAYSCALE))
        feature = np.array(feature)
        feature = np.reshape(feature, newshape=(1, 1))
        # a image is classified according to its feature
        dists = []
        for i in range(cfg.cluster):
            dists.append(np.linalg.norm(cluster.cluster_centers_[i] - feature[0]))
        dists = np.array(dists)
        label = np.argmin(dists)
        shutil.copy(dt_path + '/' + item, new_dt_path + '/' + str(label) + '/' + item)


def sample_from_clustered(clustered_dt_path, target_sample_path):
    selected_features = []
    selected_names = []

    def can_select(f):
        s = True
        for item in selected_features:
            if abs(f - item) < 0.1:
                s = False
                break
        return s

    for item in os.listdir(clustered_dt_path):
        feature = kmeans.image_to_data(cv2.resize(cv2.imread(clustered_dt_path + '/' + item, cv2.IMREAD_GRAYSCALE),
                                                  dsize=(cfg.windoww, int(cfg.windowh / 2))))
        if can_select(feature):
            selected_features.append(feature)
            selected_names.append(item)

    os.makedirs(target_sample_path)
    interval = int(len(selected_names) / 16)
    interval = max(interval, 1)
    count = 0

    sorted_pairs = sorted(zip(selected_features, selected_names))
    selected_features, selected_names = zip(*sorted_pairs)
    for item in selected_names:
        count += 1
        if count % interval == 0:
            shutil.copy(clustered_dt_path + '/' + item, target_sample_path + '/' + item)


def get_DTWdist_from_sequence(seq1, seq2, method='ctw'):
    seq1 = np.reshape(np.array(seq1), (len(seq1), 1))
    seq2 = np.reshape(np.array(seq2), (len(seq2), 1))
    return DTWDistance(seq1, seq2, method=method)[0]


def get_DTWdistpath_from_sequence(seq1, seq2, method='fdtw'):
    seq1 = np.reshape(np.array(seq1), (len(seq1), 1))
    seq2 = np.reshape(np.array(seq2), (len(seq2), 1))
    return DTWDistance(seq1, seq2, method=method)[1]


def get_DTWdist_from_img_path(path1, path2, n):
    _, abstract_pattern1, _, _, _ = imitate(cv2.resize(cv2.imread(path1),
                                                       dsize=(cfg.windoww, int(cfg.windowh / 2))),
                                            n=n,
                                            for_demo=False)
    _, abstract_pattern2, _, _, _ = imitate(cv2.resize(cv2.imread(path2),
                                                       dsize=(cfg.windoww, int(cfg.windowh / 2))),
                                            n=n,
                                            for_demo=False)

    s1 = np.reshape(np.array(abstract_pattern1), newshape=(len(abstract_pattern1), 1))
    s2 = np.reshape(np.array(abstract_pattern2), newshape=(len(abstract_pattern2), 1))

    del abstract_pattern1
    del abstract_pattern2
    gc.collect()
    dist, path = DTWDistance(s1, s2)
    return dist


def get_DTWBINdist_from_img_path(path1, path2, n):
    _, abstract_pattern1, _, _, _ = imitate(cv2.resize(cv2.imread(path1),
                                                       dsize=(cfg.windoww, int(cfg.windowh / 2))),
                                            n=n,
                                            for_demo=False)
    _, abstract_pattern2, _, _, _ = imitate(cv2.resize(cv2.imread(path2),
                                                       dsize=(cfg.windoww, int(cfg.windowh / 2))),
                                            n=n,
                                            for_demo=False)

    s1 = np.reshape(np.array(abstract_pattern1), newshape=(len(abstract_pattern1), 1))
    s2 = np.reshape(np.array(abstract_pattern2), newshape=(len(abstract_pattern2), 1))

    del abstract_pattern1
    del abstract_pattern2
    gc.collect()
    return DTWDistance_bin(s1, s2)


def get_dtw_threshold_from_sample(sample_path, n):
    dtw_dists = []
    features = []
    sample_names = os.listdir(sample_path)

    for i in range(len(sample_names)):
        features.append(kmeans.image_to_data(cv2.imread(sample_path + '/' + sample_names[i],
                                                        cv2.IMREAD_GRAYSCALE)))

    sorted_pairs = sorted(zip(features, sample_names))
    features, sample_names = zip(*sorted_pairs)
    for i in [0, len(sample_names) - 1]:
        for j in range(1, len(sample_names) - 1):
            dtw_dists.append(get_DTWdist_from_img_path(sample_path + '/' + sample_names[i],
                                                       sample_path + '/' + sample_names[j],
                                                       n))
    dtw_dists.append(get_DTWdist_from_img_path(sample_path + '/' + sample_names[0],
                                               sample_path + '/' + sample_names[len(sample_names) - 1],
                                               n))

    f = open(sample_path + '/' + 'dtw_dist.txt', 'w')
    f.write(str(np.max(dtw_dists)))
    f.close()

    del dtw_dists
    del features
    del sample_names
    gc.collect()


def get_dtw_bin_threshold_from_sample(sample_path, n):
    dtw_dists = []
    features = []
    sample_names = os.listdir(sample_path)

    for i in range(len(sample_names)):
        if sample_names[i][-4:] == '.txt':
            continue
        features.append(kmeans.image_to_data(cv2.imread(sample_path + '/' + sample_names[i],
                                                        cv2.IMREAD_GRAYSCALE)))

    sorted_pairs = sorted(zip(features, sample_names))
    features, sample_names = zip(*sorted_pairs)
    for i in [0, len(sample_names) - 1]:
        for j in range(1, len(sample_names) - 1):
            dtw_dists.append(get_DTWBINdist_from_img_path(sample_path + '/' + sample_names[i],
                                                          sample_path + '/' + sample_names[j],
                                                          n))
    dtw_dists.append(get_DTWBINdist_from_img_path(sample_path + '/' + sample_names[0],
                                                  sample_path + '/' + sample_names[len(sample_names) - 1],
                                                  n))

    f = open(sample_path + '/' + 'dtw_bin_dist.txt', 'w')
    f.write(str(np.max(dtw_dists)))
    f.close()

    del dtw_dists
    del features
    del sample_names
    gc.collect()


def interpolate_array(array, new_length):
    x = np.arange(len(array))
    f = interp1d(x, array, kind='cubic')
    new_x = np.linspace(0, len(array) - 1, new_length)
    interpolated_array = f(new_x)

    return interpolated_array


def ignore_centromere(seq):
    seq[seq.index(9)] = seq[seq.index(9) + 1]
    return seq


def get_sequence_and_length(input_path, n):
    '''

    :param input_path: path of input folder
    :param n: id
    :return: seq and len and centromere
    '''
    length = 0
    centromere_percent = 0
    sequence = []
    bands = []
    for sample in os.listdir(input_path):
        if sample[-3:] == 'jpg' or sample[-3:] == 'png':
            _, band, _, _, _ = imitate(cv2.resize(cv2.imread(input_path + '/' + sample),
                                                  dsize=(cfg.windoww, int(cfg.windowh / 2))),
                                       n=n,
                                       for_demo=False)
            length += len(band)
            centromere = band.index(9)
            centromere_percent += centromere / len(band)
            band[centromere] = band[centromere + 1]
            bands.append(band)

    length = int(length / len(os.listdir(input_path)))
    centromere_percent = centromere_percent / len(os.listdir(input_path))
    for band in bands:
        sequence.extend(interpolate_array(band, length))

    del bands
    gc.collect()

    return sequence, length, int(centromere_percent * length)


def get_band_template(seq, length, centromere):
    template = chromosome_band_forecast(length, seq)

    template = filter_(template)
    milestones = get_milestone_for_template(template) + [10]
    new_template = []
    for i in range(len(template)):
        for j in range(len(milestones)):
            if milestones[j] > template[i]:
                new_template.append(j)
                break
    template = new_template
    template[centromere] = 9

    return template


def draw_template_and_save(template, target_path):
    np.save(target_path + '/template.npy', template)
    canvas = np.full(shape=(int(cfg.windowh / 2), cfg.windoww, 3), fill_value=255, dtype='uint8')
    xstart = 130
    ystart = 25
    xoffset = 40
    centromere = template.index(9)
    template[centromere] = template[centromere + 1]

    cv2.rectangle(canvas,
                  (xstart - 1, ystart - 1),
                  (xstart + xoffset + 1, ystart + len(template) + 1),
                  color=cfg.rect_color)
    for i in range(len(template)):
        cv2.line(canvas,
                 (xstart, ystart + i),
                 (xstart + xoffset, ystart + i),
                 color=eval('cfg.color' + str(template[i] + 1)))
    cv2.arrowedLine(canvas,
                    (xstart + xoffset + 43, ystart + centromere),
                    (xstart + xoffset + 3, ystart + centromere),
                    color=cfg.color_spec,
                    thickness=5)

    banding_count = 1
    for i in range(len(template) - 1):
        if template[i] != template[i + 1]:
            banding_count += 1
    cv2.putText(canvas, f'band:{banding_count}', (20, 20),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0))
    cv2.imwrite(target_path + '/template.png', canvas)


def fourier2d_filter(img, param=8):
    '''

    :param img: image of a chromosome
    :return: shifted FT result, processed FT result, and high-pass filtered image
    '''
    # 傅里叶变换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_result = np.fft.fft2(img)
    f_shift_result = np.fft.fftshift(f_result)
    # 高通滤波，即对频域图进行变换
    # TODO: 高通滤波即裁剪掉频域图中低频的部分，裁剪方式可以考虑更改。
    img_f_shift = f_shift_result.copy()  # 避免浅拷贝
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    img_f_shift[crow - param: crow + param, ccol - param: ccol + param] = 0
    # 对变换后的频域图进行逆变换，得到高通滤波的图像。
    ishift = np.fft.ifftshift(img_f_shift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    iimg = 255 - iimg

    return np.log(np.abs(f_shift_result)), np.log(np.abs(img_f_shift)), iimg


def make_img2seq_dataset():
    '''
    计划将所有图像都放在一个文件夹内，重命名为n_idx.png以解决重名问题。
    将{n_idx.png: seq}作为一个键值对保存到json文件中，这样方便查询。
    '''
    img_dataset_path = '/home/kms/wangyu_dtst/single_data__for_band'
    img2seq_dataset_path = '/home/kms/wangyu_dtst/img2seq_V2'  # /imgs/  /labels.json
    labels = {}  # 存储名称：序列的键值对。一个元素是一个键值对。

    for i in range(24):  # 0-23, denoting 1-22 and xy
        print(str(i) + ' / 23 process..')
        root_path = img_dataset_path + '/' + str(i)
        img_names = os.listdir(root_path)
        for img_name in img_names:
            img_path = root_path + '/' + img_name
            seq = imitate(cv2.resize(cv2.imread(img_path), dsize=(400, 400)),
                          n=i, for_demo=False)[1]

            # 得到序列后，还应该padding，去掉着丝粒位置。
            if len(seq) >= 400:
                continue
            seq = ignore_centromere(seq)
            paddings = [8] * (400 - len(seq))
            seq += paddings

            # 现在有了处理好的序列和图像路径，开始制作数据集。
            new_name = str(i) + '_' + img_path.split('/')[-1]

            target_path = img2seq_dataset_path + '/imgs/' + new_name
            shutil.copy(img_path, target_path)
            labels[new_name] = seq

    # 保存labels
    def new_dump(obj):
        # Convert numpy classes to JSON serializable objects.
        return obj.item()

    with open(img2seq_dataset_path + '/labels.json', 'w') as f:
        json.dump(labels, f, default=new_dump)


# imitate(cv2.resize(cv2.imread('./demo_pic/134.png'), dsize=(400, 400)), -1, for_demo=False)
