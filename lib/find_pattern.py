import numpy as np
from scipy.ndimage import gaussian_filter1d
from . import Config as cfg
from sklearn.cluster import KMeans


def filter_(banding_pattern,
            break_iter=100000, smooth_iter=0):
    '''

    :param smooth_iter:
    :param break_iter:
    :param banding_pattern:
    :return: filtered band pattern
    '''
    f_b = gaussian_filter1d(banding_pattern, sigma=2)
    prev_i_b = np.copy(f_b)
    i_b = []
    R = 2
    same_pattern = False
    counter = 0

    while not same_pattern:
        counter += 1
        if counter >= break_iter:
            break

        for i in range(0, len(f_b)):
            crrnt = prev_i_b[i]
            if i == 0:
                prv = crrnt
            else:
                prv = prev_i_b[i - 1]

            if i == len(f_b) - 1:
                nxt = crrnt
            else:
                nxt = prev_i_b[i + 1]

            neighbourhood = [prv, crrnt, nxt]
            dif_min = crrnt - np.min(neighbourhood)
            dif_max = np.max(neighbourhood) - crrnt

            if dif_max <= dif_min:
                i_b.append(crrnt + dif_max / R)
            else:
                i_b.append(crrnt - dif_min / R)

        R = 1
        same_pattern = np.all(prev_i_b == i_b)
        prev_i_b = i_b
        i_b = []
        if counter < smooth_iter:
            prev_i_b = gaussian_filter1d(prev_i_b, sigma=2)

    return prev_i_b


# 寻找肘部点对应的聚类数
def find_optimal_clusters_elbow_method(wcss):
    x1, y1 = 1, wcss[0]
    x2, y2 = len(wcss), wcss[-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 1
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    # return distances.index(max(distances)) + 1
    # test for paper
    return 5


def get_best_clusters_elbow_method(band_pattern):
    '''
    :param band_pattern:
    :return: best clusters for classify band.
    '''
    # band_pattern = np.reshape(band_pattern, newshape=(len(band_pattern), 1))
    # 计算不同聚类数对应的损失函数值
    # print("band_pattern:", band_pattern)
    # To avoid the warning: Number of distinct clusters (29) found smaller than n_clusters (30). Possibly due to duplicate points in X.
    n_clusters = 17
    wcss = []
    # for i in range(1, len(band_pattern)+1):
    for i in range(1, n_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(band_pattern)
        wcss.append(kmeans.inertia_)

    return find_optimal_clusters_elbow_method(wcss)


def get_milestone(band_pattern, gray_level_4_band=4):
    '''

    :param band_pattern:
    :return: milestone for classify band.
    '''
    # 这里kmeans聚五类而不是四类。是希望条带图中白的部分多一些，更符合人类直觉。
    # kmeans = KMeans(n_clusters=5, n_init=10)

    band_pattern = np.reshape(band_pattern, newshape=(len(band_pattern), 1))

    # 根据最优聚类数进行聚类
    # kmeans = KMeans(n_clusters=get_best_clusters_elbow_method(band_pattern), n_init=10)
    # 根据最优聚类数进行聚类不符合实际情况，使用固定的聚类数
    kmeans = KMeans(n_clusters=gray_level_4_band, n_init=10)
    cluster = kmeans.fit(band_pattern)
    centers = cluster.cluster_centers_
    centers = np.reshape(centers,
                         newshape=(len(centers),))
    centers = np.sort(centers)
    n_clusters = len(centers)

    milestones = [0]
    for i in range(1, n_clusters):
        milestones.append((centers[i - 1] + centers[i]) / 2)
    milestones.append(255)

    # milestones = [0,
    #               (centers[1] + centers[2]) / 2,
    #               (centers[2] + centers[3]) / 2,
    #               (centers[3] + centers[4]) / 2,
    #               255]

    return milestones

def get_milestone_for_template(band_pattern, gray_level_4_band=4):
    '''

    :param band_pattern:
    :return: milestone for classify band.
    '''
    band_pattern = np.reshape(band_pattern, newshape=(len(band_pattern), 1))

    # kmeans = KMeans(n_clusters=get_best_clusters_elbow_method(band_pattern), n_init=10)
    # 根据最优聚类数进行聚类不符合实际情况，使用固定的聚类数
    kmeans = KMeans(n_clusters=gray_level_4_band, n_init=10)
    cluster = kmeans.fit(band_pattern)
    centers = cluster.cluster_centers_
    centers = np.reshape(centers,
                         newshape=(len(centers),))
    centers = np.sort(centers)
    n_clusters = len(centers)

    milestones = []
    for i in range(1, n_clusters):
        milestones.append((centers[i - 1] + centers[i]) / 2)

    # milestones = [(centers[1] + centers[2]) / 2,
    #               (centers[2] + centers[3]) / 2,
    #               (centers[3] + centers[4]) / 2]

    return milestones

def pop_white(widths, band_pattern):
    '''

    :param band_pattern:
    :return: processed band pattern
    '''
    new_bp = np.copy(band_pattern)
    new_w = np.copy(widths)

    start_del = [0]
    start_i = 0
    while band_pattern[start_i] == band_pattern[start_i + 1]:
        start_i += 1
        start_del.append(start_i)
    new_bp = np.delete(new_bp, start_del)
    new_w = np.delete(new_w, start_del)

    end_del = [len(new_bp) - 1]
    end_i = len(new_bp) - 1
    while new_bp[end_i] == new_bp[end_i - 1]:
        end_i -= 1
        end_del.append(end_i)
    new_bp = np.delete(new_bp, end_del)
    new_w = np.delete(new_w, end_del)

    return new_w, new_bp
