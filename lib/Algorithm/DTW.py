import gc

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from tslearn.metrics import ctw_path

def DTWDistance(s1, s2, method='fdtw'):
    """

    :param s1: seq1
    :param s2: seq2
    :param method: 'fdtw' for fast DTW
                    'ctw' for CTW (Canonical Time Warping)
    :return:
    """
    if method == 'fdtw':
        distance, path = fastdtw(s1, s2, dist=euclidean)
        return distance, path

    elif method == 'ctw':
        path, CCA, distance = ctw_path(s1, s2)
        return distance, path

    else:
        raise ValueError("Cannot identify the name of the method (in calculating distance between seq.s).")

def DTWDistance_bin(s1, s2):
    centromere1 = s1 == 9
    indices1 = np.where(centromere1)[0]
    centromere1 = indices1[0]
    centromere2 = s2 == 9
    indices2 = np.where(centromere2)[0]
    centromere2 = indices2[0]

    if centromere1 < 3:
        centromere1 += 2
    if centromere1 > len(s1) - 4:
        centromere1 -= 2
    if centromere2 < 3:
        centromere2 += 2
    if centromere2 > len(s1) - 4:
        centromere2 -= 2

    s1_1 = s1[:centromere1]
    s1_2 = s1[centromere1+1:]
    s2_1 = s2[:centromere2]
    s2_2 = s2[centromere2+1:]
    distance1, path1 = fastdtw(s1_1, s2_1, dist=euclidean)
    distance2, path2 = fastdtw(s1_2, s2_2, dist=euclidean)
    return distance1 + distance2

